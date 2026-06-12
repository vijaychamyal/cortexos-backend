import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed import TextEmbedding
from google import genai
from dotenv import load_dotenv
from .config import retrieval_config, collection_name

top_k = retrieval_config.top_k
top_n = retrieval_config.top_n

load_dotenv()

# ── Qdrant ────────────────────────────────────────────────────────────────────

def setup_qdrant():
    try:
        client = QdrantClient(
            url=os.environ.get("QDRANT_CLOUD_URL"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )
        client.get_collections()
        print("[CortexOS] Qdrant connected.")
        return client
    except Exception as e:
        print("Qdrant not connected — check env vars.")
        raise e


# ── Embedding model ───────────────────────────────────────────────────────────

def load_model():
    print("[AI Engine] Loading MiniLM via fastembed...")
    # Single-thread to keep memory low on free tier
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    return TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")


# ── Reranker (lazy — loaded on first query, not at startup) ───────────────────

_reranker_cache = None

def load_reranker():
    """
    Returns a sentinel string instead of actually loading the model at startup.
    The real model is loaded the first time rerank_chunks() is called.
    This saves ~100-150 MB of startup RAM on Render's free tier.
    """
    return "lazy"


def _get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        print("[AI Engine] Loading reranker (first query)...")
        _reranker_cache = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    return _reranker_cache


# ── Gemini (direct google-genai, no LangChain) ────────────────────────────────

def load_llm():
    """
    Returns a configured google-genai Client.
    Replaces ChatGoogleGenerativeAI from langchain-google-genai.
    """
    api_key = os.environ.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    print("[AI Engine] Gemini client ready.")
    return client


RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions strictly based on the provided context.

Context from the document:
{context}

Instructions:
- Answer only based on the context above
- If the answer is not in the context, say "I could not find this information in the document"
- Mention the page number where you found the answer
- Be concise and precise

Question: {question}

Answer:"""


def create_prompt():
    """Returns the prompt template string. No LangChain object needed."""
    return RAG_PROMPT_TEMPLATE


def create_rag_chain(llm_client, prompt_template: str):
    """
    Returns a simple callable dict that bundles the client + template.
    Replaces the LangChain pipe chain (prompt | llm | StrOutputParser).
    Usage: chain.invoke({"context": ..., "question": ...})
    """
    class SimpleChain:
        def __init__(self, client, template):
            self._client = client
            self._template = template

        def invoke(self, inputs: dict) -> str:
            prompt_text = self._template.format(
                context=inputs["context"],
                question=inputs["question"]
            )
            response = self._client.models.generate_content(
                model=retrieval_config.gemini_model,
                contents=prompt_text
            )
            return response.text

    return SimpleChain(llm_client, prompt_template)


# ── Qdrant search ─────────────────────────────────────────────────────────────

def search_qdrant(query, model, client, filename=None, user_id=None):
    query_vector = list(model.embed([query]))[0].tolist()

    must_conditions = []

    if filename:
        must_conditions.append(
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=filename)
            )
        )

    if user_id:
        must_conditions.append(
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id)
            )
        )

    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True
    ).points

    return [
        {
            "text": r.payload["chunk_text"],
            "page_num": r.payload["page_num"],
            "source": r.payload["source"],
            "score": round(r.score, 3)
        }
        for r in results
    ]


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank_chunks(query, chunks, reranker):
    """reranker may be the 'lazy' sentinel string or None — both handled."""
    if not chunks:
        return []

    actual_reranker = _get_reranker()
    documents = [chunk["text"] for chunk in chunks]
    scores = list(actual_reranker.rerank(query, documents))

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = round(float(scores[i]), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# ── Context formatting ────────────────────────────────────────────────────────

def chunks_to_context(chunks):
    parts = [
        f"[Source Page {chunk['page_num']}]\n{chunk['text']}"
        for chunk in chunks
    ]
    return "\n\n---\n\n".join(parts)


# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answer(query, reranked_chunks, chain):
    if not reranked_chunks:
        return "No relevant information found in the document."

    context = chunks_to_context(reranked_chunks)
    return chain.invoke({"context": context, "question": query})


# ── Main query pipeline ───────────────────────────────────────────────────────

def ask_query(query, model, client, reranker, chain, filename=None, user_id=None):
    chunks = search_qdrant(query, model, client, filename=filename, user_id=user_id)
    reranked = rerank_chunks(query, chunks, reranker)
    answer = generate_answer(query, reranked, chain)
    print(f"\nAnswer: {answer}")
    return answer


# ── Interactive dev loop ──────────────────────────────────────────────────────

if __name__ == "__main__":
    client   = setup_qdrant()
    model    = load_model()
    reranker = load_reranker()
    llm      = load_llm()
    prompt   = create_prompt()
    chain    = create_rag_chain(llm, prompt)

    print("Type your question (type 'exit' to quit)")
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == "exit":
            break
        if query:
            ask_query(query, model, client, reranker, chain)