import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed import TextEmbedding
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
from .config import retrieval_config, collection_name

top_k = retrieval_config.top_k
top_n = retrieval_config.top_n

load_dotenv()

# Reranking strongly improves answer quality: Qdrant's raw vector order is
# decent but the cross-encoder re-scores candidates much more accurately.
# It stays ON by default. Only set USE_RERANKER=false as a last resort if you
# hit hard memory limits on Render (frees ~100-150 MB).
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() not in ("false", "0", "no")

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
You are CortexOS, an expert document analyst. Answer the user's question using \
the document excerpts provided below.

Document excerpts (each tagged with its page number):
{context}

Guidelines:
- Base your answer primarily on the excerpts above.
- For summary / "key points" / "main idea" style requests, synthesize across \
ALL the excerpts into a clear, well-structured answer (use short bullets or \
sections where helpful).
- Cite the page number(s) you used, like (p. 3), where relevant.
- If the excerpts genuinely don't contain anything related to the question, \
say so briefly and mention what the document does appear to cover.
- Be accurate, helpful, and well-organized. Do not invent facts that aren't \
supported by the excerpts.

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
            response = _generate(self._client, prompt_text)
            # response.text can be None if the model returned no parts (e.g.
            # safety block or an empty thinking-only turn). Fall back gracefully
            # instead of letting a None crash the endpoint.
            text = getattr(response, "text", None)
            if text:
                return text
            try:
                parts = response.candidates[0].content.parts
                joined = "".join(getattr(p, "text", "") or "" for p in parts).strip()
                if joined:
                    return joined
            except Exception:
                pass
            return ("I wasn't able to generate an answer for that. "
                    "Please try rephrasing your question.")

    return SimpleChain(llm_client, prompt_template)


def _generate(client, prompt_text):
    """
    Call Gemini robustly across google-genai versions.

    Older google-genai builds don't accept `thinking_budget` (and some don't
    accept a `config=` object at all). We try the richest call first and
    progressively fall back so document chat never 500s on a version mismatch.
    """
    model = retrieval_config.gemini_model
    thinking_budget = int(os.getenv("GEMINI_THINKING", "512"))

    # Attempt 1: full config with a small thinking budget (best quality).
    if thinking_budget >= 0:
        try:
            cfg = genai_types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
                thinking_config=genai_types.ThinkingConfig(
                    thinking_budget=thinking_budget
                ),
            )
            return client.models.generate_content(
                model=model, contents=prompt_text, config=cfg
            )
        except Exception as e:
            print(f"[llm] thinking_config not supported, retrying without it: {e}")

    # Attempt 2: config without thinking.
    try:
        cfg = genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048,
        )
        return client.models.generate_content(
            model=model, contents=prompt_text, config=cfg
        )
    except Exception as e:
        print(f"[llm] config object not supported, retrying bare: {e}")

    # Attempt 3: bare call (works on the oldest builds).
    return client.models.generate_content(model=model, contents=prompt_text)


# ── Qdrant search ─────────────────────────────────────────────────────────────

def _run_search(query_vector, client, filename=None, user_id=None):
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

    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True
    ).points


def search_qdrant(query, model, client, filename=None, user_id=None):
    query_vector = list(model.embed([query]))[0].tolist()

    # Primary search: this user's chunks from the selected file.
    results = _run_search(query_vector, client, filename=filename, user_id=user_id)

    # Safety net: if a specific file was requested but nothing matched (e.g.
    # legacy data indexed before the filename fix), fall back to searching all
    # of this user's documents so chat still returns something useful.
    if not results and filename and user_id:
        print("[search] no match for filename; falling back to user-wide search")
        results = _run_search(query_vector, client, filename=None, user_id=user_id)

    return [
        {
            "text": r.payload["chunk_text"],
            "page_num": r.payload.get("page_num", "?"),
            "source": r.payload.get("source", "?"),
            "score": round(r.score, 3)
        }
        for r in results
    ]


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank_chunks(query, chunks, reranker):
    """reranker may be the 'lazy' sentinel string or None — both handled."""
    if not chunks:
        return []

    # If reranking is disabled, just trust Qdrant's vector similarity order.
    if not USE_RERANKER:
        return chunks[:top_n]

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