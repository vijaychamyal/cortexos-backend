import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
from .config import retrieval_config, collection_name
from .gembed import GeminiEmbedder

top_k = retrieval_config.top_k
top_n = retrieval_config.top_n

load_dotenv()

# Reranking improves answer quality. We now rerank with Gemini itself (an API
# call, ZERO local memory) instead of a local cross-encoder model that ate
# ~120 MB of RAM. Set USE_RERANKER=false to skip it (slightly faster).
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
    """Return an API-based embedder (no local ML weights -> tiny memory).

    This replaces the fastembed MiniLM model that used ~190 MB of RAM and was
    the main cause of OOM kills on Render's 512 MB tier.
    """
    print("[AI Engine] Using Google text-embedding API (no local model).")
    return GeminiEmbedder(load_llm())


# ── Reranker ──────────────────────────────────────────────────────────────────
# The old local cross-encoder (~120 MB) is gone. Reranking, when enabled, is
# done by Gemini via an API call (see rerank_chunks) — zero local memory.

def load_reranker():
    return "gemini" if USE_RERANKER else "off"


# ── Gemini (direct google-genai, no LangChain) ────────────────────────────────

_llm_singleton = None


def load_llm():
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton
    _llm_singleton = _build_llm()
    return _llm_singleton


def _build_llm():
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


def _thinking_supported() -> bool:
    """True only if the installed google-genai's ThinkingConfig accepts
    `thinking_budget`. Older builds (e.g. 1.2.0) don't, so we skip that path
    entirely instead of triggering a guaranteed error every request."""
    try:
        fields = getattr(genai_types.ThinkingConfig, "model_fields", {})
        return "thinking_budget" in fields
    except Exception:
        return False


_THINKING_OK = _thinking_supported()


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
    # Only attempted if the installed library actually supports the field.
    if _THINKING_OK and thinking_budget >= 0:
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
    # Use the RETRIEVAL_QUERY task type for the question side (better matching).
    try:
        query_vector = list(model.embed([query], task_type="RETRIEVAL_QUERY"))[0].tolist()
    except TypeError:
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
    """Rerank candidate chunks by relevance to the query.

    Uses Gemini (an API call, zero local memory) to score each chunk, instead
    of the old local cross-encoder that consumed ~120 MB of RAM. Falls back to
    the vector-similarity order if reranking is off or the API call fails.
    """
    if not chunks:
        return []

    if not USE_RERANKER:
        return chunks[:top_n]

    try:
        ranked = _gemini_rerank(query, chunks)
        if ranked:
            return ranked[:top_n]
    except Exception as e:
        print(f"[rerank] gemini rerank failed, using vector order: {e}")

    return chunks[:top_n]


def _gemini_rerank(query, chunks):
    """Ask Gemini to order the candidate chunks by relevance and return the
    reordered list. Robust to malformed model output."""
    client = load_llm()

    listing = "\n".join(
        f"[{i}] {c['text'][:500]}" for i, c in enumerate(chunks)
    )
    prompt = (
        "You are a search reranker. Given a QUESTION and numbered PASSAGES, "
        "return the passage indices ordered from MOST to LEAST relevant.\n"
        "Respond with ONLY a JSON array of integers, e.g. [3,0,1].\n\n"
        f"QUESTION: {query}\n\nPASSAGES:\n{listing}\n\nJSON:"
    )

    resp = _generate(client, prompt)
    text = (getattr(resp, "text", None) or "").strip()

    # Extract the JSON array of indices.
    start, end = text.find("["), text.rfind("]")
    order = []
    if start != -1 and end != -1 and end > start:
        try:
            order = [int(x) for x in json.loads(text[start:end + 1])]
        except Exception:
            order = []

    if not order:
        return None

    seen, result = set(), []
    for idx in order:
        if 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            result.append(chunks[idx])
    # Append any chunks the model omitted, preserving original order.
    for i, c in enumerate(chunks):
        if i not in seen:
            result.append(c)
    return result


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