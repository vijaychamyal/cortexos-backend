"""
services/document_chat/gembed.py — API-based text embeddings via Google.

Why: fastembed loads onnxruntime + the MiniLM model (~190 MB) and the
cross-encoder reranker adds ~120 MB more. On Render's 512 MB tier that baseline
is so high that even a tiny upload's embedding spike triggers an OOM kill.

By calling Google's hosted `text-embedding-004` model we keep ZERO ML weights
in the process. Baseline memory drops by ~300 MB and embedding quality goes up.
The embedder exposes `.embed(texts)` so it is a drop-in replacement for the old
fastembed model object used across the pipeline.
"""

import os
import time

# Google embedding model + vector size. 768 is the native dimension.
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# How many texts per API call. Google allows up to 100 per batchEmbedContents.
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "50"))


# Numpy-like wrapper so existing `.tolist()` calls keep working unchanged.
class _Vec(list):
    def tolist(self):
        return list(self)


class GeminiEmbedder:
    """Drop-in replacement for the fastembed TextEmbedding object.

    Usage: embedder.embed([...texts...]) -> yields list[float] vectors.
    Keeps no model weights in memory; just calls the Google API.
    """

    def __init__(self, client, model: str = EMBED_MODEL, dim: int = EMBED_DIM):
        self._client = client
        self._model = model
        self._dim = dim

    # `batch_size` kept for signature-compatibility with the old fastembed call.
    def embed(self, texts, batch_size: int = EMBED_BATCH, task_type: str = "RETRIEVAL_DOCUMENT"):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        bs = batch_size or EMBED_BATCH

        from google.genai import types as genai_types

        for start in range(0, len(texts), bs):
            batch = texts[start:start + bs]
            vectors = self._embed_batch(batch, task_type, genai_types)
            for v in vectors:
                yield _Vec(v)

    def _embed_batch(self, batch, task_type, genai_types, retries: int = 3):
        cfg = None
        try:
            cfg = genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dim,
            )
        except Exception:
            cfg = None

        last_err = None
        for attempt in range(retries):
            try:
                resp = self._client.models.embed_content(
                    model=self._model,
                    contents=batch,
                    config=cfg,
                )
                return [list(e.values) for e in resp.embeddings]
            except Exception as e:
                last_err = e
                # Fallback: drop the config (older API shapes) once.
                if cfg is not None and attempt == 0:
                    cfg = None
                    continue
                print(f"[gembed] batch embed failed (attempt {attempt+1}): {e}")
                time.sleep(1.5 * (attempt + 1))
        raise last_err



