"""
services/document_chat/gembed.py — API-based text embeddings via Google.

Why: fastembed loads onnxruntime + the MiniLM model (~190 MB) and the
cross-encoder reranker adds ~120 MB more. On Render's 512 MB tier that baseline
is so high that even a tiny upload's embedding spike triggers an OOM kill.

By calling Google's hosted embedding model we keep ZERO ML weights in the
process. Baseline memory drops by ~300 MB and embedding quality goes up.
The embedder exposes `.embed(texts)` so it is a drop-in replacement for the old
fastembed model object used across the pipeline.

Different API keys / API versions expose embedding models under different names
(e.g. "text-embedding-004", "models/text-embedding-004", "embedding-001",
"gemini-embedding-001"). Rather than hardcode one and 404, we auto-detect the
first candidate that actually works and cache it.
"""

import os
import time
import math

# Requested output dimension. text-embedding-004 is natively 768; the newer
# gemini-embedding-001 defaults to 3072 but supports truncation to 768. We ask
# for 768 to keep storage/memory low. The ACTUAL dimension produced is detected
# at runtime (see GeminiEmbedder.dim) so the Qdrant collection always matches.
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# How many texts per API call.
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "50"))

# Candidate model names, tried in order. An explicit EMBED_MODEL env var (if set)
# is tried first. Names with and without the "models/" prefix are both probed
# because different google-genai/API versions disagree on the required form.
_DEFAULT_CANDIDATES = [
    "text-embedding-004",
    "models/text-embedding-004",
    "gemini-embedding-001",
    "models/gemini-embedding-001",
    "embedding-001",
    "models/embedding-001",
    "text-embedding-005",
    "models/text-embedding-005",
]


def _candidate_models():
    env = os.getenv("EMBED_MODEL")
    cands = []
    if env:
        cands.append(env)
        if not env.startswith("models/"):
            cands.append(f"models/{env}")
    for c in _DEFAULT_CANDIDATES:
        if c not in cands:
            cands.append(c)
    return cands


# Numpy-like wrapper so existing `.tolist()` calls keep working unchanged.
class _Vec(list):
    def tolist(self):
        return list(self)


class GeminiEmbedder:
    """Drop-in replacement for the fastembed TextEmbedding object.

    Usage: embedder.embed([...texts...]) -> yields vectors.
    Keeps no model weights in memory; just calls the Google API.
    """

    def __init__(self, client, model: str = None, dim: int = EMBED_DIM):
        self._client = client
        self._dim = dim                # requested dimension
        self.dim = dim                 # ACTUAL dimension (updated after probe)
        # Resolved lazily on first call (auto-detect a working model name).
        self._model = model
        self._use_config = True  # whether EmbedContentConfig is accepted

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

    # ── internal ──────────────────────────────────────────────────────────────
    def _make_cfg(self, task_type, genai_types):
        if not self._use_config:
            return None
        try:
            return genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dim,
            )
        except Exception:
            self._use_config = False
            return None

    @staticmethod
    def _normalize(vec):
        # Unit-normalize. Required for gemini-embedding-001 truncated dims and
        # harmless for already-normalized models. We use COSINE distance, so
        # normalized vectors are exactly what we want.
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            return [x / norm for x in vec]
        return vec

    def _call(self, model, batch, cfg):
        resp = self._client.models.embed_content(
            model=model, contents=batch, config=cfg
        )
        vectors = [self._normalize(list(e.values)) for e in resp.embeddings]
        if vectors:
            self.dim = len(vectors[0])  # record the real dimension produced
        return vectors

    def _resolve_model(self, batch, task_type, genai_types):
        """Find the first candidate model name that works, using a 1-item probe."""
        probe = batch[:1] or ["test"]
        last_err = None
        for name in _candidate_models():
            for use_cfg in (True, False):
                cfg = self._make_cfg(task_type, genai_types) if use_cfg else None
                try:
                    self._call(name, probe, cfg)
                    self._model = name
                    self._use_config = use_cfg
                    print(f"[gembed] using embedding model: {name} "
                          f"(config={'on' if use_cfg else 'off'})")
                    return
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    # 404 / not found -> try next name. Other errors -> also try,
                    # but remember the error for the final raise.
                    continue
        raise RuntimeError(
            f"No working Google embedding model found. Last error: {last_err}"
        )

    def _embed_batch(self, batch, task_type, genai_types, retries: int = 3):
        if self._model is None:
            self._resolve_model(batch, task_type, genai_types)

        last_err = None
        for attempt in range(retries):
            cfg = self._make_cfg(task_type, genai_types)
            try:
                return self._call(self._model, batch, cfg)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # If the cached model suddenly 404s, re-detect once.
                if "not found" in msg or "404" in msg:
                    self._model = None
                    try:
                        self._resolve_model(batch, task_type, genai_types)
                        continue
                    except Exception as e2:
                        last_err = e2
                        break
                # Config rejected -> retry without it.
                if self._use_config:
                    self._use_config = False
                    continue
                print(f"[gembed] batch embed failed (attempt {attempt+1}): {e}")
                time.sleep(1.5 * (attempt + 1))
        raise last_err
