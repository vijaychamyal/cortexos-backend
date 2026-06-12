from .config import config

batch_size    = config.batch_size
chunk_size    = config.chunk_size
chunk_overlap = config.chunk_overlap


# ── Pure-Python recursive text splitter ──────────────────────────────────────
# Replaces langchain_text_splitters.RecursiveCharacterTextSplitter.
# Same behaviour: tries each separator in order, recurses if chunks still too big.

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, separators: list[str], size: int, overlap: int) -> list[str]:
    """Recursively split text by trying each separator in order."""
    if len(text) <= size:
        return [text]

    separator = ""
    remaining = list(separators)

    while remaining:
        sep = remaining.pop(0)
        if sep == "" or sep in text:
            separator = sep
            break

    splits = text.split(separator) if separator else list(text)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for part in splits:
        part_len = len(part) + len(separator)
        if current_len + part_len > size and current_parts:
            chunk = separator.join(current_parts).strip()
            if chunk:
                chunks.append(chunk)
            # keep overlap: walk back until we're within overlap budget
            while current_parts and current_len > overlap:
                removed = current_parts.pop(0)
                current_len -= len(removed) + len(separator)
        current_parts.append(part)
        current_len += part_len

    if current_parts:
        chunk = separator.join(current_parts).strip()
        if chunk:
            chunks.append(chunk)

    # Recurse on any chunk that's still too large
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) > size and len(remaining) > 0:
            final.extend(_split_text(chunk, separators, size, overlap))
        else:
            final.append(chunk)

    return final


# ── Public API (same signatures as before) ───────────────────────────────────

def make_chunks(pages: list[dict]) -> list[dict]:
    all_chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        raw_chunks = _split_text(page["text"], list(SEPARATORS), chunk_size, chunk_overlap)
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if len(chunk) < 80:
                continue
            all_chunks.append({
                "chunk_text": chunk,
                "page_num":   page["page_num"],
                "source":     page["source"],
                "chunk_id":   chunk_id
            })
            chunk_id += 1

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def embed_chunks(chunks: list[dict], model) -> list:
    texts = [chunk["chunk_text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks")
    return list(model.embed(texts, batch_size=batch_size))


def embed_texts_iter(texts, model):
    """
    Memory-friendly generator: yields embedding vectors one at a time.
    fastembed's .embed() is itself a lazy generator, so iterating it here
    means we never hold all vectors in RAM simultaneously.
    """
    yield from model.embed(texts, batch_size=batch_size)