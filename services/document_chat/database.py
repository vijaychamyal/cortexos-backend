import gc

from .config import collection_name, config
from qdrant_client.models import Distance, VectorParams, PointStruct

vector_size  = config.vector_size
batch_size   = config.batch_size
insert_batch = config.insert_batch


def ensure_payload_indexes(client):
    """
    Make sure the payload indexes needed for filtered search exist.
    Filtering by `user_id` / `source` during chat REQUIRES a keyword index;
    without it Qdrant returns 400 "Index required but not found".

    Each index is created independently and idempotently so an older
    collection (created before indexes existed) gets upgraded automatically.
    wait=True forces Qdrant to actually build the index before returning.
    """
    for field in ("source", "user_id"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword",
                wait=True,
            )
            print(f"Payload index ensured for '{field}'.")
        except Exception as e:
            # Already-exists errors are fine; log anything else but don't crash.
            msg = str(e).lower()
            if "already exists" in msg or "already created" in msg:
                print(f"Payload index for '{field}' already exists.")
            else:
                print(f"[warn] could not create index for '{field}': {e}")


def create_collection(client):
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' created.")

    # Always ensure indexes exist (safe to call even if they already exist)
    ensure_payload_indexes(client)


def _id_offset(client):
    """Current point count so new IDs append instead of colliding."""
    try:
        info = client.get_collection(collection_name)
        return info.points_count or 0
    except Exception:
        return 0


def insert_to_qdrant(chunks, vectors, client):
    """
    Uses upsert with auto-incrementing IDs based on current collection size
    so new uploads append rather than collide with existing points.

    Inserts in small batches and frees each batch immediately to keep peak
    memory low (important on Render's 512 MB free tier).
    """
    id_offset = _id_offset(client)

    points = []
    total = 0
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "chunk_text": chunk["chunk_text"],
            "page_num":   chunk["page_num"],
            "source":     chunk["source"],
            "chunk_id":   chunk["chunk_id"],
        }
        if "user_id" in chunk:
            payload["user_id"] = chunk["user_id"]

        points.append(PointStruct(
            id=id_offset + i,
            vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
            payload=payload
        ))

        if len(points) >= insert_batch:
            client.upsert(collection_name=collection_name, points=points)
            total += len(points)
            points.clear()
            gc.collect()

    if points:
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
        points.clear()
        gc.collect()

    print(f"Inserted {total} points (total offset was {id_offset})")


def insert_stream_to_qdrant(chunks, vector_iter, client):
    """
    Streaming insert: pulls vectors one at a time from a generator and
    upserts them in small batches. Never holds all vectors in RAM at once.
    """
    id_offset = _id_offset(client)

    points = []
    total = 0
    for i, (chunk, vector) in enumerate(zip(chunks, vector_iter)):
        payload = {
            "chunk_text": chunk["chunk_text"],
            "page_num":   chunk["page_num"],
            "source":     chunk["source"],
            "chunk_id":   chunk["chunk_id"],
        }
        if "user_id" in chunk:
            payload["user_id"] = chunk["user_id"]

        points.append(PointStruct(
            id=id_offset + i,
            vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
            payload=payload
        ))

        if len(points) >= insert_batch:
            client.upsert(collection_name=collection_name, points=points)
            total += len(points)
            points.clear()
            gc.collect()

    if points:
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
        points.clear()
        gc.collect()

    print(f"Streamed-insert {total} points (total offset was {id_offset})")
    return total