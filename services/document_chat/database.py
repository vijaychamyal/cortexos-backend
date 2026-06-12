from .config import collection_name, config
from qdrant_client.models import Distance, VectorParams, PointStruct

vector_size = config.vector_size
batch_size  = config.batch_size


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
    client.create_payload_index(
        collection_name=collection_name,
        field_name="source",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="user_id",
        field_schema="keyword"
    )
    print("Payload indexes ensured for 'source' and 'user_id'.")


def insert_to_qdrant(chunks, vectors, client):
    """
    Uses upsert with auto-incrementing IDs based on current collection size
    so new uploads append rather than collide with existing points.
    """
    # Get current count so new point IDs don't overwrite existing ones
    try:
        info = client.get_collection(collection_name)
        id_offset = info.points_count or 0
    except Exception:
        id_offset = 0

    points = []
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
            vector=vector.tolist(),
            payload=payload
        ))

    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"Inserted {len(points)} points (total offset was {id_offset})")