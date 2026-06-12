from .config import collection_name, config
from qdrant_client.models import Distance, VectorParams, PointStruct

vector_size = config.vector_size
batch_size  = config.batch_size


def create_collection(client):
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        client.delete_collection(collection_name=collection_name)
        print(f"Old collection '{collection_name}' deleted")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{collection_name}' created")
    return True


def insert_to_qdrant(chunks, vectors, client):
    points = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "chunk_text": chunk["chunk_text"],
            "page_num":   chunk["page_num"],
            "source":     chunk["source"],
            "chunk_id":   chunk["chunk_id"],
        }
        # Store user_id if present so queries can filter by owner
        if "user_id" in chunk:
            payload["user_id"] = chunk["user_id"]

        points.append(PointStruct(
            id=i,
            vector=vector.tolist(),
            payload=payload
        ))

    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"Inserted {len(points)} points")