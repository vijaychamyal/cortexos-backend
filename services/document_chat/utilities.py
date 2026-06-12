import re
import os

from qdrant_client import QdrantClient
from .config import collection_name, QdrantConfig


def clean_text(text: str) -> str:
    text = re.sub(r'\x00', ' ', text)
    text = re.sub(r'[\u2500-\u27FF]', ' ', text)
    text = re.sub(r'[\u2000-\u206F]', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def is_garbage_text(text: str) -> bool:
    cleaned = re.sub(r'[^\x20-\x7E]', ' ', text)
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    words   = cleaned.split()

    if len(words) < 10:
        return True
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 2.5:
        return True
    return False






def verify_insert(client) -> None:
    info   = client.get_collection(collection_name)
    sample = client.retrieve(
        collection_name=collection_name,
        ids=[0],
        with_payload=True,
        with_vectors=False
    )
    print("verification")
    print(f"  Points stored : {info.points_count}")
    print(f"  Vector size   : {info.config.params.vectors.size}")
    print(f"  chunk_text    : {sample[0].payload['chunk_text'][:200]}")
    print(f"  page_num      : {sample[0].payload['page_num']}")
    print(f"  source        : {sample[0].payload['source']}")