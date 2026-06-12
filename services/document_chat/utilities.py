import re
import os
import torch

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from .config import collection_name
from .config import QdrantConfig
# symbol removal and text cleaning
def clean_text(text):
    text = re.sub(r'\x00', ' ', text)
    text = re.sub(r'[\u2500-\u27FF]', ' ', text)
    text = re.sub(r'[\u2000-\u206F]', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

# garbage detection after cleaning
def is_garbage_text(text):
    cleaned = re.sub(r'[^\x20-\x7E]', ' ', text)
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    words   = cleaned.split()

    if len(words) < 10:                 
        return True

    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 2.5:
        return True

    return False


def setup_qdrant():
    try:
        qdrant_config = QdrantConfig()
        client = QdrantClient(
            host=qdrant_config.host,
            port=qdrant_config.port
        )
        return client
    except Exception as e:
        print("Qdrant is not connected")
        print("First run: docker run -p 6333:6333 qdrant/qdrant")
        raise e


# 1. Force low-level C++ libraries to only use 1 thread (must be set before model loads)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def load_model():
    # 2. Force PyTorch itself to use exactly 1 CPU thread (Massive RAM savings)
    torch.set_num_threads(1)
    
    print("[AI Engine] Loading MiniLM in low-memory CPU mode...")
    
    # 3. Explicitly tell the model to load only on the CPU
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", 
        device="cpu"
    )
    
    return model


# verify insertion
def verify_insert(client):
    info   = client.get_collection(collection_name)
    sample = client.retrieve(
        collection_name=collection_name,
        ids            =[0],
        with_payload   =True,
        with_vectors   =False
    )

    print(f"verification")
    print(f"  Points stored : {info.points_count}")
    print(f"  Vector size : {info.config.params.vectors.size}")
    print(f"  chunk_text : {sample[0].payload['chunk_text'][:200]}")
    print(f"  page_num : {sample[0].payload['page_num']}")
    print(f"  source : {sample[0].payload['source']}")