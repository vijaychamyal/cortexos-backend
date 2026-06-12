print("[Build Step] Pre-downloading embedding model to cache...")
from sentence_transformers import SentenceTransformer
# This forces the download to happen right now during the build
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("[Build Step] Model downloaded successfully!")