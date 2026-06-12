from fastembed import TextEmbedding

print("Pre-downloading embedding model...")
TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
print("Model downloaded successfully!")