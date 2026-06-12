# No local ML model to download anymore.
# Embeddings are served by Google's text-embedding API (see services/document_chat/gembed.py),
# which keeps the process memory tiny on Render's free tier.
print("No local model to pre-download (using Google embedding API).")
