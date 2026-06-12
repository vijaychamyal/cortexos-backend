from pydantic import BaseModel, Field

# New collection name because we switched from 384-dim MiniLM (fastembed) to
# 768-dim Google text-embedding-004. A collection's vector size is fixed at
# creation, so we must use a fresh collection rather than the old 384-dim one.
collection_name = "cortex_docs_v2"

class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333

class EmbedConfig(BaseModel):
    # 768 = Google text-embedding-004 native dimension.
    vector_size   : int = Field(default=768,  gt=0)
    # Keep batches small so peak RAM stays well under Render's 512 MB cap.
    batch_size    : int = Field(default=8,    gt=0)
    chunk_size    : int = Field(default=1000, gt=0)
    chunk_overlap : int = Field(default=200,  gt=0)
    # How many chunks to embed + insert into Qdrant at a time before freeing
    # the vectors. This bounds peak memory regardless of document size.
    insert_batch  : int = Field(default=64,   gt=0)

class RetrievalConfig(BaseModel):
    # top_k = how many candidates Qdrant returns for the reranker to re-score.
    # A wider pool gives the cross-encoder more to work with = better ranking.
    top_k        : int = Field(default=12, gt=0)
    # top_n = how many of the best chunks are actually sent to the LLM.
    # More context = better, more grounded answers (at a little more latency).
    top_n        : int = Field(default=5,  gt=0)
    gemini_model : str = "models/gemini-2.5-flash"
    rerank_model : str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

config           = EmbedConfig()
qdrant_config    = QdrantConfig()
retrieval_config = RetrievalConfig()