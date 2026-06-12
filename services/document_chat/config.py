from pydantic import BaseModel, Field

collection_name = "check_1_ppt"

class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333

class EmbedConfig(BaseModel):
    vector_size   : int = Field(default=384,  gt=0)
    batch_size    : int = Field(default=32,   gt=0)
    chunk_size    : int = Field(default=1000, gt=0)
    chunk_overlap : int = Field(default=200,  gt=0)

class RetrievalConfig(BaseModel):
    top_k         : int = Field(default=10, gt=0)
    top_n         : int = Field(default=3,  gt=0)
    gemini_model  : str = "models/gemini-2.5-flash"
    rerank_model  : str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

config = EmbedConfig()
qdrant_config = QdrantConfig()
retrieval_config = RetrievalConfig()