from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import config
batch_size =config.batch_size
chunk_size= config.chunk_size         
chunk_overlap= config.chunk_overlap 
def make_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size     =chunk_size,
        chunk_overlap  =chunk_overlap,
        separators     =["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    all_chunks = []
    chunk_id   = 0

    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 80:
                continue
            all_chunks.append({
                "chunk_text": chunk,
                "page_num"  : page["page_num"],
                "source"    : page["source"],
                "chunk_id"  : chunk_id
            })
            chunk_id += 1
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks

def embed_chunks(chunks, model):
    texts = [chunk["chunk_text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks")
    vectors = model.encode(
        texts,
        batch_size          =batch_size,
        show_progress_bar   =True,
        convert_to_numpy    =True,
        normalize_embeddings=True
    )
    return vectors
