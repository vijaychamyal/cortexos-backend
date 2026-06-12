import os
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from .config import retrieval_config, collection_name
from qdrant_client.http import models
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed import TextEmbedding

top_k = retrieval_config.top_k #from qdrant
top_n = retrieval_config.top_n

load_dotenv()
gemini_api_key=  os.getenv("gemini_api_key")
gemini_model    = "models/gemini-2.5-flash"
rerank_model    = "cross-encoder/ms-marco-MiniLM-L-6-v2"  

# connect to qdrant running on docker
def setup_qdrant():
    try:
        client = QdrantClient(
        url=os.environ.get("QDRANT_CLOUD_URL"),
        api_key=os.environ.get("QDRANT_API_KEY")
    )
        client.get_collections()
        print("loading")
        return client
    except Exception as e:
        print("qdrant not connected, check if docker is running")
        raise e

# load the same minilm model used in embed.py


def load_model():
    print("[AI Engine] Loading MiniLM via fastembed...")
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    return model

# load cross encoder for reranking

def load_reranker():
    reranker = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    return reranker


def load_llm():
    llm = ChatGoogleGenerativeAI(
        model=retrieval_config.gemini_model,
        google_api_key=gemini_api_key,
        temperature=0
    )
    return llm

# create prompt template for rag
def create_prompt():
    template = """
You are a helpful assistant that answers questions strictly based on the provided context.

Context from the document:
{context}

Instructions:
- Answer only based on the context above
- If the answer is not in the context say I could not find this information in the document
- Mention the page number where you found the answer
- Be concise and precise

Question: {question}

Answer:"""

    return PromptTemplate(
        template       =template,
        input_variables=["context", "question"]
    )


# build langchain rag chain using pipe operator
def create_rag_chain(llm, prompt):
    output_parser = StrOutputParser()
    chain         = prompt | llm | output_parser
    return chain


# search qdrant for top k relevant chunks
# def search_qdrant(query, model, client):

#     query_vector = model.encode(
#         query,
#         normalize_embeddings=True
#     ).tolist()

#     results = client.query_points(
#         collection_name=collection_name,
#         query          =query_vector,
#         limit          =top_k,
#         with_payload   =True
#     ).points

#     chunks = []
#     for r in results:
#         chunks.append({
#             "text"    : r.payload["chunk_text"],
#             "page_num": r.payload["page_num"],
#             "source"  : r.payload["source"],
#             "score"   : round(r.score, 3)
#         })

#     return chunks
# search qdrant for top k relevant chunks, filtered by filename if provided
def search_qdrant(query, model, client, filename=None):
    query_vector = list(model.embed([query]))[0].tolist()

    # Build the strict filter
    query_filter = None
    if filename:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source", # This matches the metadata key saved during upload
                    match=models.MatchValue(value=filename)
                )
            ]
        )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter, # Apply the filter here
        limit=top_k,
        with_payload=True
    ).points

    chunks = []
    for r in results:
        chunks.append({
            "text": r.payload["chunk_text"],
            "page_num": r.payload["page_num"],
            "source": r.payload["source"],
            "score": round(r.score, 3)
        })

    return chunks


# rerank chunks using cross encoder and keep top n
def rerank_chunks(query, chunks, reranker):
    if not chunks:
        return []

    documents = [chunk["text"] for chunk in chunks]
    scores = list(reranker.rerank(query, documents))

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = round(float(scores[i]), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# convert reranked chunks into a single context string
def chunks_to_context(chunks):
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        part = f"[Source Page {chunk['page_num']}]\n{chunk['text']}"
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)


# generate final answer using gemini and retrieved context
def generate_answer(query, reranked_chunks, chain):
    if not reranked_chunks:
        return "No relevant information found in the document."

    context = chunks_to_context(reranked_chunks)

    answer = chain.invoke({
        "context" : context,
        "question": query
    })

    return answer



# main pipeline that runs when user asks a question
def ask_query(query, model, client, reranker, chain, filename=None):
    # Pass the filename down to the Qdrant search
    chunks   = search_qdrant(query, model, client, filename)
    reranked = rerank_chunks(query, chunks, reranker)
    answer   = generate_answer(query, reranked, chain)
    print(f"\nAnswer: {answer}")
    return answer


# entry point with interactive loop
if __name__ == "__main__":

    # setup all components
    client   = setup_qdrant()
    model    = load_model()
    reranker = load_reranker()
    llm      = load_llm()
    prompt   = create_prompt()
    chain    = create_rag_chain(llm, prompt)

    print("Type your question (type exit to quit)")

    # interactive question loop
    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        ask_query(query, model, client, reranker, chain)
