from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import db 

# Import all setup and execution functions from processor.py
from services.document_chat.processor import (
    setup_qdrant,
    load_model,
    load_reranker,
    load_llm,
    create_prompt,
    create_rag_chain,
    ask_query
)
# Import your document upload pipeline
from services.document_chat.embed import main_pipeline

# Dictionary to hold our heavy models globally in memory
ai_models = {}

# 1. Define the Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block runs EXACTLY ONCE when the actual worker starts up
    print("\n[CortexOS] Initializing AI Engines and Database Connections...")
    
    ai_models["qdrant_client"] = setup_qdrant()
    ai_models["embedding_model"] = load_model()
    ai_models["reranker_model"] = load_reranker()
    ai_models["gemini_llm"] = load_llm()
    
    rag_prompt = create_prompt()
    ai_models["rag_chain"] = create_rag_chain(ai_models["gemini_llm"], rag_prompt)
    
    print("[CortexOS] All systems fully loaded and online!\n")
    
    yield # Server is now running and accepting requests
    
    # Clean up when the server shuts down
    ai_models.clear()

# 2. Pass the lifespan handler into FastAPI
app = FastAPI(title="CortexOS Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added filename and user_id fields to the request model
class ChatRequest(BaseModel):
    question: str
    filename: Optional[str] = None
    user_id: str  # Tracks either guest session UUID or authenticated user UUID

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...)  # Accepts the user tracking identifier from the UI form
):
    try:
        contents = await file.read()
        
        # 1. Stream file directly into your Supabase public cloud storage bucket
        storage_path = db.upload_file_to_cloud(user_id, contents, file.filename)
        
        # 2. Compute readable file sizing metadata
        size_mb = f"{round(len(contents) / (1024 * 1024), 1)} MB · Just now"
        
        # 3. Insert metadata records inside the Postgres user_documents table
        db.register_user_document(user_id, file.filename, storage_path, size_mb)
        
        # 4. Process local copy briefly for the Qdrant indexing engine pipeline
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(contents)
            
        # Passing user_id inside your layout ensures vectors can be partitioned by owner later
        main_pipeline(temp_file_path, user_id=user_id)
        
        return {
            "message": f"File '{file.filename}' processed and indexed successfully into Qdrant!",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        # Pull the pre-warmed models out of the ai_models dictionary
        answer = ask_query(
            query=request.question,
            model=ai_models["embedding_model"],
            client=ai_models["qdrant_client"],
            reranker=ai_models["reranker_model"],
            chain=ai_models["rag_chain"],
            filename=request.filename,
            user_id=request.user_id # Passes tenant context tracking downwards to prevent leakages
        )
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))