from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import traceback
import os
import db

from services.document_chat.processor import (
    setup_qdrant,
    load_model,
    load_reranker,
    load_llm,
    create_prompt,
    create_rag_chain,
    ask_query
)

from services.document_chat.embed import main_pipeline

ai_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[CortexOS] Initializing AI Engines and Database Connections...")
    ai_models["qdrant_client"]  = setup_qdrant()
    ai_models["embedding_model"] = load_model()
    ai_models["reranker_model"]  = load_reranker()   # lazy sentinel
    ai_models["gemini_llm"]      = load_llm()
    rag_prompt                   = create_prompt()
    ai_models["rag_chain"]       = create_rag_chain(ai_models["gemini_llm"], rag_prompt)
    print("[CortexOS] All systems fully loaded and online!\n")
    yield
    ai_models.clear()


app = FastAPI(title="CortexOS Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    filename: Optional[str] = None
    user_id:  str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "CortexOS Backend"}


@app.post("/upload")
async def upload_file(
    file:    UploadFile = File(...),
    user_id: str        = Form(...)
):
    temp_file_path = f"temp_{file.filename}"
    try:
        contents = await file.read()

        # 1. Upload to Supabase cloud storage
        storage_path = db.upload_file_to_cloud(user_id, contents, file.filename)

        # 2. File size metadata
        size_mb = f"{round(len(contents) / (1024 * 1024), 1)} MB · Just now"

        # 3. Register in Postgres
        db.register_user_document(user_id, file.filename, storage_path, size_mb)

        # 4. Write temp file for Qdrant indexing
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        # 5. Embed and index into Qdrant
        main_pipeline(temp_file_path, user_id=user_id, qdrant_client=ai_models["qdrant_client"])    

        return {
            "message":  f"File '{file.filename}' processed and indexed successfully!",
            "filename": file.filename
        }

    except Exception as e:
        # Print full traceback to Render logs so you can see the real error
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        answer = ask_query(
            query    = request.question,
            model    = ai_models["embedding_model"],
            client   = ai_models["qdrant_client"],
            reranker = ai_models["reranker_model"],
            chain    = ai_models["rag_chain"],
            filename = request.filename,
            user_id  = request.user_id
        )
        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))