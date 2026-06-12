from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import traceback
import gc
import os
import uuid
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
from services.stock.analysis import stock_chat, predict_prices

ai_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[CortexOS] Initializing AI Engines and Database Connections...")
    ai_models["qdrant_client"]  = setup_qdrant()
    # Ensure the existing collection has the payload indexes that filtered
    # chat search needs (fixes 400 "Index required but not found for user_id").
    try:
        from services.document_chat.database import create_collection
        create_collection(ai_models["qdrant_client"])
    except Exception as e:
        print(f"[CortexOS] index ensure at startup failed (non-fatal): {e}")
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


class StockChatRequest(BaseModel):
    question: str


class StockPredictRequest(BaseModel):
    ticker: str
    horizon: Optional[int] = 7


@app.get("/")
def health_check():
    return {"status": "ok", "service": "CortexOS Backend"}


# Reject very large files up front — they can't be processed within the
# 512 MB free-tier budget anyway. Tune via MAX_UPLOAD_MB env var.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))


@app.post("/upload")
async def upload_file(
    file:    UploadFile = File(...),
    user_id: str        = Form(...)
):
    # Use a unique temp name (preserving extension) so concurrent uploads of
    # the same filename don't clobber each other. The real name is tracked
    # separately and stored as the Qdrant `source`.
    _ext = os.path.splitext(file.filename or "")[1]
    temp_file_path = f"temp_{uuid.uuid4().hex}{_ext}"
    try:
        # 1. Stream the upload straight to disk in chunks so we never hold the
        #    whole file in RAM. Track size as we go and enforce a hard cap.
        size_bytes = 0
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        with open(temp_file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB at a time
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB."
                    )
                f.write(chunk)

        # 2. Upload to Supabase from the temp file path (no extra in-RAM copy).
        storage_path = db.upload_file_from_path(user_id, temp_file_path, file.filename)

        # 3. File size metadata
        size_mb = f"{round(size_bytes / (1024 * 1024), 1)} MB · Just now"

        # 4. Register in Postgres
        db.register_user_document(user_id, file.filename, storage_path, size_mb)

        # 5. Embed and index into Qdrant (streams internally).
        #    Pass the ORIGINAL filename so the stored `source` matches what
        #    the chat endpoint filters by (otherwise search returns nothing).
        gc.collect()
        main_pipeline(
            temp_file_path,
            user_id=user_id,
            qdrant_client=ai_models["qdrant_client"],
            original_filename=file.filename,
        )
        gc.collect()

        return {
            "message":  f"File '{file.filename}' processed and indexed successfully!",
            "filename": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        # Print full traceback to Render logs so you can see the real error
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        gc.collect()


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


# ── Capital Pulse: Stock Intelligence ─────────────────────────────────────────

@app.post("/stock/chat")
async def stock_chat_endpoint(request: StockChatRequest):
    """Analytical stock chatbot: explains market moves using live data + news."""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        result = stock_chat(request.question, ai_models["gemini_llm"])
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stock/predict")
async def stock_predict_endpoint(request: StockPredictRequest):
    """Lightweight 7-day price forecast with RMSE/MAE/MAPE backtest metrics."""
    if not request.ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    try:
        horizon = max(1, min(int(request.horizon or 7), 30))
        result = predict_prices(request.ticker.strip().upper(), horizon=horizon)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))