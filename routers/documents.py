# routers/documents.py

from fastapi import APIRouter, UploadFile

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile):
    return {"filename": file.filename}