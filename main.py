import os
import shutil

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from rag_graph import rag_app, index_pdf, RAGState, SUBJECT_STORES

app = FastAPI(title="Agentic Education RAG API")

UPLOAD_DIR = "uploads"

# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question:    str
    answer:      str
    summary:     str
    source:      str
    page:        str
    retry_count: int

class UploadResponse(BaseModel):
    subject:        str
    filename:       str
    chunks_created: int

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Agentic Education RAG API is running"}


@app.post("/upload-pdf", response_model=list[UploadResponse])
def upload_pdf(
    physics_file: UploadFile = File(None, description="Physics PDF file"),
    chemistry_file: UploadFile = File(None, description="Chemistry PDF file"),
    biology_file: UploadFile = File(None, description="Biology PDF file"),
):
    """
    Upload PDFs for physics, chemistry and biology.
    You can upload one, two or all three at once.
    """
    uploads = [
        ("physics",   physics_file),
        ("chemistry", chemistry_file),
        ("biology",   biology_file),
    ]
 
    responses = []
    for subject, file in uploads:
        if file is None or not file.filename:
            continue  # skip if not uploaded
 
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")
 
        # Save uploaded file to uploads/ folder and pass the path
        pdf_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
 
        chunks = index_pdf(pdf_path, subject)
        responses.append(UploadResponse(subject=subject, filename=file.filename, chunks_created=chunks))
 
    if not responses:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
 
    return responses


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Ask a question. Runs the full LangGraph RAG pipeline and returns the answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag_app.invoke(RAGState(question=request.question))

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        summary=result["summary"],
        source=result["source"],
        page=result["page"],
        retry_count=result["retry_count"],
    )
