import os
from fastapi import FastAPI, UploadFile, Form, HTTPException
from dotenv import load_dotenv
import asyncio

from ingest import ingest_document
from query import query_document 

load_dotenv()

app = FastAPI(title="RAG Service with LangChain + Qdrant")

@app.post("/ingest/")
async def ingest(file: UploadFile = None, text: str = Form(None)):
    if file:
        content = (await file.read()).decode("utf-8")
    elif text:
        content = text
    else:
        return {"error": "Provide either file or text"}

    count = ingest_document(content)
    if count == 0:
        return {"error": "Error ingesting document"}
    return {"status": "ok", "chunks": count}


@app.post("/query/")
async def query(prompt: str = Form(...)):
    """
    Run a query against ingested knowledge (RAG) with a timeout.
    """
    timeout_seconds = 120
    try:
        resposta = await asyncio.wait_for(
            asyncio.to_thread(query_document, prompt), 
            timeout=timeout_seconds
        )
            
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, 
            detail="Request timed out after 120 seconds."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred: {str(e)}"
        )
    
    return {"answer": resposta}