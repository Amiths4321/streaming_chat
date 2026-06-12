# stream_api.py
# uvicorn stream_api:app --port 8001 --reload

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from stream_rag import stream_rag_answer, retrieve_chunks
from stream_ollama import stream_tokens
import json

app = FastAPI(title="Streaming AI API")


class ChatRequest(BaseModel):
    message: str
    history: list = []
    use_rag: bool = True


def sse_generator(generator):
    """
    Wrap a token generator as Server-Sent Events.
    Each event: data: {"token": "..."}\n\n
    """
    for token in generator:
        data = json.dumps({"token": token})
        yield f"data: {data}\n\n"
    # Signal end of stream
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint.
    Returns Server-Sent Events — one event per token.
    """
    if req.use_rag:
        gen = stream_rag_answer(req.message, req.history)
    else:
        # Plain chat without RAG
        messages = req.history + [{"role": "user", "content": req.message}]
        from stream_ollama import stream_chat
        gen = stream_chat(messages)

    return StreamingResponse(
        sse_generator(gen),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.post("/chat/context")
def get_context(req: ChatRequest):
    """Get retrieved chunks without streaming — for showing sources."""
    chunks = retrieve_chunks(req.message)
    return {"chunks": chunks}


@app.get("/health")
def health():
    return {"status": "ok"}