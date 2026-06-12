# stream_rag.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RAG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "rag_system"
)
sys.path.insert(0, RAG_PATH)

from typing import Generator
from stream_ollama import stream_tokens

TOP_K = 4


def retrieve_chunks(query: str) -> list[dict]:
    """Retrieve relevant chunks from ChromaDB."""
    try:
        from rag import get_collection, embed_texts
        qvec       = embed_texts([query])[0]
        collection = get_collection()

        if collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[qvec],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text":       doc,
                "source":     meta.get("source", ""),
                "similarity": round(1 - dist, 3)
            })
        return chunks
    except Exception as e:
        print(f"RAG error: {e}")
        return []


def stream_rag_answer(
    query:   str,
    history: list[dict] = None
) -> Generator[str, None, None]:
    """
    Full streaming RAG pipeline.
    Yields tokens one by one as Qwen generates them.
    """
    # Step 1 — retrieve (fast, no streaming needed here)
    chunks = retrieve_chunks(query)

    # Step 2 — build prompt
    context = "\n\n---\n\n".join(
        f"[{c['source']} | {c['similarity']}]\n{c['text']}"
        for c in chunks
    ) if chunks else "No relevant documents found."

    # Build conversation history string
    history_text = ""
    if history:
        for msg in history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful AI assistant for TechCorp.
Answer using ONLY the context below. If not found, say so clearly.

CONTEXT:
{context}

{f"CONVERSATION:{chr(10)}{history_text}" if history_text else ""}
User: {query}
Assistant:"""

    # Step 3 — stream tokens
    yield from stream_tokens(prompt)