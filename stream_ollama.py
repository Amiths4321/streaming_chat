# stream_ollama.py
# The core streaming function — used by everything else

import requests
import json
from typing import Generator

OLLAMA_HOST  = "http://10.22.39.192:11434"
OLLAMA_MODEL = "qwen2.5vl:latest"


def stream_tokens(
    prompt:     str,
    max_tokens: int   = 1024,
    temperature: float = 0.2
) -> Generator[str, None, None]:
    """
    Stream tokens from Ollama one by one.
    Yields each token as a string as it arrives.

    Usage:
        for token in stream_tokens("What is AI?"):
            print(token, end="", flush=True)
    """
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  True,          # ← THE KEY CHANGE
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        },
        stream=True,                  # ← keep connection open
        timeout=300
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data  = json.loads(line.decode("utf-8"))
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break
        except json.JSONDecodeError:
            continue


def stream_chat(
    messages:    list[dict],
    max_tokens:  int   = 1024,
    temperature: float = 0.2
) -> Generator[str, None, None]:
    """
    Stream tokens for a multi-turn conversation.
    messages = [{"role": "user/assistant", "content": "..."}]
    """
    # Build prompt from message history
    prompt = ""
    for msg in messages:
        role    = msg["role"].upper()
        content = msg["content"]
        prompt += f"\n{role}:\n{content}\n"
    prompt += "\nASSISTANT:\n"

    yield from stream_tokens(prompt, max_tokens, temperature)


def collect_stream(generator: Generator) -> str:
    """Collect all tokens from a stream into a single string."""
    return "".join(generator)