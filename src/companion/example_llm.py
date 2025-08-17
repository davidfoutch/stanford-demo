# example_llm.py
import os
import requests

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8001/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # optional for local servers

def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    r = requests.post(LLM_ENDPOINT, json=payload, headers=headers, timeout=45)
    r.raise_for_status()
    data = r.json()
    # OpenAI-compatible shape
    return data["choices"][0]["message"]["content"].strip()
