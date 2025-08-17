from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os

from src.companion.session import ChatSession
from src.companion.example_llm import llm_call

app = FastAPI(title="Stanford Demo API")

@app.get("/health")
def health():
    return {"ok": True}

# --- Companion ---
class ChatIn(BaseModel):
    text: str
    use_llm: Optional[bool] = False
    k: Optional[int] = 5
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 600

# simple in-process session for the demo
_session = ChatSession()

@app.post("/companion/chat")
def companion_chat(req: ChatIn):
    # Update session config based on request
    _session.k = req.k or 5
    _session.temperature = req.temperature or 0.2
    _session.max_tokens = req.max_tokens or 600
    _session.llm_fn = (lambda sys, usr: llm_call(sys, usr, _session.temperature, _session.max_tokens)) if req.use_llm else None

    reply = _session.chat(req.text)
    return {"reply": reply}

@app.get("/companion/config")
def companion_config():
    return {
        "mode": "llm" if _session.llm_fn else "deterministic",
        "k": _session.k,
        "temperature": getattr(_session, "temperature", None),
        "max_tokens": getattr(_session, "max_tokens", None),
        "llm_endpoint": os.getenv("LLM_ENDPOINT"),
        "llm_model": os.getenv("LLM_MODEL"),
    }