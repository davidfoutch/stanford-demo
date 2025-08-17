# src/companion/session.py

import json
import re
from datetime import datetime
from typing import Callable, List, Optional
from .memory import MemoryStore  # your existing class/module

def _safe_meta(meta: dict | None) -> dict:
    """Ensure all metadata values are scalars for ChromaDB."""
    out = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out

def _compress(s: str, n: int = 180) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else (s[: n - 1] + "…")

def _dedupe_keep_order(lines: List[str]) -> List[str]:
    seen, out = set(), []
    for ln in lines:
        t = ln.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

class ChatSession:
    def __init__(
        self,
        store: MemoryStore | None = None,
        k: int = 5,
        llm_fn: Optional[Callable[[str, str], str]] = None,  # (system, user) -> reply
    ):
        self.mem = store or MemoryStore()
        self.k = k
        self.llm_fn = llm_fn  # leave as None for deterministic mode

    # --- retrieval helpers ---
    def _extract_hits(self, res, k: int) -> List[str]:
        """
        Normalize different possible shapes from MemoryStore.search into List[str].
        Expected common dict shape: {"documents": [["...", "..."]], ...}
        """
        try:
            if isinstance(res, dict) and "documents" in res:
                docs = res.get("documents") or []
                if isinstance(docs, list) and docs:
                    return [str(x) for x in docs[0][:k]]
                return []
            if isinstance(res, list) and res and isinstance(res[0], list):
                return [str(x) for x in res[0][:k]]
            if isinstance(res, list):
                return [str(x) for x in res[:k]]
        except Exception:
            pass
        return []

    def _summarize_hits(self, hits: List[str], max_bullets: int = 5) -> List[str]:
        bullets = []
        for h in hits:
            if not h:
                continue
            s = re.sub(r"^(USER|ASSISTANT):\s*", "", str(h).strip(), flags=re.I)
            if len(s) < 4:
                continue
            bullets.append(_compress(s, 180))
        return _dedupe_keep_order(bullets)[:max_bullets]

    # --- main entrypoint ---
    def chat(self, text: str) -> str:
        # 1) retrieve context
        raw = self.mem.search(text, k=self.k)
        hits = self._extract_hits(raw, k=self.k)
        bullets = self._summarize_hits(hits)

        # 2) generate reply
        if self.llm_fn:
            # LLM path (expects a system + user prompt)
            system_prompt = (
                "You are a local scientific assistant for computational structural biology. "
                "Use prior context only if clearly relevant. If the new query contradicts past notes, say so and "
                "prefer the new query. Answer concisely, then give a short 'Next actions' list (<=3 bullets). "
                "Never invent file paths or versions; if unsure, say how to verify."
            )
            ctx = "\n".join(f"- {b}" for b in bullets) if bullets else "- (no prior context found)"
            user_prompt = (
                f"Prior context (top-{self.k}):\n{ctx}\n\n"
                f"User query:\n{text}\n\n"
                "Format: direct answer (2–5 sentences) then 'Next actions' (<=3 bullets)."
            )
            body = self.llm_fn(system_prompt, user_prompt).strip()
            reply = f"[seen {len(hits)} related]\n{body}"
        else:
            # Deterministic path
            ctx = "\n".join(f"- {b}" for b in bullets) if bullets else "(no related context)"
            reply = (
                f"[seen {len(hits)} related]\n"
                f"Here’s what I remember:\n{ctx}\n\n"
                f"Reply:\n{text}\n\n"
                "Next actions:\n- Tell me if I should change code or config\n- Name the file and exact change\n- (Optional) set k/context size"
            )

        # 3) persist both sides (with Chroma-safe metadata)
        now = datetime.utcnow().isoformat()
        self.mem.add(f"USER: {text}", _safe_meta({"role": "user", "ts": now}))
        self.mem.add(
            f"ASSISTANT: {reply}",
            _safe_meta({
                "role": "assistant",
                "ts": now,
                "ctx_count": len(hits),
                "ctx_preview": hits[:2],   # will be JSON-encoded by _safe_meta
                "mode": "llm" if self.llm_fn else "deterministic",
            }),
        )
        return reply
