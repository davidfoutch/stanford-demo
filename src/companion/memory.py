from pathlib import Path
from typing import Dict, Any
import chromadb

class MemoryStore:
    """
    Minimal persistent text memory using Chroma.
    - add(text, meta): store an item with metadata
    - search(query, k): retrieve top-k similar texts
    """
    def __init__(self, path: str = "artifacts/runs/chroma"):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path)
        self.col = self.client.get_or_create_collection("memories")

    def add(self, text: str, meta: Dict[str, Any] | None = None):
        meta = meta or {}
        _id = meta.get("id") or str(abs(hash(text)))
        self.col.add(documents=[text], metadatas=[meta], ids=[_id])

    def search(self, query: str, k: int = 5):
        return self.col.query(query_texts=[query], n_results=k)
