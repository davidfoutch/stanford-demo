from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
from pathlib import Path

from src.companion.session import ChatSession
from src.companion.example_llm import llm_call

from fastapi.responses import HTMLResponse
from fastapi import Query
import pandas as pd, json
from Bio.PDB import PDBParser, is_aa

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

def _ca_coords(pdb_path: str):
    s = PDBParser(QUIET=True).get_structure("s", pdb_path)
    out = {}
    for model in s:
        for ch in model:
            for res in ch:
                if not is_aa(res, standard=True): 
                    continue
                if "CA" in res:
                    out[f"{ch.id}:{res.get_id()[1]}"] = res["CA"].coord.tolist()
    return out

@app.get("/viz/psn_net", response_class=HTMLResponse)
def viz_psn_net(
    pdb_path: str = Query(..., description="Absolute path to local PDB"),
    edges_csv: str = Query(..., description="CSV with u,v or Residue1,Residue2 (+ optional Contacts/Weight)"),
    kmin: int = 0,
    style: str = "cartoon",
):
    try:
        df = pd.read_csv(edges_csv)
        u, v = ('u','v') if {'u','v'}.issubset(df.columns) else ('Residue1','Residue2')
        if kmin and 'Contacts' in df.columns:
            df = df[df['Contacts'] >= kmin]

        coords = _ca_coords(pdb_path)
        cyl = []
        for _, r in df.iterrows():
            a, b = r[u], r[v]
            if a in coords and b in coords:
                (x1,y1,z1) = coords[a]
                (x2,y2,z2) = coords[b]
                cyl.append(dict(x1=x1,y1=y1,z1=z1, x2=x2,y2=y2,z2=z2))

        edges_js = json.dumps(cyl)
        pdb_js   = json.dumps(Path(pdb_path).read_text())

        style_map = {
            "cartoon": {"cartoon": {"color": "gray"}},
            "stick":   {"stick": {}},
            "surface": {"surface": {"opacity": 0.6}},
        }
        style_obj_js = json.dumps(style_map.get(style, style_map["cartoon"]))

        html = f"""<!doctype html><html><head>
<meta charset="utf-8" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.5/3Dmol-min.js"></script>
<style>html,body,#viewer{{height:100%;margin:0}}#viewer{{width:100%;height:100%}}</style>
</head><body><div id="viewer"></div>
<script>
let v=$3Dmol.createViewer(document.getElementById('viewer'),{{backgroundColor:'white'}});
v.addModel({pdb_js},"pdb");                 // embed PDB text (not a path)
v.setStyle({{}}, {style_obj_js});
v.addSurface($3Dmol.VDW, {{opacity:0.6, color:'white'}});
let E={edges_js};
for (const e of E) {{
  v.addCylinder({{
    start:{{x:e.x1,y:e.y1,z:e.z1}},
    end:  {{x:e.x2,y:e.y2,z:e.z2}},
    radius:0.12, fromCap:1, toCap:1, color:'red'
  }});
}}
v.zoomTo(); v.render();
</script></body></html>"""
        return HTMLResponse(html)
    except Exception as ex:
        return HTMLResponse(f"<pre>Error: {ex}</pre>", status_code=500)