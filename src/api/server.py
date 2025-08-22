from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import requests
from pathlib import Path
from urllib.parse import quote

from src.companion.session import ChatSession
from src.companion.example_llm import llm_call

from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import Query, HTTPException
import pandas as pd, numpy as np, json
from Bio.PDB import PDBParser, is_aa
from scipy.spatial import cKDTree

app = FastAPI(title="Stanford Demo API")

def fetch_pdb_rcsb(pdb_id: str) -> str:
    pdb_id = pdb_id.upper()
    out = Path(f"data/cache/pdb/{pdb_id}.pdb")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        r = requests.get(url, timeout=20); r.raise_for_status()
        out.write_text(r.text)
    return str(out)

def _build_psn_weighted(pdb_path: str, cutoff: float = 4.5, chains=None, heavy_only: bool = True) -> pd.DataFrame:
    parser = PDBParser(QUIET=True); s = parser.get_structure("s", pdb_path)
    chains = set(chains) if chains else None
    res_ids, coords = [], []
    for model in s:
        for ch in model:
            if chains and ch.id not in chains: continue
            for res in ch:
                if not is_aa(res, standard=True): continue
                for atom in res:
                    if heavy_only and atom.element == "H": continue
                    res_ids.append(f"{ch.id}:{res.get_id()[1]}"); coords.append(atom.coord)
    coords = np.asarray(coords, dtype=float)
    tree = cKDTree(coords); pairs = tree.query_pairs(r=cutoff)
    from collections import defaultdict
    edge = defaultdict(list)
    for i,j in pairs:
        a,b = res_ids[i], res_ids[j]
        if a==b: continue
        u,v = (a,b) if a<b else (b,a)
        d = float(np.linalg.norm(coords[i]-coords[j]))
        edge[(u,v)].append(d)
    rows = []
    for (u,v), ds in edge.items():
        cnt = len(ds); avg = sum(ds)/cnt; w = cnt*avg
        rows.append((u,v,w,cnt,avg))
    return pd.DataFrame(rows, columns=["Residue1","Residue2","Weight","Contacts","AvgDist"])

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/viz/psn_build", response_class=HTMLResponse)
def viz_psn_build(pdb_id: str, chains: str="A", cutoff: float=4.5, kmin: int=0, style: str="cartoon"):
    pdb_path = fetch_pdb_rcsb(pdb_id)
    chain_list = [c.strip() for c in chains.split(",") if c.strip()]
    wdf = _build_psn_weighted(pdb_path, cutoff=cutoff, chains=chain_list, heavy_only=True)

    out_csv = Path(f"artifacts/runs/psn/{pdb_id}_c{cutoff}_{'-'.join(chain_list) or 'ALL'}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wdf.to_csv(out_csv, index=False)

    # Reuse the existing overlay viewer
    return viz_psn_net(pdb_path=pdb_path, edges_csv=str(out_csv), kmin=kmin, style=style)

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

def fetch_pdb_rcsb(pdb_id: str) -> str:
    pdb_id = pdb_id.upper()
    out = Path(f"data/cache/pdb/{pdb_id}.pdb")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        out.write_text(r.text)
    return str(out)

def _build_psn_weighted(pdb_path: str, cutoff: float = 4.5, chains=None, heavy_only: bool = True) -> pd.DataFrame:
    parser = PDBParser(QUIET=True); s = parser.get_structure("s", pdb_path)
    chains = set(chains) if chains else None
    res_ids, coords = [], []
    for model in s:
        for ch in model:
            if chains and ch.id not in chains: continue
            for res in ch:
                if not is_aa(res, standard=True): continue
                for atom in res:
                    if heavy_only and atom.element == "H": continue
                    res_id = f"{ch.id}:{res.get_id()[1]}"
                    res_ids.append(res_id); coords.append(atom.coord)
    coords = np.asarray(coords, dtype=float)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff)
    from collections import defaultdict
    edge = defaultdict(list)
    for i,j in pairs:
        a,b = res_ids[i], res_ids[j]
        if a==b: continue
        u,v = (a,b) if a<b else (b,a)
        d = float(np.linalg.norm(coords[i]-coords[j]))
        edge[(u,v)].append(d)
    rows = []
    for (u,v), ds in edge.items():
        cnt = len(ds); avg = sum(ds)/cnt; w = cnt*avg
        rows.append((u,v,w,cnt,avg))
    return pd.DataFrame(rows, columns=["Residue1","Residue2","Weight","Contacts","AvgDist"])


# --- ROUTE A: overlay a given PDB path + edges CSV ---
@app.get("/viz/psn_net", response_class=HTMLResponse)
def viz_psn_net(
    pdb_path: str = Query(...),
    edges_csv: str = Query(...),
    kmin: int = 0,
    style: str = "cartoon",
    weight_by: str = "Contacts",   # or "Weight"
    rmin: float = 0.06,
    rmax: float = 0.35,
    res_scores_csv: str = Query(None),
    topk: int = 25,
    node_rmin: float = 0.4,
    node_rmax: float = 1.2,
):
    try:
        df = pd.read_csv(edges_csv)
        u, v = ('u','v') if {'u','v'}.issubset(df.columns) else ('Residue1','Residue2')
        if kmin and 'Contacts' in df.columns:
            df = df[df['Contacts'] >= kmin]

        coords = _ca_coords(pdb_path)
        hasC = 'Contacts' in df.columns
        hasW = 'Weight'   in df.columns
        metric = weight_by if (weight_by in df.columns) else ("Contacts" if hasC else ("Weight" if hasW else None))

        cyl = []
        for _, row in df.iterrows():
            a, b = row[u], row[v]
            if a in coords and b in coords:
                x1, y1, z1 = coords[a]
                x2, y2, z2 = coords[b]
                C = int(row['Contacts']) if hasC else 1
                W = float(row['Weight']) if hasW else 1.0
                M = float(row[metric]) if metric else 1.0
                cyl.append({"x1":x1,"y1":y1,"z1":z1,"x2":x2,"y2":y2,"z2":z2,"C":C,"W":W,"M":M})

        edges_js = json.dumps(cyl)
        pdb_js   = json.dumps(Path(pdb_path).read_text())

        res_js = "null"
        if res_scores_csv:
            dfres = pd.read_csv(res_scores_csv)
            # keep top-K by score
            dfres = dfres.sort_values("score", ascending=False).head(topk)
            res_list = [{"chain":str(r.chain), "resi":int(r.resi), "s":float(r.score)} for r in dfres.itertuples(index=False)]
            res_js = json.dumps(res_list)


        style_map = {
            "cartoon": {"cartoon": {"color": "gray"}},
            "stick":   {"stick": {}},
            "surface": {"surface": {"opacity": 0.6}},
        }
        style_obj_js = json.dumps(style_map.get(style, style_map["cartoon"]))

        html = """
<!doctype html><html><head>
<meta charset="utf-8" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.5/3Dmol-min.js"></script>
<style>html,body,#viewer{height:100%%;margin:0}#viewer{width:100%%;height:100%%}</style>
</head><body><div id="viewer"></div>
<script>
let v=$3Dmol.createViewer(document.getElementById('viewer'),{backgroundColor:'white'});
v.addModel(%(pdb)s,"pdb");
v.setStyle({}, %(style)s);
v.addSurface($3Dmol.VDW, {opacity:0.6, color:'white'});

let E=%(edges)s;
const vals = E.map(e => e.M || 1);
const vmin = Math.min(...vals), vmax = Math.max(...vals);
const span = (vmax>vmin) ? (vmax - vmin) : 1;

function heatColor(t){
  var r=255, g=Math.round(255-180*t), b=Math.round(50*(1-t));
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}

E.sort((a,b) => (a.M||1) - (b.M||1));
for (const e of E){
  var t = ((e.M || 1) - vmin) / span;
  var radius = %(rmin)f + (%(rmax)f - %(rmin)f) * t;
  v.addCylinder({
    start:{x:e.x1,y:e.y1,z:e.z1},
    end:  {x:e.x2,y:e.y2,z:e.z2},
    radius: radius,
    fromCap:1, toCap:1,
    color: heatColor(t)
  });
}

// Residue highlights by score
let R = %(res)s;
if (R){
  const svals = R.map(r => r.s);
  const smin = Math.min(...svals), smax = Math.max(...svals), sspan = (smax>smin)?(smax-smin):1;
  function nodeColor(t){ var r=255, g=Math.round(255-200*t), b=0; return 'rgb(' + r + ',' + g + ',' + b + ')'; }
  for (const r of R){
    var t = (r.s - smin)/sspan;
    var rad = %(node_rmin)f + (%(node_rmax)f - %(node_rmin)f) * t;
    v.setStyle({chain:r.chain, resi:r.resi}, {sphere:{radius:rad, color: nodeColor(t)}});
  }
}

// Save PNG
const btn=document.createElement('button');
btn.textContent='Save PNG';
btn.style.position='absolute'; btn.style.top='10px'; btn.style.right='10px';
document.body.appendChild(btn);
btn.onclick=function(){ var a=document.createElement('a'); a.href=v.pngURI(); a.download='psn_overlay.png'; a.click(); };

v.zoomTo(); v.render();
</script></body></html>
""" % {"pdb": pdb_js, 
       "style": style_obj_js, 
       "edges": edges_js, 
       "rmin": rmin, 
       "rmax": rmax,
       "res": res_js,         # <-- adds data for residue scores
       "node_rmin": node_rmin,# <-- for node (sphere) min radius
       "node_rmax": node_rmax # <-- for node (sphere) max radius
       }

        return HTMLResponse(html)
    except Exception as ex:
        return HTMLResponse(f"<pre>Error: {ex}</pre>", status_code=500)


# --- ROUTE B: build from PDB id, then reuse A ---
@app.get("/viz/psn_build", response_class=HTMLResponse)
def viz_psn_build(
    pdb_id: str,
    chains: str = "A",
    cutoff: float = 4.5,
    kmin: int = 0,
    style: str = "cartoon",
):
    BASE = Path(__file__).resolve().parents[2]   # /home/david/stanford-demo
    ART = BASE / "artifacts" / "runs" / "psn"
    ART.mkdir(parents=True, exist_ok=True)

    # 1) Fetch/cache PDB -> absolute path string
    try:
        pdb_path = Path(fetch_pdb_rcsb(pdb_id)).resolve()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RCSB fetch failed for {pdb_id}: {e}")

    chain_list = [c.strip() for c in chains.split(",") if c.strip()]

    # 2) Build PSN (heavy atoms only) and write edges CSV
    try:
        wdf = _build_psn_weighted(str(pdb_path), cutoff=cutoff, chains=chain_list, heavy_only=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PSN build failed for {pdb_id} chains={chain_list}: {e}")

    edges_csv_path = (ART / f"test_{pdb_id}.csv").resolve()
    wdf.to_csv(edges_csv_path, index=False)  # do NOT reassign this variable

    # 3) Redirect to /viz/psn_net with URL-encoded absolute paths
    url = (
        "/viz/psn_net"
        f"?pdb_path={quote(str(pdb_path))}"
        f"&edges_csv={quote(str(edges_csv_path))}"
        f"&kmin={kmin}"
        f"&style={quote(style)}"
    )
    return RedirectResponse(url)