#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np, networkx as nx
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser, is_aa
from scipy.spatial import cKDTree

AA20 = {
    'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,'GLN':5,'GLU':6,'GLY':7,'HIS':8,'ILE':9,
    'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,'SER':15,'THR':16,'TRP':17,'TYR':18,'VAL':19
}

def fetch_pdb_rcsb(pdb_id:str, cache_dir=Path("data/cache/pdb"))->Path:
    import requests
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pdb_id.upper()}.pdb"
    if not out.exists():
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=20)
        r.raise_for_status(); out.write_text(r.text)
    return out

def build_psn_weighted(pdb_path:Path, cutoff:float=4.5, chains=None, heavy_only=True)->pd.DataFrame:
    parser = PDBParser(QUIET=True); s = parser.get_structure("s", str(pdb_path))
    chains = set(chains) if chains else None
    res_ids, coords = [], []
    for model in s:
        for ch in model:
            if chains and ch.id not in chains: continue
            for res in ch:
                if not is_aa(res, standard=True): continue
                for atom in res:
                    if heavy_only and atom.element == 'H': continue
                    res_ids.append(f"{ch.id}:{res.get_id()[1]}"); coords.append(atom.coord)
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0: return pd.DataFrame(columns=["Residue1","Residue2","Weight","Contacts","AvgDist"])
    tree = cKDTree(coords); pairs = tree.query_pairs(r=cutoff)
    from collections import defaultdict
    edge = defaultdict(list)
    for i,j in pairs:
        a,b = res_ids[i], res_ids[j]
        if a==b: continue
        u,v = (a,b) if a<b else (b,a)
        d = float(np.linalg.norm(coords[i]-coords[j]))
        edge[(u,v)].append(d)
    rows=[]
    for (u,v), ds in edge.items():
        cnt=len(ds); avg=sum(ds)/cnt; w=cnt*avg
        rows.append((u,v,w,cnt,avg))
    return pd.DataFrame(rows, columns=["Residue1","Residue2","Weight","Contacts","AvgDist"])

def parse_res_meta(pdb_path:Path):
    """Return dicts: resname['A:100']->'ASP', resseq['A:100']->100, bfac['A:100']->float."""
    resname, resseq, bfac = {}, {}, {}
    s = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    for model in s:
        for ch in model:
            for res in ch:
                if not is_aa(res, standard=True): continue
                rid = f"{ch.id}:{res.get_id()[1]}"
                resname[rid] = res.get_resname()
                resseq[rid] = res.get_id()[1]
                if "CA" in res:
                    bfac[rid] = float(res["CA"].get_bfactor())
    return resname, resseq, bfac

def residue_features(node_ids, resname, resseq, deg, cluster, bfac):
    N = len(node_ids)
    X = np.zeros((N, 20 + 1 + 1 + 1 + 1), dtype=np.float32)  # AA20 + idx_norm + degree + cluster + bfac_norm
    # per-chain scaling for residue index
    chains = {}
    for rid in node_ids:
        c, i = rid.split(":"); i = int(i)
        chains.setdefault(c, []).append(i)
    chain_minmax = {c: (min(v), max(v)) for c,v in chains.items()}
    # bfactor scaling
    bvals = [bfac.get(rid, 0.0) for rid in node_ids]
    bmin, bmax = (min(bvals), max(bvals)) if bvals else (0.0, 1.0)
    span_b = (bmax - bmin) if bmax > bmin else 1.0

    for idx, rid in enumerate(node_ids):
        # AA one-hot
        aa = resname.get(rid, "UNK")
        if aa in AA20: X[idx, AA20[aa]] = 1.0
        # idx_norm
        c, i = rid.split(":"); i = int(i)
        mn, mx = chain_minmax[c]; span = (mx - mn) if mx > mn else 1
        X[idx, 20] = (i - mn) / span
        # degree
        X[idx, 21] = deg.get(rid, 0) / max(1, len(node_ids)-1)
        # clustering
        X[idx, 22] = cluster.get(rid, 0.0)
        # bfactor norm
        X[idx, 23] = (bfac.get(rid, bmin) - bmin) / span_b
    return X

def make_graph(pdb_id:str, chains:str, cutoff:float, outdir:Path):
    pdb_path = fetch_pdb_rcsb(pdb_id)
    chain_list = [c.strip() for c in chains.split(",") if c.strip()]
    psn = build_psn_weighted(pdb_path, cutoff=cutoff, chains=chain_list, heavy_only=True)
    if len(psn)==0: return None

    # nodes
    nodes = sorted(set(psn["Residue1"]).union(set(psn["Residue2"])))
    idx = {rid:i for i,rid in enumerate(nodes)}

    # edges
    u = psn["Residue1"].map(idx).to_numpy()
    v = psn["Residue2"].map(idx).to_numpy()
    edge_index = np.vstack([np.concatenate([u,v]), np.concatenate([v,u])])  # undirected
    contacts = np.concatenate([psn["Contacts"].to_numpy(), psn["Contacts"].to_numpy()]).astype(np.float32)
    weight   = np.concatenate([psn["Weight"].to_numpy(),   psn["Weight"].to_numpy()]).astype(np.float32)
    edge_attr = np.stack([contacts, weight], axis=1)

    # node features
    resname, resseq, bfac = parse_res_meta(pdb_path)
    G = nx.Graph(); G.add_edges_from(zip(psn["Residue1"], psn["Residue2"]))
    deg = dict(G.degree())
    cluster = nx.clustering(G)
    X = residue_features(nodes, resname, resseq, deg, cluster, bfac)

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=None,  # label set later
    )
    data.res_ids = nodes  # keep mapping for attribution
    data.pdb_id = pdb_id
    data.chains = chains
    data.cutoff = cutoff

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{pdb_id}_{chains.replace(',','')}_c{cutoff:.1f}.pt"
    torch.save(data, out_path)
    print(f"Wrote {out_path}  x={tuple(data.x.shape)}  E={data.edge_index.size(1)}")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="CSV: pdb_id,chains,label")
    ap.add_argument("--out", default="artifacts/graphs")
    ap.add_argument("--cutoff", type=float, default=4.5)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    outdir = Path(args.out)
    paths = []
    for _, row in labels.iterrows():
        p = make_graph(row["pdb_id"], str(row["chains"]), args.cutoff, outdir)
        if p: paths.append((p, row["label"]))
    # write an index for training
    idx_csv = outdir / "index.csv"
    pd.DataFrame([{"path":p, "label":lbl} for p,lbl in paths]).to_csv(idx_csv, index=False)
    print(f"Index: {idx_csv}")

if __name__ == "__main__":
    main()
