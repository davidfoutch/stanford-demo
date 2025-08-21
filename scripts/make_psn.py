#!/usr/bin/env python3
import argparse
from collections import defaultdict
from typing import Iterable, Tuple, Optional, Set
import numpy as np, pandas as pd
from Bio.PDB import PDBParser, is_aa
from scipy.spatial import cKDTree

def iter_res_atom_coords(structure, chains: Optional[Set[str]] = None, heavy_only: bool = True):
    for model in structure:
        for chain in model:
            if chains and chain.id not in chains: continue
            for res in chain:
                if not is_aa(res, standard=True): continue
                resseq = res.get_id()[1]
                res_id = f"{chain.id}:{resseq}"
                for atom in res:
                    if heavy_only and atom.element == 'H': continue
                    yield res_id, atom.coord

def build_psn(pdb_path: str, cutoff: float = 4.5, chains=None, heavy_only: bool = True):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    chains = set(chains) if chains else None
    res_ids, coords = [], []
    for rid, c in iter_res_atom_coords(structure, chains, heavy_only):
        res_ids.append(rid); coords.append(c)
    coords = np.asarray(coords, dtype=float)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff)
    edge = defaultdict(list)
    for i,j in pairs:
        a,b = res_ids[i], res_ids[j]
        if a==b: continue
        u,v = (a,b) if a<b else (b,a)
        d = float(np.linalg.norm(coords[i]-coords[j]))
        edge[(u,v)].append(d)
    w_edges = [(u,v,len(ds)* (sum(ds)/len(ds)), len(ds), sum(ds)/len(ds)) for (u,v),ds in edge.items()]
    uw_edges= [(u,v) for (u,v) in edge.keys()]
    wdf = pd.DataFrame(w_edges, columns=["Residue1","Residue2","Weight","Contacts","AvgDist"])
    udf = pd.DataFrame(uw_edges, columns=["Residue1","Residue2"])
    return wdf, udf

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build residue-level PSN from a PDB file")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", action="append", help="repeat for multiple chains")
    ap.add_argument("--cutoff", type=float, default=4.5)
    ap.add_argument("--heavy-only", action="store_true", default=True)
    ap.add_argument("--out-weighted", default="weighted_psn.csv")
    ap.add_argument("--out-unweighted", default="unweighted_psn.csv")
    a = ap.parse_args()
    w,u = build_psn(a.pdb, a.cutoff, a.chain, a.heavy_only)
    w.to_csv(a.out_weighted, index=False)
    u.to_csv(a.out_unweighted, index=False)
    print(f"Wrote {a.out_weighted} ({len(w)} edges)")
    print(f"Wrote {a.out_unweighted} ({len(u)} edges)")