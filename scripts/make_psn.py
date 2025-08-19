# Create a cleaned, standalone script based on the notebook's intent.
# It builds a residue-level PSN by counting inter-residue atom-atom contacts within a cutoff
# and computing a simple weight = (#contacts) * (avg distance). It outputs weighted and unweighted CSVs.
from pathlib import Path

script = r'''#!/usr/bin/env python3
"""
make_psn.py — Build a Protein Structure Network (PSN) from a PDB file.

Nodes   = residues (e.g., A:100)
Edges   = pairs of residues with >=1 atom-atom contact within cutoff
Weight  = (#contacts) * (average distance of those contacts)

Deps: biopython, numpy, pandas, scipy
    pip install biopython numpy pandas scipy

Example:
    python make_psn.py --pdb 1yok.pdb --chain A --cutoff 4.5 \
        --out-weighted weighted_psn.csv --out-unweighted unweighted_psn.csv
"""

import argparse
from collections import defaultdict
from typing import Iterable, Tuple, Optional, Set

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, is_aa
from scipy.spatial import cKDTree


def iter_residue_atom_coords(structure, chains: Optional[Set[str]] = None, heavy_only: bool = True
                             ) -> Iterable[Tuple[str, np.ndarray]]:
    """
    Yield (res_id, atom_coord) for atoms in the selected chains.
    res_id is like 'A:100' (chain:resseq). Inserts no insertion codes for simplicity.
    """
    for model in structure:
        for chain in model:
            if chains and chain.id not in chains:
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                resseq = res.get_id()[1]  # (hetero flag, resseq, icode) -> resseq
                res_id = f"{chain.id}:{resseq}"
                for atom in res:
                    if heavy_only and atom.element == 'H':
                        continue
                    yield res_id, atom.coord


def generate_weighted_and_unweighted_psn(pdb_path: str, distance_cutoff: float = 4.5,
                                         chains: Optional[Iterable[str]] = None,
                                         heavy_only: bool = True):
    """
    Build residue-level PSN using atom-atom contacts within cutoff.
    Returns (weighted_df, unweighted_df).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)

    chains = set(chains) if chains else None
    res_ids = []
    coords = []

    for res_id, coord in iter_residue_atom_coords(structure, chains=chains, heavy_only=heavy_only):
        res_ids.append(res_id)
        coords.append(coord)

    if not coords:
        raise ValueError("No atom coordinates collected. Check chain filter and PDB path.")

    coords = np.asarray(coords, dtype=float)
    tree = cKDTree(coords)
    # All unique atom-atom pairs within cutoff
    pairs = tree.query_pairs(r=distance_cutoff)

    edge_data = defaultdict(list)  # (res_i, res_j) -> list of distances

    for i, j in pairs:
        ri, rj = res_ids[i], res_ids[j]
        if ri == rj:
            continue
        # canonical order to avoid duplicate edges
        a, b = (ri, rj) if ri < rj else (rj, ri)
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        edge_data[(a, b)].append(dist)

    weighted_edges = []
    unweighted_edges = []

    for (a, b), dists in edge_data.items():
        count = len(dists)
        avg_dist = sum(dists) / count
        weight = count * avg_dist
        weighted_edges.append((a, b, weight, count, avg_dist))
        unweighted_edges.append((a, b))

    weighted_df = pd.DataFrame(weighted_edges, columns=["Residue1", "Residue2", "Weight", "Contacts", "AvgDist"])
    unweighted_df = pd.DataFrame(unweighted_edges, columns=["Residue1", "Residue2"])
    return weighted_df, unweighted_df


def main():
    ap = argparse.ArgumentParser(description="Build a residue-level PSN from a PDB file.")
    ap.add_argument("--pdb", required=True, help="Path to local PDB file (e.g., 1yok.pdb)")
    ap.add_argument("--chain", action="append", help="Chain(s) to include (use multiple --chain for more). Default: all")
    ap.add_argument("--cutoff", type=float, default=4.5, help="Distance cutoff in Å (atom-atom)")
    ap.add_argument("--heavy-only", action="store_true", default=True, help="Ignore hydrogens (default: True)")
    ap.add_argument("--out-weighted", default="weighted_psn.csv", help="Output CSV for weighted edges")
    ap.add_argument("--out-unweighted", default="unweighted_psn.csv", help="Output CSV for unweighted edges")
    args = ap.parse_args()

    wdf, udf = generate_weighted_and_unweighted_psn(
        pdb_path=args.pdb,
        distance_cutoff=args.cutoff,
        chains=args.chain,
        heavy_only=args.heavy_only
    )
    wdf.to_csv(args.out_weighted, index=False)
    udf.to_csv(args.out_unweighted, index=False)
    print(f"Wrote: {args.out_weighted} ({len(wdf)} edges)")
    print(f"Wrote: {args.out_unweighted} ({len(udf)} edges)")


if __name__ == "__main__":
    main()
'''
out_path = Path('/mnt/data/make_psn.py')
out_path.write_text(script, encoding='utf-8')
print(f"Saved script to {out_path}")