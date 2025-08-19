import argparse, json
from pathlib import Path
import pandas as pd, networkx as nx, matplotlib.pyplot as plt

def main(bundle, outdir):
    bdir = Path("data") / bundle
    edges_csv = bdir / "psn_edges.csv"
    meta_json = bdir / "metadata.json"
    meta = json.loads(meta_json.read_text()) if meta_json.exists() else {}

    df = pd.read_csv(edges_csv)  # cols: u,v[,w]
    G = nx.Graph()
    if {"u","v","w"}.issubset(df.columns):
        G.add_weighted_edges_from(df[["u","v","w"]].itertuples(index=False))
    else:
        G.add_edges_from(df[["u","v"]].itertuples(index=False))

    pos = nx.spring_layout(G, seed=42)
    plt.figure()
    nx.draw(G, pos, node_size=18, width=0.5, with_labels=False)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    png = out / f"{bundle}_psn.png"
    plt.title(f"{bundle} PSN (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    print(f"PNG: {png}")

    pdb_id = meta.get("pdb_id","")
    hi = ",".join(meta.get("default_highlight", []))
    style = meta.get("default_style","cartoon")
    if pdb_id:
        print(f"3D viewer: /viz/psn?pdb_id={pdb_id}&highlight={hi}&style={style}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default="artifacts/figs")
    args = ap.parse_args()
    main(args.bundle, args.out)
