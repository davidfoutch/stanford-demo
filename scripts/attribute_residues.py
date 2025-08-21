#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch, numpy as np, pandas as pd
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from captum.attr import IntegratedGradients

class GATGraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, num_classes=2, edge_dim=2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, edge_dim=edge_dim, dropout=0.0)
        self.gat2 = GATConv(hidden*heads, hidden, heads=1, edge_dim=edge_dim, dropout=0.0)
        self.lin  = nn.Linear(hidden, num_classes)
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr).relu()
        x = self.gat2(x, edge_index, edge_attr).relu()
        g = global_mean_pool(x, batch)
        return self.lin(g)

def attribute_graph(model, data:Data, target:int):
    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()

    def fwd(inp):
        # inp shape: [N, F]
        out = model(inp, data.edge_index, data.edge_attr, data.batch if hasattr(data,'batch') else torch.zeros(inp.size(0), dtype=torch.long, device=inp.device))
        # return logits for target class
        return out[:, target]

    x = data.x.clone().detach().requires_grad_(True)
    ig = IntegratedGradients(fwd)
    attributions = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=32)  # [N,F]
    node_scores = attributions.abs().sum(dim=1)  # [N]
    node_scores = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min() + 1e-8)
    return node_scores.detach().cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gdir = Path(args.graphs)
    idx = pd.read_csv(gdir / "index.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load one sample to infer dims
    d0: Data = torch.load(idx.iloc[0]["path"])
    model = GATGraphClassifier(in_dim=d0.x.size(1), edge_dim=d0.edge_attr.size(1)).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    for _, row in idx.iterrows():
        d: Data = torch.load(row["path"])
        # target = its label (explain what drives its class)
        y = row["label"]; y = 0 if (isinstance(y,str) and y.upper() in ("A","0")) else (int(y) if not isinstance(y,str) else 1)
        scores = attribute_graph(model, d, target=y)  # [N]

        # map to chain:resi
        res_ids = d.res_ids  # saved in build step
        recs = []
        for rid, s in zip(res_ids, scores):
            chain, resi = rid.split(":")
            recs.append({"chain":chain, "resi":int(resi), "score":float(s)})

        pdb_id = getattr(d, "pdb_id", "unknown")
        out_csv = outdir / f"{pdb_id}_res_scores.csv"
        pd.DataFrame(recs).to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} ({len(recs)} residues)")

if __name__ == "__main__":
    main()
