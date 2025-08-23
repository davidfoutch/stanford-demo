import argparse, json
from pathlib import Path
import torch, torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

# --------- safe graph loader (PyTorch 2.6/2.7) ----------
def load_graph(path: str) -> Data:
    obj = torch.load(path, map_location="cpu", weights_only=False)  # trusted local artifacts
    if isinstance(obj, dict) and "edge_index" in obj:
        return Data(**{k: v for k, v in obj.items() if v is not None})
    return obj

# --------- same GAT as trainer (edge_attr optional) ----------
class GATGraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, num_classes=2, edge_dim=None, dropout=0.1):
        super().__init__()
        if edge_dim is None:
            self.gat1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
            self.gat2 = GATConv(hidden*heads, hidden, heads=1, dropout=dropout)
        else:
            self.gat1 = GATConv(in_dim, hidden, heads=heads, edge_dim=edge_dim, dropout=dropout)
            self.gat2 = GATConv(hidden*heads, hidden, heads=1, edge_dim=edge_dim, dropout=dropout)
        self.lin = nn.Linear(hidden, num_classes)
        self.use_edges = edge_dim is not None

    def forward(self, x, edge_index, edge_attr, batch):
        if self.use_edges and edge_attr is not None:
            x = self.gat1(x, edge_index, edge_attr).relu()
            x = self.gat2(x, edge_index, edge_attr).relu()
        else:
            x = self.gat1(x, edge_index).relu()
            x = self.gat2(x, edge_index).relu()
        g = global_mean_pool(x, batch)
        return self.lin(g)

# --------- vanilla Integrated Gradients over node features ----------
@torch.no_grad()
def _make_batch_vec(n, device):
    return torch.zeros(n, dtype=torch.long, device=device)

def integrated_gradients(model, data: Data, target: int = 1, steps: int = 32, device="cpu"):
    model.eval()
    # prepare tensors
    x0 = torch.zeros_like(data.x, device=device)
    x1 = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr  = data.edge_attr.to(device) if getattr(data, "edge_attr", None) is not None else None
    batch      = data.batch.to(device) if getattr(data, "batch", None) is not None else _make_batch_vec(x1.size(0), device)

    total_grad = torch.zeros_like(x1)
    for alpha in torch.linspace(1.0/steps, 1.0, steps, device=device):
        x = (x0 + alpha * (x1 - x0)).detach().requires_grad_(True)
        logits = model(x, edge_index, edge_attr, batch)
        logit  = logits[:, target].sum()
        for p in model.parameters():
            if p.grad is not None: p.grad = None
        if x.grad is not None: x.grad = None
        logit.backward()
        total_grad += x.grad

    ig = (x1 - x0) * (total_grad / steps)
    node_scores = ig.abs().sum(dim=1).detach().cpu().numpy()
    return node_scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="Path to a single .pt graph file")
    ap.add_argument("--model", required=True, help="Path to best_model.pt")
    ap.add_argument("--out", required=True, help="Output CSV path (res_scores.csv)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--target", type=int, default=1, help="class index to attribute (1 = 'B')")
    ap.add_argument("--steps", type=int, default=32)
    args = ap.parse_args()

    # load graph
    g: Data = load_graph(args.graph)
    g.x = g.x.float()
    if getattr(g, "edge_attr", None) is not None:
        g.edge_attr = g.edge_attr.float()

    in_dim  = g.x.size(1)
    edge_dim = g.edge_attr.size(1) if getattr(g, "edge_attr", None) is not None else None

    # model (mirror trainer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATGraphClassifier(in_dim, hidden=args.hidden, heads=args.heads,
                               num_classes=2, edge_dim=edge_dim, dropout=0.2).to(device)
    sd = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(sd)

    # IG node scores
    scores = integrated_gradients(model, g, target=args.target, steps=args.steps, device=device)

    # residue labels if present on Data; fall back to node index
    res_ids = None
    for k in ("res_id", "res_ids", "residue_ids", "residue"):
        if hasattr(g, k):
            res_ids = getattr(g, k)
            break
    if res_ids is None:
        residue = [f"{i}" for i in range(len(scores))]
    else:
        if isinstance(res_ids, (list, tuple)):
            residue = [str(x) for x in res_ids]
        elif torch.is_tensor(res_ids):
            residue = [str(x) for x in res_ids.tolist()]
        else:
            residue = [str(res_ids[i]) for i in range(len(scores))]

    df = pd.DataFrame({"residue": residue, "node": np.arange(len(scores)), "score": scores})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # quick stats to stdout
    print(json.dumps({
        "n": len(scores),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "out": str(args.out)
    }, indent=2))

if __name__ == "__main__":
    main()
