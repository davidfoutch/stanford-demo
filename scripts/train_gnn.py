#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

class GATGraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, num_classes=2, edge_dim=2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, edge_dim=edge_dim, dropout=0.1)
        self.gat2 = GATConv(hidden*heads, hidden, heads=1, edge_dim=edge_dim, dropout=0.1)
        self.lin  = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr).relu()
        x = self.gat2(x, edge_index, edge_attr).relu()
        g = global_mean_pool(x, batch)
        return self.lin(g)

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_dataset(graph_dir:Path):
    idx = pd.read_csv(graph_dir / "index.csv")
    data = []
    for _, row in idx.iterrows():
        d: Data = torch.load(row["path"])
        y = row["label"]
        if isinstance(y, str): y = 0 if y.upper() in ("A","0") else 1
        d.y = torch.tensor([y], dtype=torch.long)
        data.append(d)
    return data

def split_idx(N, train=0.7, val=0.15, seed=42):
    idx = np.arange(N); rng = np.random.default_rng(seed); rng.shuffle(idx)
    n_tr = int(N*train); n_val = int(N*val)
    return idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    set_seed(42)

    gdir = Path(args.graphs)
    ds = load_dataset(gdir)
    assert len(ds)>1, "Need at least 2 graphs"
    in_dim = ds[0].x.size(1)
    edge_dim = ds[0].edge_attr.size(1)

    tr, va, te = split_idx(len(ds))
    train_ds = [ds[i] for i in tr]; val_ds = [ds[i] for i in va]; test_ds = [ds[i] for i in te]

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4)
    test_loader  = DataLoader(test_ds, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATGraphClassifier(in_dim, hidden=64, heads=4, num_classes=2, edge_dim=edge_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()

    def run(loader, train=False):
        if train: model.train()
        else: model.eval()
        losses, ys, ps = [], [], []
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = crit(logits, batch.y.view(-1))
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            ys.append(batch.y.view(-1).cpu().numpy())
            ps.append(torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy())
        if len(ys)==0:
            return 0.0, 0.0, 0.0
        y = np.concatenate(ys); p = np.concatenate(ps)
        try: auc = roc_auc_score(y, p)
        except Exception: auc = float('nan')
        f1 = f1_score(y, (p>=0.5).astype(int))
        return np.mean(losses), auc, f1

    hist=[]
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_auc, tr_f1 = run(train_loader, train=True)
        va_loss, va_auc, va_f1 = run(val_loader, train=False)
        hist.append({"epoch":epoch,"tr_loss":tr_loss,"va_loss":va_loss,"va_auc":va_auc,"va_f1":va_f1})
        print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.3f} va_loss {va_loss:.3f} va_auc {va_auc:.3f} va_f1 {va_f1:.3f}")

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "model.pt")
    with open(outdir / "metrics.json","w") as f: json.dump(hist[-1], f, indent=2)
    print(f"Saved model to {outdir/'model.pt'}; metrics to {outdir/'metrics.json'}")

if __name__ == "__main__":
    main()
