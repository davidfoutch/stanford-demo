import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# ---------- utils ----------
def set_seed(s: int = 42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_graph(path: str) -> Data:
    # PyTorch 2.7: turn off weights-only for our trusted local artifacts
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "edge_index" in obj:
        # Support dict-of-tensors format too
        return Data(**{k: v for k, v in obj.items() if v is not None})
    return obj  # already a Data

def _label_to_int(y):
    if isinstance(y, str):
        y = y.strip().upper()
        return 0 if y in ("A", "0") else 1
    return int(y)

def load_dataset(graph_dir: Path):
    idx_path = graph_dir / "index.csv"
    paths, labels = [], []
    if idx_path.is_file():
        idx = pd.read_csv(idx_path)
        path_col = "path" if "path" in idx.columns else ("pt_path" if "pt_path" in idx.columns else None)
        if path_col is None:
            # fallback: infer filenames from pdb_id/chain if present
            for _, r in idx.iterrows():
                guess = graph_dir / f"{r['pdb_id']}_{r.get('chain','A')}_c4.5.pt"
                paths.append(str(guess)); labels.append(_label_to_int(r["label"]))
        else:
            root = graph_dir.resolve()
            for _, r in idx.iterrows():
                raw = str(r[path_col]).strip()
                p = Path(raw)

                if p.is_absolute():
                    final = p
                else:
                    # If the stored path already starts with the graphs dir name or "data/", don't prefix.
                    first = p.parts[0] if p.parts else ""
                    if first in {root.name, "data", "."}:
                        final = p
                    else:
                        final = root / p

                paths.append(str(final))
                labels.append(_label_to_int(r["label"]))
    else:
        # final fallback: load all .pt files in dir; labels must be encoded in filename
        for p in sorted(graph_dir.glob("*.pt")):
            paths.append(str(p))
            # naive parse: *_A_* → 0, *_B_* → 1
            lbl = 0 if "_A_" in p.name or p.name.endswith("_A.pt") else 1
            labels.append(lbl)

    data = []
    for p, y in zip(paths, labels):
        d: Data = load_graph(p)
        d.y = torch.tensor([y], dtype=torch.long)
        # ensure tensors are the right dtype
        d.x = d.x.float()
        if getattr(d, "edge_attr", None) is not None:
            d.edge_attr = d.edge_attr.float()
        data.append(d)
    return data

def stratified_split(labels, train=0.70, val=0.15, seed=42):
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    rng = np.random.default_rng(seed)
    tr_idx, va_idx, te_idx = [], [], []
    for c in np.unique(labels):
        c_idx = idx[labels == c]
        rng.shuffle(c_idx)
        n_tr = int(len(c_idx) * train)
        n_va = int(len(c_idx) * val)
        tr_idx.extend(c_idx[:n_tr])
        va_idx.extend(c_idx[n_tr:n_tr+n_va])
        te_idx.extend(c_idx[n_tr+n_va:])
    return np.array(tr_idx), np.array(va_idx), np.array(te_idx)

# ---------- model ----------
class GATGraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, num_classes=2, edge_dim=None, dropout=0.1):
        super().__init__()
        # Build convs with/without edge_attr support
        if edge_dim is None:
            self.gat1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
            self.gat2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout)
        else:
            self.gat1 = GATConv(in_dim, hidden, heads=heads, edge_dim=edge_dim, dropout=dropout)
            self.gat2 = GATConv(hidden * heads, hidden, heads=1, edge_dim=edge_dim, dropout=dropout)
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

# ---------- training ----------
def run_epoch(model, loader, crit, opt=None, device="cpu"):
    train = opt is not None
    model.train() if train else model.eval()
    losses, ys, ps = [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(train):
            logits = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None), batch.batch)
            loss = crit(logits, batch.y.view(-1))
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        ys.append(batch.y.view(-1).cpu().numpy())
        ps.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    if not ys:
        return 0.0, float("nan"), float("nan")
    y = np.concatenate(ys); p = np.concatenate(ps)
    try: auc = roc_auc_score(y, p)
    except Exception: auc = float("nan")
    f1 = f1_score(y, (p >= 0.5).astype(int))
    return float(np.mean(losses)), float(auc), float(f1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True, help="Directory containing graphs and index.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    # new (optional) knobs
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4, dest="wd")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    gdir = Path(args.graphs)
    ds = load_dataset(gdir)
    assert len(ds) >= 2, "Need at least 2 graphs"

    labels = [int(d.y.item()) for d in ds]
    tr_idx, va_idx, te_idx = stratified_split(labels, train=0.70, val=0.15, seed=args.seed)
    train_ds = [ds[i] for i in tr_idx]; val_ds = [ds[i] for i in va_idx]; test_ds = [ds[i] for i in te_idx]

    in_dim = ds[0].x.size(1)
    edge_dim = ds[0].edge_attr.size(1) if getattr(ds[0], "edge_attr", None) is not None else None

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)
    test_loader  = DataLoader(test_ds, batch_size=args.batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATGraphClassifier(in_dim, hidden=args.hidden, heads=args.heads,
                               num_classes=2, edge_dim=edge_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()

    best = {"epoch": 0, "va_auc": -1.0}
    hist = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_auc, tr_f1 = run_epoch(model, train_loader, crit, opt=opt, device=device)
        va_loss, va_auc, va_f1 = run_epoch(model, val_loader, crit, opt=None, device=device)
        rec = {"epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss, "va_auc": va_auc, "va_f1": va_f1}
        hist.append(rec)
        print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.3f} va_loss {va_loss:.3f} va_auc {va_auc:.3f} va_f1 {va_f1:.3f}")
        if np.isfinite(va_auc) and va_auc > best["va_auc"]:
            best = {**rec}
            (Path(args.out).mkdir(parents=True, exist_ok=True))
            torch.save(model.state_dict(), Path(args.out) / "best_model.pt")

    # final test eval
    te_loss, te_auc, te_f1 = run_epoch(model, test_loader, crit, opt=None, device=device)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "last_model.pt")
    summary = {**best, "test_loss": te_loss, "test_auc": te_auc, "test_f1": te_f1,
               "in_dim": int(in_dim), "edge_dim": (int(edge_dim) if edge_dim is not None else None),
               "hidden": args.hidden, "heads": args.heads, "batch": args.batch, "lr": args.lr,
               "dropout": args.dropout, "weight_decay": args.wd, "seed": args.seed}
    with open(outdir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] best_model.pt / last_model.pt -> {outdir}")
    print(f"[metrics] {summary}")

if __name__ == "__main__":
    main()
