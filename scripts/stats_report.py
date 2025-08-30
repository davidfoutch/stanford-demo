import argparse, os, sys, re, numpy as np, pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, spearmanr
try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    print("Missing statsmodels. Install with:\n  ./.venv/bin/pip install statsmodels")
    sys.exit(1)

LOW  = "1YUC 3TX7 4DOR 1YOK 1ZDU 4DOS 4ONI 4PLE 4RWV".split()
HIGH = "3PLZ 5UNJ 5L11 6OR1 6VC2 5SYZ 6OQX 4PLD 6OQY".split()

def infer_col(df, want="res"):
    cols = list(df.columns)
    if want == "res":
        cands = [c for c in cols if any(k in c.lower() for k in ("res_index","resi","resid","residue","res"))]
        # prefer exact 'res_index'
        cands.sort(key=lambda c: (0 if c.lower()=="res_index" else 1, len(c)))
    else:
        cands = [c for c in cols if any(k in c.lower() for k in ("ig","score","mean","importance","attr")) and "delta" not in c.lower()]
    if not cands:
        raise ValueError(f"Could not infer {'residue' if want=='res' else 'score'} column from {cols}")
    return cands[0]

_num_re = re.compile(r"-?\d+")
def _to_res_int(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, np.integer)): return int(x)
    m = _num_re.search(str(x))
    return int(m.group()) if m else np.nan

def read_ig_csv(path):
    df = pd.read_csv(path)
    r = infer_col(df,"res"); s = infer_col(df,"score")
    out = df[[r,s]].rename(columns={r:"res_index", s:"ig"}).copy()
    out["res_index"] = out["res_index"].apply(_to_res_int)
    out = out.dropna(subset=["res_index"])
    out["res_index"] = out["res_index"].astype(int)
    out["ig"] = pd.to_numeric(out["ig"], errors="coerce").fillna(0.0)
    return out

def per_structure_path(pdb, cutoff):
    return f"artifacts/importance/{pdb}_res_scores_c7.0.csv" if cutoff == "7.0" else f"artifacts/importance/{pdb}_res_scores.csv"

def load_group_structures(pdbs, cutoff):
    items = []
    for p in pdbs:
        path = per_structure_path(p, cutoff)
        if os.path.exists(path):
            df = read_ig_csv(path)
            df["pdb"] = p
            items.append(df)
    if not items:
        raise SystemExit(f"No IG CSVs found for cutoff {cutoff} in artifacts/importance/")
    return pd.concat(items, ignore_index=True)

def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0: return np.nan
    a_sorted = np.sort(a); b_sorted = np.sort(b)
    i = j = gt = lt = 0
    while i < na and j < nb:
        if a_sorted[i] > b_sorted[j]:
            gt += (na - i); j += 1
        elif a_sorted[i] < b_sorted[j]:
            lt += (nb - j); i += 1
        else:
            ai = i; bj = j
            while i < na and a_sorted[i] == a_sorted[ai]: i += 1
            while j < nb and b_sorted[j] == b_sorted[bj]: j += 1
    return (gt - lt) / (na * nb)

def load_roi_for_pdb(pdb):
    path = f"artifacts/roi/{pdb}_A_lig6.csv"
    if not os.path.exists(path): return set()
    df = pd.read_csv(path)
    r = infer_col(df,"res")
    vals = pd.to_numeric(df[r], errors="coerce")
    if vals.isna().all():
        vals = df[r].apply(_to_res_int)
    return set(pd.to_numeric(vals, errors="coerce").dropna().astype(int).tolist())

def roi_fraction_for_structure(struct_df, roi_set, mode="pos"):
    ig = struct_df["ig"].values
    if mode == "pos": ig = np.clip(ig, 0, None)
    elif mode == "abs": ig = np.abs(ig)
    total = ig.sum()
    if total <= 0: return np.nan
    in_roi = struct_df["res_index"].isin(roi_set).values
    return float(ig[in_roi].sum() / total)

def group_means(all_df):
    gH = (all_df.loc[all_df.group=="HIGH", ["res_index","ig"]]
           .groupby("res_index", as_index=False)["ig"].mean()
           .rename(columns={"ig":"mean_ig"}))
    gL = (all_df.loc[all_df.group=="LOW", ["res_index","ig"]]
           .groupby("res_index", as_index=False)["ig"].mean()
           .rename(columns={"ig":"mean_ig"}))
    d  = pd.merge(gH, gL, on="res_index", how="outer", suffixes=("_H","_L")).fillna(0.0)
    d["delta"] = d["mean_ig_H"] - d["mean_ig_L"]
    return gH, gL, d[["res_index","delta"]]

def one_sided_p_from_t(t, p_two_sided):
    return p_two_sided/2 if t > 0 else 1 - (p_two_sided/2)

def per_residue_bhfdr(per_struct_df, alpha=0.10):
    rows = []
    for r, sub in per_struct_df.groupby("res_index"):
        vH = sub.loc[sub.group=="HIGH","ig"].values
        vL = sub.loc[sub.group=="LOW","ig"].values
        if len(vH) == 0 or len(vL) == 0: 
            continue
        t, p2 = ttest_ind(vH, vL, equal_var=False, nan_policy="omit")
        p = one_sided_p_from_t(t, p2)   # HIGH > LOW
        delta = float(np.nanmean(vH) - np.nanmean(vL))
        rows.append((r, delta, t, p, len(vH), len(vL)))
    if not rows:
        return pd.DataFrame(columns=["res_index","delta","t","p","q","sig_pos","nH","nL"])
    res = pd.DataFrame(rows, columns=["res_index","delta","t","p","nH","nL"]).sort_values("p")
    rej, q, _, _ = multipletests(res["p"].values, method="fdr_bh", alpha=alpha)
    res["q"] = q
    res["sig_pos"] = (rej) & (res["delta"] > 0)
    return res.sort_values(["q","p","delta"], ascending=[True, True, False]).reset_index(drop=True)

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", nargs="+", default=["7.0","4.5"], choices=["7.0","4.5"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--ks", nargs="+", type=int, default=[10,20,30])
    ap.add_argument("--roi_mode", choices=["pos","abs","raw"], default="pos")
    ap.add_argument("--write_overlays", action="store_true")
    args = ap.parse_args()

    roi_1YOK = load_roi_for_pdb("1YOK")
    d_means = {}

    for cutoff in args.cutoffs:
        outdir = "artifacts/importance/group_c7.0" if cutoff=="7.0" else "artifacts/importance/group"
        os.makedirs(outdir, exist_ok=True)

        H_df = load_group_structures(HIGH, cutoff); H_df["group"] = "HIGH"
        L_df = load_group_structures(LOW,  cutoff); L_df["group"] = "LOW"
        all_df = pd.concat([H_df, L_df], ignore_index=True)

        # Group means + delta (writes)
        gH, gL, d = group_means(all_df)
        gH.to_csv(os.path.join(outdir, f"group_mean_HIGH_{cutoff.replace('.','p')}.csv"), index=False)
        gL.to_csv(os.path.join(outdir, f"group_mean_LOW_{cutoff.replace('.','p')}.csv"),  index=False)
        d.to_csv(  os.path.join(outdir, f"delta_{cutoff.replace('.','p')}.csv"),          index=False)
        d_means[cutoff] = d

        # Top-k summary (+ optional overlays)
        print(f"\n== {cutoff}Å | TOP-K (HIGH vs LOW) ==")
        for k in args.ks:
            topH = gH.sort_values("mean_ig", ascending=False).head(k)["res_index"].tolist()
            topL = gL.sort_values("mean_ig", ascending=False).head(k)["res_index"].tolist()
            shared = sorted(set(topH) & set(topL))
            H_only = sorted(set(topH) - set(shared))
            L_only = sorted(set(topL) - set(shared))
            j = jaccard(topH, topL)
            def roi_count(S): 
                return sum(1 for r in S if r in roi_1YOK) if roi_1YOK else 0
            print(f" k={k:>2} | J={j:.2f} | shared={len(shared)} (ROI {roi_count(shared)}) | "
                  f"H-only={len(H_only)} (ROI {roi_count(H_only)}) | "
                  f"L-only={len(L_only)} (ROI {roi_count(L_only)})")
            if args.write_overlays:
                def write(name, rows):
                    pd.DataFrame(rows, columns=["res_index","score"]).to_csv(os.path.join(outdir, name), index=False)
                gHd = gH.set_index("res_index")["mean_ig"].to_dict()
                gLd = gL.set_index("res_index")["mean_ig"].to_dict()
                write(f"topk{k}_SHARED.csv",    [(r, (gHd.get(r,0)+gLd.get(r,0))/2) for r in shared])
                write(f"topk{k}_HIGH_only.csv", [(r, gHd.get(r,0)) for r in H_only])
                write(f"topk{k}_LOW_only.csv",  [(r, gLd.get(r,0)) for r in L_only])

        # ROI fractions per structure (PDB-specific ROI)
        roi_rows = []
        for p in sorted(set(all_df["pdb"])):
            roi_set = load_roi_for_pdb(p)
            sub = all_df[all_df["pdb"]==p]
            frac = roi_fraction_for_structure(sub, roi_set, mode=args.roi_mode) if roi_set else np.nan
            grp  = "HIGH" if p in HIGH else "LOW"
            roi_rows.append((p, grp, frac))
        roi_df = pd.DataFrame(roi_rows, columns=["pdb","group","roi_frac"]).sort_values(["group","pdb"])
        roi_df.to_csv(os.path.join(outdir, "roi_fractions.csv"), index=False)

        # MWU + Cliff's δ
        rH = roi_df.loc[roi_df.group=="HIGH","roi_frac"].dropna().values
        rL = roi_df.loc[roi_df.group=="LOW","roi_frac"].dropna().values
        print(f"\n== {cutoff}Å | ROI FRACTION (mode={args.roi_mode}) ==")
        if len(rH) and len(rL):
            U_low_gt_high, p_low_gt_high = mannwhitneyu(rL, rH, alternative="greater")
            U_high_gt_low, p_high_gt_low = mannwhitneyu(rH, rL, alternative="greater")
            delta = cliffs_delta(rL, rH)  # positive means LOW>HIGH
            print(f" medians LOW={np.nanmedian(rL):.3f} | HIGH={np.nanmedian(rH):.3f}")
            print(f" MWU one-sided: LOW>HIGH p={p_low_gt_high:.3g} | HIGH>LOW p={p_high_gt_low:.3g}")
            print(f" Cliff's δ (LOW vs HIGH) = {delta:+.2f}")
        else:
            print(" (Insufficient ROI data.)")

        # Per-residue BH–FDR
        bh = per_residue_bhfdr(all_df, alpha=args.alpha)
        bh.to_csv(os.path.join(outdir, f"bhfdr_{cutoff.replace('.','p')}.csv"), index=False)
        n_sig = int(bh["sig_pos"].sum())
        print(f"\n== {cutoff}Å | BH–FDR (alpha={args.alpha}) ==")
        print(f" significant positives (HIGH>LOW): {n_sig}")
        if n_sig:
            print(bh.loc[bh.sig_pos, ["res_index","delta","q"]].head(10).to_string(index=False))

    # Cross-cutoff Spearman on Δ
    if "4.5" in d_means and "7.0" in d_means:
        d4 = d_means["4.5"]; d7 = d_means["7.0"]
        merged = pd.merge(d4, d7, on="res_index", how="inner", suffixes=("_4p5","_7p0"))
        if not merged.empty:
            rho, p = spearmanr(merged["delta_4p5"], merged["delta_7p0"])
            print(f"\n== Δ STABILITY ACROSS CUTOFFS ==")
            print(f" Spearman(Δ4.5, Δ7.0) = {rho:.2f} (p={p:.3g})")
        else:
            print("\n== Δ STABILITY ACROSS CUTOFFS ==\n (No overlapping residues found.)")

if __name__ == "__main__":
    main()
