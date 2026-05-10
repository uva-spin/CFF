#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate_results.py

Aggregate all per-point .npz files from jacobian_worker.py and produce:

  1. A summary CSV
  2. Heatmaps of effective rank  (single global colorbar, no overlap)
  3. Heatmaps of condition number (global log scale capped at KAPPA_MAX,
     points with κ > KAPPA_MAX marked with ×)
  4. Phase diagram colored by minimal-subset size; subset identities
     conveyed via a companion LaTeX table
  5. A .tex file with the minimal-subset summary table

Run after all Slurm tasks complete:

    python aggregate_results.py --indir results/ --outdir analysis/
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ── constants ────────────────────────────────────────────────────────────────

N_CFFS  = 8
ALL_OBS = ["XS", "BSA", "BCA", "TSA", "DSA"]

# Condition-number cap: points with κ > KAPPA_MAX are marked × and excluded
# from the color scale
KAPPA_MAX = 1e4

# Font sizes
# "tick" controls numerical axis/colorbar tick labels
# all other entries control word labels
FS = {
    "title":     19,   # subplot titles (observable set label)
    "label":     20,   # axis labels and colorbar label
    "tick":      13,   # numerical tick labels on axes and colorbar — unchanged
    "legend":    18,   # legend text
    "leg_title": 19,   # legend title
}

# Subsets shown in per-subset heatmaps
HIGHLIGHT_SUBSETS: List[str] = [
    "XS",
    "BSA",
    "XS,BSA",
    "XS,BSA,BCA",
    "XS,BSA,BCA,TSA",
    "XS,BSA,BCA,TSA,DSA",
]

# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def _latex_obs(name: str) -> str:
    return r"\mathrm{" + name + "}"


def subset_to_latex(subset_str: str) -> str:
    """
    Fixes the notation in the plot titles
    """
    parts  = [s.strip() for s in subset_str.split(",")]
    inner  = r",\,".join(_latex_obs(p) for p in parts)
    return rf"$\{{{inner}\}}$"


# ── CLI / I-O ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--indir",    default="results",  help="Directory with point_*.npz")
    p.add_argument("--outdir",   default="analysis", help="Output directory")
    p.add_argument("--rank-tol", type=float, default=1e-3)
    return p.parse_args()


def load_all(indir: str) -> list:
    files = sorted(glob.glob(os.path.join(indir, "point_*.npz")))
    if not files:
        raise FileNotFoundError(f"No point_*.npz files in {indir}")
    records = []
    for fpath in files:
        d = np.load(fpath, allow_pickle=True)
        kin    = d["kin"]
        records.append({
            "t":       float(kin[0]),
            "xB":      float(kin[1]),
            "Q2":      float(kin[2]),
            "subsets": list(d["subsets"]),
            "rank":    d["rank"].tolist(),
            "cond":    d["cond"].tolist(),
            "sv_all":  d["sv_all"].tolist(),
        })
    print(f"Loaded {len(records)} kinematic points from {indir}")
    return records


def write_csv(records: list, outpath: str) -> None:
    rows = []
    for rec in records:
        for i, sub in enumerate(rec["subsets"]):
            r = rec["rank"][i]
            rows.append({
                "t":         rec["t"],
                "xB":        rec["xB"],
                "Q2":        rec["Q2"],
                "subset":    sub,
                "n_obs":     sub.count(",") + 1,
                "rank":      r,
                "cond":      rec["cond"][i],
                "sigma_min": rec["sv_all"][i][r - 1] if r > 0 else 0.0,
            })
    if not rows:
        return
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote CSV: {outpath}  ({len(rows)} rows)")


def pivot(records: list, subset_str: str, field: str):
    """Return (xB, t, values) arrays for one subset and one field."""
    xs, ts, vs = [], [], []
    for rec in records:
        if subset_str in rec["subsets"]:
            i = rec["subsets"].index(subset_str)
            xs.append(rec["xB"])
            ts.append(rec["t"])
            vs.append(rec[field][i])
    return np.array(xs), np.array(ts), np.array(vs)


# ── colorbar tick helpers ─────────────────────────────────────────────────────

def _log_ticks(vmin: float, vmax: float, min_ticks: int = 2) -> list:
    """
    Return tick positions for a log-scale spanning [vmin, vmax].
    Guarantees at least `min_ticks` ticks by adding half-decade marks if needed.
    """
    lo = int(np.floor(np.log10(vmin)))
    hi = int(np.ceil (np.log10(vmax)))
    decade = [10.0**k for k in range(lo, hi + 1) if vmin <= 10.0**k <= vmax]
    if len(decade) < min_ticks:
        half = [10.0**k * 10.0**0.5 for k in range(lo, hi + 1)
                if vmin <= 10.0**k * 10.0**0.5 <= vmax]
        decade = sorted(set(decade + half))
    return decade


def _tick_label(v: float) -> str:
    lv = np.log10(v)
    if abs(lv - round(lv)) < 0.01:
        return rf"$10^{{{int(round(lv))}}}$"
    m   = v / 10.0**np.floor(lv)
    exp = int(np.floor(lv))
    return rf"${m:.2g} \times 10^{{{exp}}}$"


# ── shared colorbar factory ───────────────────────────────────────────────────

def _add_colorbar(fig, mappable, label: str, ticks=None, tick_labels=None,
                  cbar_rect=(0.88, 0.12, 0.025, 0.76)):
    """
    Place a colorbar in a manually specified axes rectangle so it never
    overlaps the subplot grid.  Call fig.subplots_adjust(right=0.86) before
    this to leave room.
    """
    cax = fig.add_axes(cbar_rect)
    cb  = fig.colorbar(mappable, cax=cax, label=label)
    cb.set_label(label, fontsize=FS["label"])
    cb.ax.tick_params(labelsize=FS["tick"])
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels(tick_labels or [_tick_label(t) for t in ticks],
                          fontsize=FS["tick"])
    return cb


# ── scatter helper ────────────────────────────────────────────────────────────

def _scatter(ax, t_arr, xb_arr, vals, **kwargs):
    return ax.scatter(-t_arr, xb_arr, c=vals, s=120,
                      edgecolors="k", linewidths=0.5, **kwargs)


# ── 1. RANK HEATMAPS  (single global colorbar) ───────────────────────────────

def plot_rank_heatmaps(records: list, outdir: str,
                       subsets_to_plot: List[str]) -> None:
    n     = len(subsets_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=N_CFFS)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols + 1.5, 4.5 * nrows),
                             squeeze=False)
    fig.subplots_adjust(right=0.86, hspace=0.4, wspace=0.35)
    axes_flat = axes.flatten()

    sc_ref = None
    for i, sub in enumerate(subsets_to_plot):
        ax = axes_flat[i]
        xb, t, ranks = pivot(records, sub, "rank")
        if len(xb) == 0:
            ax.set_visible(False)
            continue
        sc = _scatter(ax, t, xb, ranks.astype(float), cmap=cmap, norm=norm)
        sc_ref = sc
        ax.set_xlabel(r"$-t\ (\mathrm{GeV}^2)$", fontsize=FS["label"])
        ax.set_ylabel(r"$x_B$", fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_title(subset_to_latex(sub), fontsize=FS["title"])

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if sc_ref is not None:
        int_ticks = list(range(0, N_CFFS + 1))
        _add_colorbar(fig, sc_ref, label="Effective Rank",
                      ticks=int_ticks,
                      tick_labels=[str(k) for k in int_ticks])

    out = os.path.join(outdir, "rank_heatmaps.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 2. CONDITION NUMBER HEATMAPS  (global log scale, capped at KAPPA_MAX) ────

def plot_cond_heatmaps(records: list, outdir: str,
                       subsets_to_plot: List[str]) -> None:
    # First pass: collect finite conds below cap for global scale
    cache: dict = {}
    all_finite: list = []
    for sub in subsets_to_plot:
        xb, t, conds = pivot(records, sub, "cond")
        cache[sub] = (xb, t, conds)
        valid = conds[np.isfinite(conds) & (conds > 0) & (conds <= KAPPA_MAX)]
        all_finite.extend(valid.tolist())

    vmin = float(np.min(all_finite)) if all_finite else 1.0
    vmax = KAPPA_MAX
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma_r
    ticks      = _log_ticks(vmin, vmax, min_ticks=2)
    tick_lbls  = [_tick_label(v) for v in ticks]

    n     = len(subsets_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols + 1.5, 4.5 * nrows),
                             squeeze=False)
    fig.subplots_adjust(right=0.86, hspace=0.4, wspace=0.35)
    axes_flat = axes.flatten()

    sc_ref = None
    for i, sub in enumerate(subsets_to_plot):
        ax = axes_flat[i]
        xb, t, conds = cache[sub]
        if len(xb) == 0:
            ax.set_visible(False)
            continue

        # Points with κ ≤ KAPPA_MAX: colored
        colored_mask = np.isfinite(conds) & (conds > 0) & (conds <= KAPPA_MAX)
        # Points with κ > KAPPA_MAX or non-finite: marked ×
        outlier_mask = ~colored_mask

        if np.any(colored_mask):
            sc = _scatter(ax, t[colored_mask], xb[colored_mask],
                          conds[colored_mask], cmap=cmap, norm=norm)
            sc_ref = sc

        if np.any(outlier_mask):
            ax.scatter(-t[outlier_mask], xb[outlier_mask],
                       c="lightgrey", marker="x", s=90, linewidths=1.4,
                       label=rf"$\kappa > 10^{{{int(np.log10(KAPPA_MAX))}}}$",
                       zorder=4)
            ax.legend(fontsize=FS["legend"], loc="upper left")

        ax.set_xlabel(r"$-t\ (\mathrm{GeV}^2)$", fontsize=FS["label"])
        ax.set_ylabel(r"$x_B$", fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_title(subset_to_latex(sub), fontsize=FS["title"])

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if sc_ref is not None:
        _add_colorbar(fig, sc_ref,
                      label=r"Condition Number $\kappa$",
                      ticks=ticks, tick_labels=tick_lbls)

    out = os.path.join(outdir, "cond_heatmaps.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 3. PHASE DIAGRAM  (color by min-size; subsets conveyed via LaTeX table) ──

def _compute_phase(records: list):
    """
    For each kinematic point find:
      - min_size : size of smallest full-rank subset (-1 if none)
      - min_subs : set of all subsets tied for that minimum size
    Returns list of dicts with keys t, xB, Q2, min_size, min_subs.
    Also returns summary Counter: (min_size, frozenset_of_tied_subs) -> count.
    """
    results = []
    summary = collections.Counter()   # (min_size, subset_str) -> # points it appears at

    for rec in records:
        full_idx = [i for i, r in enumerate(rec["rank"]) if r >= N_CFFS]
        if not full_idx:
            results.append({"t": rec["t"], "xB": rec["xB"], "Q2": rec["Q2"],
                            "min_size": -1, "min_subs": set()})
            summary[(-1, "none")] += 1
            continue

        sizes    = [rec["subsets"][i].count(",") + 1 for i in full_idx]
        min_sz   = min(sizes)
        min_subs = {rec["subsets"][i] for i, sz in zip(full_idx, sizes) if sz == min_sz}

        results.append({"t": rec["t"], "xB": rec["xB"], "Q2": rec["Q2"],
                        "min_size": min_sz, "min_subs": min_subs})
        for sub in min_subs:
            summary[(min_sz, sub)] += 1

    return results, summary


def plot_phase_diagram(records: list, outdir: str) -> None:
    phase, _ = _compute_phase(records)

    xbs  = np.array([p["xB"]      for p in phase])
    ts   = np.array([p["t"]       for p in phase])
    szs  = np.array([p["min_size"] for p in phase], dtype=int)

    unique_sizes = sorted(set(szs.tolist()))

    # Color by min_size only: grey for "none", then tab10 palette for 2, 3, 4, …
    size_palette = {-1: "lightgrey"}
    tab_colors   = plt.cm.tab10.colors
    for k, sz in enumerate(s for s in unique_sizes if s >= 0):
        size_palette[sz] = tab_colors[k]

    fig, ax = plt.subplots(figsize=(8, 6))

    handles = []
    for sz in unique_sizes:
        mask  = szs == sz
        color = size_palette[sz]
        ax.scatter(-ts[mask], xbs[mask], c=[color], s=160,
                   edgecolors="k", linewidths=0.6, zorder=3)
        label = "No full-rank subset" if sz < 0 else f"Minimal size $= {sz}$"
        handles.append(mpatches.Patch(facecolor=color, edgecolor="k", label=label))

    ax.legend(handles=handles, title="Min. subset size",
              title_fontsize=FS["leg_title"], fontsize=FS["legend"],
              loc="upper left", framealpha=0.92)
    ax.set_xlabel(r"$-t\ (\mathrm{GeV}^2)$", fontsize=FS["label"])
    ax.set_ylabel(r"$x_B$", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    plt.tight_layout()
    out = os.path.join(outdir, "phase_diagram_min_subset_size.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved: {out}")


# ── 4. LaTeX SUMMARY TABLE ───────────────────────────────────────────────────

def write_latex_table(records: list, outpath: str) -> None:
    """
    Produce a .tex file containing a longtable summarising, for each minimal-
    full-rank observable subset, the number of kinematic points at which it
    appears as a minimal subset.

    Columns: Min. size | Observable subset | # kinematic points
    """
    _, summary = _compute_phase(records)

    # Group by size, sort subsets within each size by descending count
    by_size: dict = collections.defaultdict(list)
    for (sz, sub), cnt in summary.items():
        by_size[sz].append((sub, cnt))
    for sz in by_size:
        by_size[sz].sort(key=lambda x: -x[1])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{%")
    lines.append(r"  For each minimal observable subset achieving full Jacobian rank")
    lines.append(rf"  ($= {N_CFFS}$), the number of kinematic points in the")
    lines.append(r"  $(x_B, -t)$ grid at which that subset appears as a minimal")
    lines.append(r"  full-rank subset.  Multiple subsets of the same minimal size")
    lines.append(r"  may tie at a single kinematic point, so column counts can")
    lines.append(r"  sum to more than the total number of grid points.")
    lines.append(r"}")
    lines.append(r"\label{tab:minimal_subsets}")
    lines.append(r"\begin{tabular}{cll}")
    lines.append(r"\hline\hline")
    lines.append(r"Min.\ size & Observable subset & \# kinematic points \\")
    lines.append(r"\hline")

    prev_sz = None
    for sz in sorted(by_size.keys()):
        entries = by_size[sz]
        for k, (sub, cnt) in enumerate(entries):
            if sz == -1:
                sz_cell = r"\multicolumn{1}{c}{---}"
                sub_cell = "---"
            else:
                sz_cell  = str(sz) if k == 0 else ""
                # Format subset as LaTeX math set
                parts    = [s.strip() for s in sub.split(",")]
                inner    = r",\,".join(r"\mathrm{" + p + "}" for p in parts)
                sub_cell = rf"$\{{{inner}\}}$"
            lines.append(rf"  {sz_cell} & {sub_cell} & {cnt} \\")
        if prev_sz is not None and sz != prev_sz:
            lines.insert(-len(entries), r"\hline")
        prev_sz = sz

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    text = "\n".join(lines) + "\n"
    with open(outpath, "w") as f:
        f.write(text)
    print(f"Wrote LaTeX table: {outpath}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    records = load_all(args.indir)

    write_csv(records, os.path.join(args.outdir, "jacobian_sweep_summary.csv"))

    all_sub_strs   = records[0]["subsets"] if records else []
    subsets_to_plot = [s for s in HIGHLIGHT_SUBSETS if s in all_sub_strs]

    plot_rank_heatmaps(records, args.outdir, subsets_to_plot)
    plot_cond_heatmaps(records, args.outdir, subsets_to_plot)
    plot_phase_diagram(records, args.outdir)
    write_latex_table(records, os.path.join(args.outdir, "minimal_subsets_table.tex"))

    print(f"\nAll outputs written to: {args.outdir}/")


if __name__ == "__main__":
    main()
