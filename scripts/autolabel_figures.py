"""Paper figures + LaTeX tables for the USS auto-labeler paper (ITSC 2027).

Reads the outputs of scripts/autolabel_analysis.py (<run_dir>/analysis/) and
writes vector PDFs + .tex tables to <out_dir>/figures and <out_dir>/tables.

    PYTHONPATH=src python scripts/autolabel_figures.py \\
        outputs/autolabel/al_eval_v2 paper/itsc2027

Palette: reference categorical slots 1-3 (validated all-pairs, light), text in
neutral ink. High = slot 1 blue, Low = slot 2 orange, v2 box = slot 3 aqua
(below 3:1 contrast -> always paired with a direct label / distinct dash).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vision_uss_research.autolabel.geometry import oriented_box  # noqa: E402
from vision_uss_research.autolabel.priors import fit_priors, fold_of, prior_box  # noqa: E402
from vision_uss_research.autolabel.uss_box import (V1_BOX_LENGTH_M,  # noqa: E402
                                                   V1_BOX_WIDTH_M, polygon_from_str)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
HB_COLOR = {"high": BLUE, "low": ORANGE}
COL_W, FULL_W = 3.5, 7.16          # IEEE two-column widths (in)
K_FOLDS = 5

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

LADDER = [("v1", "Prototype (v1)"), ("lat_pos", "+ lateral offset"),
          ("cls", "+ class box size"), ("cls_off", "+ class placement offset"),
          ("cls_appr", "+ class $\\times$ approach priors")]
OBJ_LABEL = {"pole": "pole", "dummychild": "child dummy", "car": "car", "bush": "bush",
             "standardcube": "cube", "cone": "cone", "bicyclestand": "bicycle stand",
             "tree": "tree", "bollard": "bollard", "curbstone": "curbstone",
             "woodenboard": "wooden board", "speedbump": "speed bump",
             "multi_obj_curbstone_bush": "curb + bush"}


def _ci(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


# ------------------------------------------------------------------ fig: ladder

def fig_ladder(lad: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ys = np.arange(len(LADDER))[::-1]
    for (v, _), y in zip(LADDER, ys):
        for hb, dy in (("high", 0.13), ("low", -0.13)):
            r = lad[(lad["variant"] == v) & (lad["subset"] == hb)].iloc[0]
            lo, hi = _ci(r["iou_med_ci"])
            ax.plot([lo, hi], [y + dy] * 2, color=HB_COLOR[hb], lw=1.2, solid_capstyle="round")
            ax.plot(r["iou_med"], y + dy, "o" if hb == "high" else "s", ms=4.2,
                    color=HB_COLOR[hb], mec="white", mew=0.6, zorder=3)
        r = lad[(lad["variant"] == v) & (lad["subset"] == "all")].iloc[0]
        ax.text(0.665, y, f"{r['iou_med']:.3f}", va="center", ha="right", color=INK, fontsize=7)
    ax.text(0.665, len(LADDER) - 0.35, "all", ha="right", color=INK2, fontsize=7)
    ax.set_yticks(ys, [lbl for _, lbl in LADDER])
    ax.set_xlim(0.1, 0.67)
    ax.set_ylim(-0.6, len(LADDER) - 0.2)
    ax.set_xlabel("median IoU with the human label (95% CI)")
    ax.grid(axis="y", visible=False)
    for hb, mk in (("high", "o"), ("low", "s")):
        ax.plot([], [], mk, color=HB_COLOR[hb], ms=4.2, label=f"{hb} objects")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ fig: template fit

def _binned(x, y, edges):
    idx = np.digitize(x, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 15:
            rows.append((np.sqrt(edges[b] * edges[b + 1]), np.median(y[m]),
                         np.percentile(y[m], 25), np.percentile(y[m], 75)))
    return np.array(rows)


def fig_template_fit(ev: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    edges = np.logspace(np.log10(0.2), np.log10(20), 17)
    for hb in ("high", "low"):
        g = ev[ev["height_bin"] == hb]
        for col, ls, lbl in (("v1_iou", "-", "v1"), ("cls_appr_iou", "--", "v2")):
            b = _binned(g["gt_area_m2"].to_numpy(), g[col].to_numpy(), edges)
            if col == "v1_iou":
                ax.fill_between(b[:, 0], b[:, 2], b[:, 3], color=HB_COLOR[hb], alpha=0.12, lw=0)
            ax.plot(b[:, 0], b[:, 1], ls, color=HB_COLOR[hb], lw=1.5 if col == "v1_iou" else 1.2,
                    label=f"{hb}, {lbl}")
    tmpl = V1_BOX_LENGTH_M * V1_BOX_WIDTH_M
    ax.axvline(tmpl, color=INK2, lw=0.8, ls=":")
    ax.text(tmpl * 1.06, 0.93, f"v1 template\n{tmpl:.2f} m$^2$", color=INK2, fontsize=6.5,
            va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.25, 18)
    ax.set_ylim(0, 1)
    ax.set_xlabel("human box area (m$^2$, log scale)")
    ax.set_ylabel("IoU (binned median)")
    ax.legend(ncol=2, frameon=False, loc="upper right", columnspacing=1.0, handlelength=1.8,
              bbox_to_anchor=(1.0, 0.80))
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ fig: per object

def fig_objects(objs: pd.DataFrame, path: Path):
    o = objs.sort_values("cls_appr_iou").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for i, r in o.iterrows():
        c = HB_COLOR[r["height_bin"]]
        ax.plot([r["v1_iou"], r["cls_appr_iou"]], [i, i], color=c, lw=1.0, alpha=0.55)
        ax.plot(r["v1_iou"], i, "o", ms=3.6, mfc="white", mec=c, mew=1.0, zorder=3)
        ax.plot(r["cls_appr_iou"], i, "o", ms=4.2, color=c, mec="white", mew=0.5, zorder=3)
        ax.text(0.80, i, f"{int(r['n'])}", va="center", ha="right", fontsize=6.5, color=INK2)
    ax.text(0.80, len(o) - 0.3, "n", ha="right", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(o)), [OBJ_LABEL.get(x, x) for x in o["obj"]])
    ax.set_xlim(0.05, 0.81)
    ax.set_ylim(-0.7, len(o) - 0.1)
    ax.set_xlabel("median IoU:  open = v1,  filled = v2")
    ax.grid(axis="y", visible=False)
    for hb in ("high", "low"):
        ax.plot([], [], "o", color=HB_COLOR[hb], ms=4, label=hb)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.92, 0.0), handletextpad=0.2)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ fig: examples

def _local(poly, origin, yaw):
    """World -> frame with the map object at the origin and ego heading = +y."""
    p = np.asarray(poly, float) - origin
    a = np.pi / 2 - yaw
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return p @ rot.T


def _pick(ev, mask, col="cls_appr_iou"):
    g = ev[mask]
    return g.iloc[(g[col] - g[col].median()).abs().argsort().iloc[0]]


def fig_examples(ev: pd.DataFrame, path: Path):
    cases = [
        ("Displaced pole", (ev["obj"] == "pole") & (ev["approach"] == "displaced")),
        # "car" also absorbs the Car-45-Degree placement: keep cars square to the ego
        ("Car, central", (ev["obj"] == "car") & (ev["approach"] == "central")
         & ((ev["gt_axis_vs_ego_deg"] < 15) | (ev["gt_axis_vs_ego_deg"] > 75))),
        ("Curbstone, central", (ev["obj"] == "curbstone") & (ev["approach"] == "central")),
        ("Curbstone, 45$^\\circ$", (ev["obj"] == "curbstone") & (ev["approach"] == "45deg")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.15))
    for ax, (title, mask) in zip(axes, cases):
        r = _pick(ev, mask)
        train = ev[ev["sequence_id"].map(lambda s: fold_of(s, K_FOLDS))
                   != fold_of(r["sequence_id"], K_FOLDS)]
        v2, _ = prior_box(r, fit_priors(train, ["obj", "approach"], use_offset=True))
        face = np.array([r["face_cx"], r["face_cy"]])
        tmpl = oriented_box(face[0], face[1], r["ego_yaw_rad"], V1_BOX_LENGTH_M, V1_BOX_WIDTH_M)
        gt = polygon_from_str(r["gt_polygon"])
        side = int(r["obj_side"])
        yaw = r["ego_yaw_rad"] if side > 0 else r["ego_yaw_rad"] + np.pi  # car below object
        shapes = [(gt, INK, "-", 1.3, "human label"),
                  (tmpl, ORANGE, (0, (3, 1.5)), 1.1, "fixed template (+ lateral)"),
                  (v2, AQUA, "-", 1.4, "v2")]
        pts = []
        for poly, col, ls, lw, lbl in shapes:
            q = _local(poly, face, yaw)
            q = np.vstack([q, q[:1]])
            pts.append(q)
            ax.plot(q[:, 0], q[:, 1], color=col, ls=ls, lw=lw, label=lbl)
        ax.plot(0, 0, "x", color=INK, ms=5, mew=1.2, label="USS map object")
        allp = np.vstack(pts)
        span = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 2.0) / 2 + 0.35
        cx = (allp[:, 0].min() + allp[:, 0].max()) / 2
        cy = (allp[:, 1].min() + allp[:, 1].max()) / 2
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.annotate("", xy=(cx - span * 0.75, cy - span * 0.45),
                    xytext=(cx - span * 0.75, cy - span * 0.95),
                    arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.8})
        ax.text(cx - span * 0.68, cy - span * 0.72, "ego", fontsize=6.5, color=INK2)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nIoU v2 {r['cls_appr_iou']:.2f}  "
                     f"(template {r['lat_pos_iou']:.2f})", fontsize=7, color=INK)
        ax.tick_params(labelsize=6)
    h, lbls = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.90))
    fig.legend(h, lbls, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ tables

def table_ladder(lad: pd.DataFrame, path: Path):
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{median IoU} & IoU$\geq$0.5 & centroid \\",
             r"\cmidrule(lr){2-4}",
             r"Variant & all (95\% CI) & high & low & all & err.\ (m) \\", r"\midrule"]
    for v, lbl in LADDER:
        a = lad[(lad["variant"] == v) & (lad["subset"] == "all")].iloc[0]
        h = lad[(lad["variant"] == v) & (lad["subset"] == "high")].iloc[0]
        lo = lad[(lad["variant"] == v) & (lad["subset"] == "low")].iloc[0]
        ci = _ci(a["iou_med_ci"])
        lines.append(f"{lbl} & {a['iou_med']:.3f} ({ci[0]:.3f}--{ci[1]:.3f}) & "
                     f"{h['iou_med']:.3f} & {lo['iou_med']:.3f} & "
                     f"{100 * a['iou>=0.5']:.1f}\\% & {a['centroid_med_m']:.2f} \\\\")
    lines.append(r"\midrule")
    for v, lbl in (("depth", "Depth shift (all classes)"),
                   ("cls_gtaxis", "Oracle: true orientation"),
                   ("cls_oraclecentre", "Oracle: true centre")):
        a = lad[(lad["variant"] == v) & (lad["subset"] == "all")].iloc[0]
        h = lad[(lad["variant"] == v) & (lad["subset"] == "high")].iloc[0]
        lo = lad[(lad["variant"] == v) & (lad["subset"] == "low")].iloc[0]
        lines.append(f"\\textit{{{lbl}}} & {a['iou_med']:.3f} & {h['iou_med']:.3f} & "
                     f"{lo['iou_med']:.3f} & {100 * a['iou>=0.5']:.1f}\\% & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_approach(appr: pd.DataFrame, path: Path):
    lines = [r"\begin{tabular}{@{}llrcccc@{}}", r"\toprule",
             r"Approach & Height & $n$ & v1 & v2 & centr.\ v1 (m) & centr.\ lat.\ (m) \\",
             r"\midrule"]
    for _, r in appr[appr["approach"] != "untagged"].iterrows():
        a = r["approach"].replace("45deg", r"45$^\circ$")
        lines.append(f"{a} & {r['height_bin']} & {int(r['n'])} & {r['v1']:.3f} & "
                     f"{r['cls_appr']:.3f} & {r['v1_centroid']:.2f} & {r['lat_centroid']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_classes(ev: pd.DataFrame, path: Path, min_n: int = 30):
    """Evaluation set composition: class x height bin x approach geometry."""
    ev = ev.copy()
    ev["approach"] = ev["approach"].fillna("untagged")
    t = (ev.groupby(["height_bin", "obj"])
           .agg(n=("v1_iou", "size"), area=("gt_area_m2", "median"),
                central=("approach", lambda a: int((a == "central").sum())),
                displaced=("approach", lambda a: int((a == "displaced").sum())),
                deg45=("approach", lambda a: int((a == "45deg").sum())),
                untagged=("approach", lambda a: int((a == "untagged").sum())))
           .reset_index())
    small = t[t["n"] < min_n]
    t = t[t["n"] >= min_n].sort_values(["height_bin", "n"], ascending=[True, False])
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"Class & Height & $n$ & area (m$^2$) & central & displ. & 45$^\circ$ & untagged \\",
             r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{OBJ_LABEL.get(r['obj'], r['obj'])} & {r['height_bin']} & {r['n']} & "
                     f"{r['area']:.2f} & {r['central']} & {r['displaced']} & {r['deg45']} & "
                     f"{r['untagged']} \\\\")
    if len(small):
        lines.append(f"other ({len(small)} classes) & -- & {int(small['n'].sum())} & -- & "
                     f"{int(small['central'].sum())} & {int(small['displaced'].sum())} & "
                     f"{int(small['deg45'].sum())} & {int(small['untagged'].sum())} \\\\")
    tot = ev["approach"].value_counts()
    lines += [r"\midrule",
              f"total & & {len(ev)} & & {tot.get('central', 0)} & {tot.get('displaced', 0)} & "
              f"{tot.get('45deg', 0)} & {tot.get('untagged', 0)} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main(run_dir: Path, out_dir: Path):
    a = run_dir / "analysis"
    figs, tabs = out_dir / "figures", out_dir / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    lad = pd.read_csv(a / "ladder.csv")
    objs = pd.read_csv(a / "by_object.csv")
    appr = pd.read_csv(a / "by_approach.csv")
    ev = pd.read_csv(a / "eval_enriched.csv", low_memory=False)
    fig_ladder(lad, figs / "fig_ladder.pdf")
    fig_template_fit(ev, figs / "fig_template_fit.pdf")
    fig_objects(objs, figs / "fig_objects.pdf")
    fig_examples(ev, figs / "fig_examples.pdf")
    table_ladder(lad, tabs / "tab_ladder.tex")
    table_approach(appr, tabs / "tab_approach.tex")
    table_classes(ev, tabs / "tab_classes.tex")
    print("wrote", sorted(p.name for p in figs.iterdir()), sorted(p.name for p in tabs.iterdir()))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
