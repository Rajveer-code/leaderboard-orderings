"""G4.3 -- AISTATS figures F1-F4, regenerated from the committed JSON results.

F1 rank intervals per model: sampling only, agreement-stratified (eps = 0), eps-robust
   (eps0), and without pooling -- all simultaneous over the 351 orderings.
F2 certified orderings against eps, K=3 and K=2, with the preregistered eps0 marked.
F3 human-judged correctness by K=3 verifier cell, with exact intervals and cell sizes.
F4 the certification-falsification frontier (PROJECTED), with the chosen operating point.

Run: python scripts/fb_figures_aistats.py     Out: manuscript/tmlr/figures/f{1..4}.{png,pdf}
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flipbudget import certify as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "flipbudget"
OUT = ROOT / "manuscript" / "tmlr" / "figures"  # written where the paper reads them; no hand copy
# Okabe-Ito, colour-blind safe
BLUE, ORANGE, GREEN, VERMILLION, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "figure.constrained_layout.use": True})


def short(m):
    return m.split("/")[-1]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        # no creation date in the PDF: its time-zone offset would travel with the supplement
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    **({"metadata": {"CreationDate": None}} if ext == "pdf" else {}))
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def rank_plot(layers, xlabel, name):
    order = sorted(layers[0][1], key=lambda m: (layers[0][1][m][0], layers[1][1][m][0]))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    off = np.linspace(0.25, -0.25, len(layers))
    for (label, ri, colour, mark), dy in zip(layers, off):
        for i, m in enumerate(order):
            lo, hi = ri[m]
            ax.plot([lo, hi], [i + dy] * 2, color=colour, lw=1.6, solid_capstyle="butt",
                    label=label if i == 0 else None)
            ax.plot([lo, hi], [i + dy] * 2, mark, color=colour, ms=2.4)
    ax.set_yticks(range(len(order)), [short(m) for m in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_xlim(0.5, len(order) + 0.5)
    # inside the empty upper-right corner (top models have short intervals on the left), so the saved
    # figure is as wide as its plot and the model names keep their size when scaled to the page
    ax.legend(loc="upper right", ncol=1, fontsize=7.5, frameon=True, framealpha=0.95, edgecolor="#DDDDDD")
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, name)


def f1_rank_intervals(cert):
    """Primary: fixed-board rank intervals, simultaneous over the roster."""
    K3, eps0 = cert["K3"], cert["config"]["eps0_preregistered"]
    fb = K3["fixed_board"]
    layers = [("2024 verifier taken as right (as published)", K3["published_ranks"], GREY, "o"),
              ("with the audit, ε = 0", fb["rank_intervals_eps0"], BLUE, "s"),
              (f"with the audit, ε = {eps0} (reference)", fb["rank_intervals_eps_preregistered"], GREEN, "D")]
    rank_plot(layers, f"rank among the {len(fb['rank_intervals_eps0'])} models on these items",
              "f1_rank_intervals")  # the confidence level is stated in the caption


def f5_rank_intervals_generalising(cert):
    """Secondary: rank intervals that also allow for the items being a sample (appendix)."""
    K3, eps0 = cert["K3"], cert["config"]["eps0_preregistered"]
    layers = [("2024 verifier taken as right, item sampling only", K3["rank_intervals_sampling_only"]["em"],
               GREY, "o"),
              ("with the audit, ε = 0", K3["rank_intervals_simultaneous_eps0"], BLUE, "s"),
              (f"with the audit, ε = {eps0} (reference)", K3["rank_intervals_simultaneous_eps_preregistered"],
               GREEN, "D")]
    rank_plot(layers, f"rank among the {len(K3['published_ranks'])} models, secondary target",
              "f5_rank_intervals_generalising")  # the confidence level is stated in the caption


def f2_eps_curve(cert):
    eps0 = cert["config"]["eps0_preregistered"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for K, colour in (("K3", BLUE), ("K2", ORANGE)):
        c = cert[K]["eps_curve"]
        x = [p["eps"] for p in c]
        ax.plot(x, [p["audit_only_simultaneous"] for p in c], "-", color=colour, lw=1.9,
                label=f"{K[1]} verifiers, primary target")
    c = cert["K3"]["eps_curve"]
    ax.plot(x, [p["combined"] for p in c], "--", color=BLUE, lw=1.0, alpha=0.7,
            label="3 verifiers, secondary, one at a time")
    ax.plot(x, [p["combined_simultaneous"] for p in c], ":", color=BLUE, lw=1.0, alpha=0.7,
            label="3 verifiers, secondary, all at once")
    n_pairs = len(cert["K3"]["eps0_pairs"])
    ax.axvline(eps0, color=GREY, lw=0.9, ls=":")
    ax.annotate(f"reference\ntolerance {eps0}", (eps0, ax.get_ylim()[1] * 0.92), xytext=(eps0 + 0.006,
                ax.get_ylim()[1] * 0.86), fontsize=7.5, color=GREY)
    ax.set_xlabel("tolerance ε on cross-model heterogeneity in the concordant cells")
    ax.set_ylabel(f"certified orderings (of {n_pairs})")
    ax.set_ylim(0, n_pairs * 0.75)
    ax.set_xlim(-0.002, max(x) + 0.002)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(color="#EEEEEE", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "f2_certified_vs_eps")


def f3_cells(cert, amap):
    cells = amap["census_cells"]["k3_sympy"]["27"]
    audit = {c: v for c, v in amap["audit_cells"]["27|k3_sympy|sympy|as_incorrect"].items()}
    conc = cert["K3"]["inputs"]["conc_intervals"]
    order = sorted(cells, key=lambda c: -cells[c])
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for i, c in enumerate(order):
        a = audit.get(c)
        if a and a["n"]:
            p = a["x_correct"] / a["n"]
            # concordant cells show the pooled interval used in the analysis; audited discordant
            # cells show the same KL interval for display only (inference uses per-model bounds)
            lo, hi = conc[c] if c in conc else C.kl_interval(a["x_correct"], a["n"], C.sidak(2, 0.975))
            ax.plot([i], [p], "o", color=BLUE, ms=5)
            ax.plot([i, i], [lo, hi], color=BLUE, lw=1.6)
            ax.annotate(f"{a['x_correct']}/{a['n']}", (i, p), xytext=(i + 0.16, p), fontsize=7, color=GREY)
        else:
            ax.plot([i, i], [0, 1], color=VERMILLION, lw=1.6, alpha=0.5)
            ax.annotate("no labels", (i, 0.5), xytext=(i + 0.16, 0.5), fontsize=7, color=VERMILLION)
    ax.set_xticks(range(len(order)), [f"{c}\nn={cells[c]}" for c in order], fontsize=7.5)
    ax.set_xlabel("cell (2024 verifier, boxed verifier, Math-Verify)")
    ax.set_ylabel("share of audited responses\nwith reader verdict 1",
                  fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "f3_correctness_by_cell")


def f4_frontier(front, op, cert):
    pts = [p for p in front["points"] if p["F"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for cell, colour, mark in (("000", BLUE, "o"), ("111", ORANGE, "s")):
        xs, ys = [], []
        for p in pts:
            rho = p["falsification"][cell]["detectable_rho_80"]
            if rho is None:
                continue
            xs.append(rho)
            ys.append(p["certified_uniform_eps0.01"]["median"])
        ax.plot(xs, ys, mark, color=colour, ms=4, alpha=0.75, label=f"cell {cell}")
    base = cert["K3"]["eps_curve"][int(round(op["eps0"] / 0.0025))]["combined"]
    ax.axhline(base, color=GREY, lw=0.9, ls=":")
    ax.annotate(f"400-item audit alone ({base})", (0.595, base + 0.15), fontsize=7.5, color=GREY, ha="right")
    chosen = next(p for p in pts if p.get("is_operating_point"))
    ax.plot([chosen["falsification"]["000"]["detectable_rho_80"]], [chosen["certified_uniform_eps0.01"]["median"]],
            "*", color=GREEN, ms=13, zorder=5)
    ax.annotate("chosen operating point", (chosen["falsification"]["000"]["detectable_rho_80"] + 0.004,
                chosen["certified_uniform_eps0.01"]["median"]), xytext=(0.255, base + 1.6), fontsize=7.5,
                arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8}, color=GREY, va="center")
    ax.set_xlabel("per-model concordant error detectable at 80% power")
    ax.set_ylabel(f"projected certified orderings at ε = {op['eps0']}")
    observed = cert["K3"].get("with_v3")
    if observed:  # the registered audit's actual count at the chosen operating point
        obs = {q["eps"]: q["combined"] for q in observed["eps_curve"]}[op["eps0"]]
        ax.plot([chosen["falsification"]["000"]["detectable_rho_80"]], [obs], "X", color=VERMILLION, ms=10,
                zorder=6, label="observed (registered audit)")
    ax.set_title("points: simulated label outcomes; cross: observed" if observed else
                 "PROJECTED: simulated label outcomes, not observed", fontsize=8, color=VERMILLION)
    ax.legend(fontsize=7.5, frameon=False, title="falsification screen", title_fontsize=7.5)
    ax.grid(color="#EEEEEE", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "f4_frontier_projected")


def main():
    cert = json.load(open(RES / "certify_results.json", encoding="utf-8"))
    amap = json.load(open(RES / "audit_k3_map.json", encoding="utf-8"))
    front = json.load(open(RES / "g3_frontier.json", encoding="utf-8"))
    op = json.load(open(RES / "g3_operating_point.json", encoding="utf-8"))
    f1_rank_intervals(cert)
    f2_eps_curve(cert)
    f3_cells(cert, amap)
    f4_frontier(front, op, cert)
    f5_rank_intervals_generalising(cert)


if __name__ == "__main__":
    main()
