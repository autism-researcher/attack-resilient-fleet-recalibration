"""Figure: collaboration value (Fleet-Gated minus Local-Blend) vs heterogeneity and fleet size.
Reads results_real/localblend_hetero.json -> figures/collab_value_hetero.png"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("results_real/localblend_hetero.json"))
cols = {20: ("#2a78d6", "o"), 50: ("#eb6834", "s"), 100: ("#1baf7a", "^")}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex="row")
rows = [("real", "Real replay (NGSIM base, Waymo alternative)"), ("controlled", "Controlled generator (4 alternative regimes)")]
for i, (s, title) in enumerate(rows):
    fr = d["setups"][s]["fracs"]
    for j, (met, scale, lab) in enumerate([("latency", 1, "Latency difference (steps)"), ("rate_err", 1e3, r"Rate-error difference ($\times10^{-3}$)")]):
        a = ax[i, j]; a.axhline(0, color="#8a8a85", lw=1)
        for k, off in zip((20, 50, 100), (-0.012, 0, 0.012)):
            m = np.array([d["setups"][s]["cells"][f"K{k}_f{f}"]["P5_minus_B1S"][met]["mean_diff"] for f in fr]) * scale
            x = np.array(fr) * 100 + off * 100
            c, mk = cols[k]
            a.errorbar(x, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], color=c, marker=mk, ms=5, lw=2, capsize=2, label=f"K = {k}")
        a.set_ylabel(lab); a.grid(axis="y", color="#e6e6e3", lw=0.8); a.set_axisbelow(True)
        if i == 1: a.set_xlabel("Vehicles with a different post-shift regime (%)")
        a.set_title(("(a) " if (i, j) == (0, 0) else "(b) " if (i, j) == (0, 1) else "(c) " if (i, j) == (1, 0) else "(d) ") + title.split(" (")[0], fontsize=9, loc="left")
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, 0.005, "Fleet-Gated minus Local-Blend; below zero = the fleet helps. 95% paired bootstrap CI, 30 seeds.", ha="center", fontsize=8, color="#555550"); fig.tight_layout(rect=(0, 0.03, 1, 0.95)); os.makedirs("figures", exist_ok=True)
fig.savefig("figures/collab_value_hetero.png", dpi=300); print("saved")
