"""Regenerate every paper/supplement figure from the committed/re-run JSON (RESS version).
Plotting only - no data are computed or modified here.
 * Descriptive condition names (no internal labels); black-and-white safe (grey fills,
   hatches, marker + line-style encoding); single y-axis per panel; 300 dpi.
 * Certified envelopes are read from results_real/certificates.json (corrected constant)
   or from the run's own `theoretical_bound` field - never hard-coded.
Usage:  python make_figures_ress.py [DATA_DIR]      (default: this folder)
        DATA_DIR must contain results/, results_real/ and real/*.npy
Output: figures_ress/*.png
"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.getcwd(), "figures_ress"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "legend.fontsize": 8.5,
                     "hatch.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300})
REAL = os.path.join(BASE, "real") if os.path.isdir(os.path.join(BASE, "real")) else os.path.join(BASE, "..", "real")
def J(p): return json.load(open(os.path.join(BASE, p)))
def err(t): m, lo, hi = t[:3]; return m, [[m - lo], [hi - m]]
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)

NAME = {"B0": "Frozen", "B1": "Local-only", "B2": "Naive fleet-clamp", "P5": "Proposed",
        "AEB": "AEB (fixed TTC)", "FedAvg": "FedAvg", "RobFCP": "Rob-FCP-style"}
BAR = {"B0": dict(facecolor="#eeeeee", hatch="...."), "B1": dict(facecolor="white", hatch="////"),
       "B2": dict(facecolor="#bdbdbd", hatch="\\\\\\\\"), "P5": dict(facecolor="#6e6e6e", hatch=""),
       "AEB": dict(facecolor="#dddddd", hatch="oo"), "FedAvg": dict(facecolor="#d0d0d0", hatch="--"),
       "RobFCP": dict(facecolor="#a8a8a8", hatch="||")}
LINE = {"B1": dict(ls="--", marker="s", mfc="white"), "P5": dict(ls="-", marker="o", mfc="black"),
        "B2": dict(ls=":", marker="^", mfc="0.6")}
CERT = J("results_real/certificates.json")["laws"]

def bars(ax, conds, vals, ylabel, title, rot=0):
    for i, (c, v) in enumerate(zip(conds, vals)):
        m, ye = err(v)
        ax.bar(i, m, 0.68, yerr=ye, capsize=3, edgecolor="black", linewidth=0.7,
               error_kw=dict(ecolor="black", lw=0.9), **BAR[c])
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([NAME[c] for c in conds], rotation=rot)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3)

def lines(ax, xs, series, ylabel, xlabel, logx=False):
    for c, (m, lo, hi) in series.items():
        st = LINE[c]
        ax.errorbar(xs, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], color="black", lw=1.6, ls=st["ls"],
                    marker=st["marker"], ms=6, markerfacecolor=st["mfc"], markeredgecolor="black",
                    capsize=3, elinewidth=0.9, label=NAME[c])
    if logx: ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend()

def tri(d, c, keys): return ([d[c][k][0] for k in keys], [d[c][k][1] for k in keys], [d[c][k][2] for k in keys])

# ---------- Experiment A: accuracy (controlled and real) ----------
def figA(src, out):
    d = J(src); conds = ["B0", "B1", "B2", "P5"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    bars(ax[0], conds, [d[c]["rate_err"] for c in conds], r"Rate error $|\hat\tau-\tau|$", "(a) Intervention-rate error", 12)
    bars(ax[1], conds, [d[c]["missed"] for c in conds], "Missed high-risk rate", "(b) Missed high-risk events", 12)
    save(fig, out)
figA("results/A.json", "expA_accuracy_safety.png")
figA("results_real/A.json", "expA_real.png")

# ---------- Experiment B: adaptation latency vs fleet size ----------
def figB(src, out):
    d = J(src); Ks = [1, 5, 20, 100]; k = [str(x) for x in Ks]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    lines(ax, Ks, {c: tri(d, c, k) for c in ["B1", "P5"]}, "Adaptation latency (steps)", "Fleet size $K$", logx=True)
    save(fig, out)
figB("results/B.json", "expB_latency.png")
figB("results_real/B.json", "expB_real.png")

# ---------- Experiment C: under-protection, honest vs 30% adversarial ----------
def figC(src, out, env_label):
    d = J(src); groups = ["B1", "B2", "P5"]; x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for off, key, lab, st in [(-w/2, "honest", "Honest fleet", dict(facecolor="white", hatch="////")),
                              (w/2, "adv", "30% adversarial uploads", dict(facecolor="#8c8c8c", hatch=""))]:
        for i, g in enumerate(groups):
            m, ye = err(d[g][key])
            ax.bar(x[i] + off, m, w, yerr=ye, capsize=3, edgecolor="black", linewidth=0.7,
                   error_kw=dict(ecolor="black", lw=0.9), label=lab if i == 0 else None, **st)
    b = d["theoretical_bound"][0]
    ax.axhline(b, ls=(0, (6, 3)), color="black", lw=1.3, label=f"{env_label} ({b:.3f})")
    ax.set_xticks(x); ax.set_xticklabels([NAME[g] for g in groups])
    ax.set_ylabel(r"Mean per-vehicle max. $(B^{\mathrm{eff}}-B^\star)_+$")
    ax.set_ylim(0, max(b, max(d[g]["adv"][2] for g in groups)) * 1.25); ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    save(fig, out)
figC("results/C.json", "expC_underprotection.png", "Per-step certified envelope")
figC("results_real/C.json", "expC_real.png", "Per-step certified envelope")

# ---------- Experiment D: rate error vs fleet size (+K* marker by the script's own rule) ----------
def figD(src, out):
    d = J(src); Ks = [1, 2, 3, 5, 10, 20, 50, 100]; k = [str(x) for x in Ks]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    lines(ax, Ks, {c: tri(d, c, k) for c in ["B1", "P5"]}, r"Rate error $|\hat\tau-\tau|$", "Fleet size $K$", logx=True)
    kstar = next((K for K in Ks if d["P5"][str(K)][0] <= 0.9 * d["B1"][str(K)][0]), None)
    if kstar: ax.axvline(kstar, ls=":", color="black", lw=1); ax.text(kstar * 1.08, ax.get_ylim()[1] * 0.95, f"$K^*={kstar}$", va="top")
    ax.set_ylim(bottom=0); save(fig, out)
figD("results/D.json", "expD_threshold.png")
figD("results_real/D.json", "expD_real.png")

# ---------- Fig. 2: adversarial-fraction sweep, real replay (Experiment E) ----------
def figE(out):
    E = J("results_real/E.json"); fr = [f"{x:.1f}" for x in E["fracs"]]; x = [100 * v for v in E["fracs"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    lines(ax, x, {c: ([E[c][f][0] for f in fr], [E[c][f][1] for f in fr], [E[c][f][2] for f in fr]) for c in ["B2", "P5"]},
          r"Mean per-vehicle max. $(B^{\mathrm{eff}}-B^\star)_+$", "Adversarial / stale uploads (%)")
    ng, wy = CERT["ngsim"]["env_step"], CERT["waymo"]["env_step"]
    ax.axhline(ng, ls=(0, (1, 2)), color="0.3", lw=1.4, label=f"Per-step envelope, NGSIM ({ng:.3f})")
    ax.axhline(wy, ls="-.", color="0.5", lw=1.2, label=f"Per-step envelope, Waymo ({wy:.3f})")
    ax.set_ylim(0, 0.13); ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97))
    save(fig, out)
figE("fig2_adv_sweep_real.png")

# ---------- Experiment F: aggregation period (proposed scheme) - two panels, no twin axis ----------
def figF(out):
    F = J("results_real/F.json"); Ps = F["periods"]; k = [str(p) for p in Ps]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for a, key, yl, t in [(ax[0], "rate_err", r"Rate error $|\hat\tau-\tau|$", "(a) Rate error"),
                          (ax[1], "latency", "Adaptation latency (steps)", "(b) Adaptation latency"),
                          (ax[2], "under_max", "Mean per-vehicle max. under-protection", "(c) Under-protection")]:
        m = [F[key][i][0] for i in k]; lo = [F[key][i][1] for i in k]; hi = [F[key][i][2] for i in k]
        a.errorbar(Ps, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="-o", color="black", lw=1.6, ms=6, mfc="black", capsize=3, elinewidth=0.9)
        a.set_xscale("log"); a.set_xticks(Ps); a.set_xticklabels(k); a.set_xlabel("Aggregation period $P$ (steps)")
        a.set_ylabel(yl); a.set_title(t); a.grid(alpha=0.3)
    ax[2].set_ylim(0, max(F["under_max"][i][2] for i in k) * 1.6)
    save(fig, out)
figF("expF_period.png")

# ---------- Experiment G: heterogeneous vs homogeneous shift ----------
def figG(out):
    G = J("results_real/G.json"); regimes = ["heterogeneous", "homogeneous"]; x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for j, (key, yl, t) in enumerate([("rate_err", r"Rate error $|\hat\tau-\tau|$", "(a) Rate error"),
                                      ("latency", "Adaptation latency (steps)", "(b) Adaptation latency")]):
        for off, c in [(-w/2, "B1"), (w/2, "P5")]:
            for i, r in enumerate(regimes):
                m, ye = err(G[r][c][key])
                ax[j].bar(x[i] + off, m, w, yerr=ye, capsize=3, edgecolor="black", linewidth=0.7,
                          error_kw=dict(ecolor="black", lw=0.9), label=NAME[c] if i == 0 else None, **BAR[c])
        ax[j].set_xticks(x); ax[j].set_xticklabels(["Heterogeneous shift", "Homogeneous shift"])
        ax[j].set_ylabel(yl); ax[j].set_title(t); ax[j].grid(axis="y", alpha=0.3); ax[j].legend(loc="lower right")
    save(fig, out)
figG("expG_homvshet.png")

# ---------- Real per-corpus risk distributions ----------
def figReal(out):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, lab, ls, mk in [("highd", "highD", "-", "o"), ("waymo", "Waymo", "--", "s"), ("ngsim", "NGSIM", "-.", "^")]:
        x = np.sort(np.load(os.path.join(REAL, name + ".npy"))); y = np.arange(1, len(x) + 1) / len(x)
        Bs = np.quantile(x, 0.90)
        ax.plot(x, y, ls=ls, color="black", lw=1.5, marker=mk, markevery=600, ms=5, mfc="white", label=f"{lab} ($B^\\star$ = {Bs:.2f})")
        ax.axvline(Bs, ls=ls, color="0.45", lw=1)
    ax.axhline(0.90, ls=":", color="0.6", lw=1); ax.text(0.01, 0.915, r"$1-\tau=0.90$", fontsize=8.5, color="0.35")
    ax.set_xlabel(r"Per-trajectory maximum risk $R_{\max}$"); ax.set_ylabel("Empirical CDF")
    ax.grid(alpha=0.3); ax.legend(loc="lower right"); save(fig, out)
figReal("real_dist.png")

# ---------- Adaptation latency: fleet size and penetration ----------
def figLatency(out):
    B = J("results_real/B.json"); P = J("results_real/penetration.json"); Ks = [1, 5, 20, 100]; k = [str(x) for x in Ks]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    lines(ax[0], Ks, {c: tri(B, c, k) for c in ["B1", "P5"]}, "Adaptation latency (steps)", "Fleet size $K$", logx=True)
    ax[0].set_title("(a) Latency vs. fleet size")
    ps = [0.10, 0.25, 0.50, 0.75, 1.00]
    ax[1].plot([p * 100 for p in ps], [P[f"{p:.2f}"]["latency_P5"] for p in ps], "-o", color="black", lw=1.6, ms=6, mfc="black", label=NAME["P5"])
    ax[1].set_ylim(0, max(P[f"{p:.2f}"]["latency_P5"] for p in ps) * 1.3); ax[1].set_xlabel("Equipped market penetration (%)")
    ax[1].set_ylabel("Adaptation latency (steps)"); ax[1].set_title("(b) Latency vs. penetration"); ax[1].grid(alpha=0.3); ax[1].legend()
    save(fig, out)
figLatency("adaptation_latency.png")

# ---------- Parameter sensitivity: rate error and under-protection in separate rows (one y-scale each) ----------
def figSens(out):
    d = J("results_real/sensitivity.json")
    params = [("gamma", r"Step size $\gamma$", ["0.02", "0.05", "0.1"]), ("N_s", r"Window $N_s$", ["150", "300", "600"]),
              ("m", r"Margin $m$", ["0.01", "0.02", "0.05"])]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.2), sharey="row")
    for j, (key, lab, vals) in enumerate(params):
        x = range(len(vals))
        for r, (mk, yl, ls, fc) in enumerate([("rate_err_P5", r"Rate error $|\hat\tau-\tau|$", "-", "black"),
                                              ("under_max_P5", "Mean per-vehicle max. under-protection", "--", "white")]):
            ax[r][j].plot(x, [d[key][v][mk] for v in vals], ls=ls, marker="o" if r == 0 else "s", color="black", lw=1.6, ms=6, mfc=fc)
            ax[r][j].set_xticks(list(x)); ax[r][j].set_xticklabels(vals); ax[r][j].grid(alpha=0.3)
            if j == 0: ax[r][j].set_ylabel(yl)
        ax[1][j].set_xlabel(lab)
    for r, mk in enumerate(["rate_err_P5", "under_max_P5"]):
        top = max(d[k][v][mk] for k, _, vals in params for v in vals)
        ax[r][0].set_ylim(0, top * 1.2)
    save(fig, out)
figSens("sensitivity.png")

# ---------- Laplace-noise robustness ----------
def figLDP(out):
    d = J("results_real/expI_ldp.json"); eps = [0.5, 1.0, 2.0]; ek = {0.5: "eps0.5", 1.0: "eps1.0", 2.0: "eps2.0"}
    styles = [("K5", "$K=5$", "-", "o"), ("K20", "$K=20$", "--", "s"), ("K100", "$K=100$", "-.", "^")]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for Kk, lab, ls, mk in styles:
        ax[0].plot(eps, [d[Kk][ek[e]]["rate_err_dev_vs_noisefree"] for e in eps], ls=ls, marker=mk, color="black", lw=1.5, ms=6, mfc="white", label=lab)
        ax[1].plot(eps, [d[Kk][ek[e]]["under_max"] for e in eps], ls=ls, marker=mk, color="black", lw=1.5, ms=6, mfc="white", label=lab)
    b = max(d[Kk][ek[e]]["bound"] for Kk, *_ in styles for e in eps)
    ax[1].axhline(b, ls=(0, (6, 3)), color="0.35", lw=1.2, label=f"Per-step certified envelope ({b:.3f})")
    ax[0].set_ylabel("Rate-error deviation from noise-free"); ax[0].set_title("(a) Accuracy cost")
    ax[1].set_ylabel("Run-max. under-protection"); ax[1].set_title("(b) Under-protection"); ax[1].set_ylim(0, b * 1.25)
    for a in ax: a.set_xlabel(r"Laplace budget $\varepsilon_0$ (smaller = noisier)"); a.invert_xaxis(); a.grid(alpha=0.3); a.legend()
    save(fig, out)
figLDP("privacy_utility.png")

# ---------- Closed loop (current closed_loop.py; three single-axis panels) ----------
def figClosed(out):
    d = J("results_real/closed_loop.json"); C = d["conditions"]
    order = ["AEB", "B0", "B1", "FedAvg", "RobFCP", "P5"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    bars(ax[0], order, [C[c]["coll"] for c in order], "Rear-end collisions / 1000 veh-steps", "(a) Collisions by supervisor", 25)
    v = d["v2x"]; vk = ["lat0ms_loss0", "lat300ms_loss20", "lat1000ms_loss50", "realistic_lognormal+GE"]
    xt = ["0 ms\n0%", "300 ms\n20%", "1000 ms\n50%", "log-normal\n+ bursty"]; x = range(4)
    for a, key, yl, t in [(ax[1], "under_max", "Run-max. under-protection", "(b) Proposed: under-protection"),
                          (ax[2], "coll", "Collisions / 1000 veh-steps", "(c) Proposed: collisions")]:
        m = [v[k][key][0] for k in vk]; lo = [m[i] - v[k][key][1] for i, k in enumerate(vk)]; hi = [v[k][key][2] - m[i] for i, k in enumerate(vk)]
        a.errorbar(x, m, yerr=[lo, hi], fmt="-o", color="black", lw=1.5, ms=6, mfc="black", capsize=3)
        a.set_xticks(list(x)); a.set_xticklabels(xt, fontsize=8.5); a.set_xlabel("V2X latency / packet loss")
        a.set_ylabel(yl); a.set_title(t); a.grid(alpha=0.3); a.set_ylim(bottom=0)
    save(fig, out)
figClosed("closed_loop.png")

# ---------- certified envelopes vs observed under-protection over N_s (forensic_nsweep.json) ----------
def figNsweep(out):
    p = os.path.join(BASE, "results_real/forensic_nsweep.json")
    if not os.path.exists(p): return
    d = json.load(open(p)); Ns = d["windows"]
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9))
    sty = [("retro_step", "Retrospective, per-step", dict(ls="-", marker="o", mfc="white", color="0.35")),
           ("online_step", "Online (DKW), per-step", dict(ls="--", marker="s", mfc="white", color="0.35")),
           ("retro_sim", "Retrospective, simultaneous", dict(ls="-", marker="o", mfc="black", color="black")),
           ("online_sim", "Online (DKW), simultaneous", dict(ls="--", marker="s", mfc="black", color="black"))]
    for a, (law, t) in zip(ax[:3], [("controlled", "(a) Controlled"), ("ngsim", "(b) highD$\\to$NGSIM"), ("waymo", "(c) highD$\\to$Waymo")]):
        L = d["laws"][law]
        for k, lab, st in sty: a.plot(Ns, [L[str(n)][k] for n in Ns], lw=1.3, ms=5, label=lab, **st)
        for k, lab, mk in [("observed_run_max", "Observed, run maximum", "v"), ("observed_per_vehicle_max", "Observed, mean per-vehicle max.", "D")]:
            m = np.array([L[str(n)][k][0] for n in Ns]); lo = m - [L[str(n)][k][1] for n in Ns]; hi = [L[str(n)][k][2] for n in Ns] - m
            a.errorbar(Ns, m, yerr=[lo, hi], fmt=":" + mk, color="black", mfc="0.75", ms=5, lw=1.1, capsize=2, label=lab)
        a.axhline(d["m"], color="0.5", lw=0.8, ls="-."); a.text(Ns[0], d["m"] * 1.08, "margin $m$", fontsize=7.5, color="0.4")
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(0.012, 1.0); a.set_xlabel("Window length $N_s$")
        a.set_ylabel("Under-protection / envelope"); a.set_title(t); a.grid(alpha=0.3, which="both")
    ax[0].legend(loc="upper right", fontsize=7, framealpha=0.95)
    for law, mk, lab in [("controlled", "o", "Controlled"), ("ngsim", "s", "NGSIM"), ("waymo", "^", "Waymo")]:
        L = d["laws"][law]; ax[3].plot(Ns, [L[str(n)]["observed_rate_err"][0] for n in Ns], "-" + mk, color="black", mfc="white", ms=5, lw=1.2, label=lab)
    ax[3].set_xscale("log"); ax[3].set_xlabel("Window length $N_s$"); ax[3].set_ylabel("Rate error (proposed)")
    ax[3].set_title("(d) Adaptation cost of long windows"); ax[3].grid(alpha=0.3, which="both"); ax[3].legend(fontsize=8)
    save(fig, out)
figNsweep("nsweep_envelopes.png")

# ---------- fleet benefit vs heterogeneity and fleet size (forensic_hetero.json) ----------
def figHetero(out):
    p = os.path.join(BASE, "results_real/forensic_hetero.json")
    if not os.path.exists(p): return
    d = json.load(open(p)); Ks = d["Ks"]; mk = dict(zip(Ks, ["o", "s", "^", "D", "v"])); gr = dict(zip(Ks, ["0.75", "0.55", "0.35", "0.15", "0.0"]))
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2)); fig.suptitle("Proposed $-$ local-only (paired, 30 seeds; 95% bootstrap CI)", fontsize=10)
    for r, (setup, tag) in enumerate([("real", "Real replay (NGSIM base, Waymo alternative)"), ("controlled", "Controlled (congested base, four alternatives)")]):
        B = d["setups"][setup]
        for c, (met, yl) in enumerate([("latency", "$\\Delta$ latency (steps)"), ("rate_err", "$\\Delta$ rate error")]):
            a = ax[r, c]
            for K_ in Ks:
                pts = {}
                for key, cell in B["cells"].items():
                    if key.startswith(f"K{K_}_"): pts[cell["frac_alt_realized"]] = cell["P5_minus_B1"][met]["mean_diff"]
                xs = sorted(pts); m = np.array([pts[x][0] for x in xs]); lo = m - [pts[x][1] for x in xs]; hi = [pts[x][2] for x in xs] - m
                a.errorbar(xs, m, yerr=[lo, hi], fmt="-" + mk[K_], color=gr[K_], mfc="white" if K_ < 20 else gr[K_], ms=5, lw=1.2, capsize=2, label=f"K = {K_}")
            a.axhline(0, color="black", lw=0.8); a.grid(alpha=0.3)
            a.set_xlabel("Fraction of fleet with a different post-shift regime"); a.set_ylabel(yl)
            a.set_title(f"({'abcd'[2 * r + c]}) {tag}", fontsize=9.5)
    ax[0, 0].legend(fontsize=8, ncol=2)
    save(fig, out)
figHetero("hetero_fleetsize.png")

# ---------- grayscale contact sheet ----------
from PIL import Image
ims = sorted(f for f in os.listdir(OUT) if f.endswith(".png") and not f.startswith("_"))
tiles = [Image.open(os.path.join(OUT, i)).convert("L") for i in ims]; W = 1400
tiles = [t.resize((W, int(t.height * W / t.width))) for t in tiles]
sheet = Image.new("L", (W, sum(t.height for t in tiles)), 255); y = 0
for t in tiles: sheet.paste(t, (0, y)); y += t.height
sheet.save(os.path.join(OUT, "_GRAYSCALE_CHECK.png")); print("figures:", ims)
