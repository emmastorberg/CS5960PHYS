import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.titlesize": 15,
})

COLOR_PER_MACHINE = "#0072B2"
COLOR_TOTAL       = "#D55E00"

LOG2_T_BASE = 80
LOG2_50YRS = 64 + np.log2(5)

LOG2_VIABLE = LOG2_50YRS
LOG2_P_THRESHOLD = 2 * (LOG2_T_BASE - LOG2_VIABLE)
P_THRESHOLD = 2 ** LOG2_P_THRESHOLD
LOG2_TOTAL_AT_THRESHOLD = LOG2_T_BASE + LOG2_P_THRESHOLD / 2
LOG2_PER_AT_THRESHOLD = LOG2_T_BASE - LOG2_P_THRESHOLD / 2

P_values = np.logspace(0, 9, 1000)
log2_per_machine = LOG2_T_BASE - np.log2(P_values) / 2
log2_total       = LOG2_T_BASE + np.log2(P_values) / 2

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(P_values, log2_per_machine,
        color=COLOR_PER_MACHINE, lw=2.5,
        label=r"$T$-gates per machine: $\frac{2^{80}}{\sqrt{P}}$")

ax.plot(P_values, log2_total,
        color=COLOR_TOTAL, lw=2.5,
        label=r"Total $T$-gates across all machines: $\sqrt{P} \cdot 2^{80}$")

ax.axhline(LOG2_VIABLE, color="gray", lw=1.2, ls="--", alpha=0.8)
ax.text(1.5, LOG2_VIABLE + 0.6,
        r"$2^{66.3}$ operations per machine in 50 years",
        fontsize=12, color="gray")

ax.axvline(P_THRESHOLD, color="gray", lw=1.2, ls=":", alpha=0.8)
ax.text(P_THRESHOLD / 1.3, 60,
        rf"$P \approx 2^{{{LOG2_P_THRESHOLD:.1f}}} \approx 185$ million machines",
        fontsize=12, color="gray", ha="right")

ax.plot(P_THRESHOLD, LOG2_TOTAL_AT_THRESHOLD,
        marker="o", color=COLOR_TOTAL, markersize=7, zorder=5)
ax.annotate(
    rf"Approx. $2^{{{LOG2_TOTAL_AT_THRESHOLD:.1f}}}$ total $T$-gates",
    xy=(P_THRESHOLD * 0.8, LOG2_TOTAL_AT_THRESHOLD + 0.8),
    xytext=(P_THRESHOLD / 400, LOG2_TOTAL_AT_THRESHOLD + 3.0),
    fontsize=12, color=COLOR_TOTAL,
    arrowprops=dict(arrowstyle="->", color=COLOR_TOTAL, lw=0.9),
)

# ax.plot(P_THRESHOLD, LOG2_PER_AT_THRESHOLD,
#         marker="o", color=COLOR_PER_MACHINE, markersize=7, zorder=5)
# ax.annotate(
#     rf"Per machine $\approx 2^{{{LOG2_PER_AT_THRESHOLD:.1f}}}$ T-gates",
#     xy=(P_THRESHOLD * 0.8, LOG2_PER_AT_THRESHOLD - 0.8),
#     xytext=(P_THRESHOLD / 400, LOG2_PER_AT_THRESHOLD - 4.0),
#     fontsize=12, color=COLOR_PER_MACHINE,
#     arrowprops=dict(arrowstyle="->", color=COLOR_PER_MACHINE, lw=0.9),
# )

ax.set_xscale("log")
ax.set_xlabel(r"Number of machines $P$")
ax.set_ylabel(r"$\log_2$($T$-gate count)")
ax.set_title(
    r"$T$-gate cost of parallelized Grover's algorithm on AES-128"
    # "\n"
    # r"Per-machine cost decreases; total cost increases",
)
ax.legend()
ax.set_xlim(1, 1e9)
ax.set_ylim(50, 100)

fig.tight_layout()

out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "parallelism_tradeoff.pdf"), bbox_inches="tight")
plt.close(fig)

print("=" * 65)
print("PARALLELISM TRADEOFF: Grover's algorithm on AES-128")
print(f"Base T-gate count (P=1, full key space): 2^{LOG2_T_BASE}")
print("=" * 65)
print(f"{'P':>12}  {'T-gates/machine (log2)':>22}  {'Total (log2)':>12}")
print("-" * 65)
for P in [1, 10, 100, 1000, 10000, 100000, int(1e6), int(1e8), int(P_THRESHOLD)]:
    per = LOG2_T_BASE - np.log2(P) / 2
    tot = LOG2_T_BASE + np.log2(P) / 2
    print(f"{P:>12}  {per:>22.1f}  {tot:>12.1f}")
print("=" * 65)
print()
print(f"Viability threshold: 50 years = 2^{LOG2_VIABLE:.1f} T-gates per machine")
print(f"P required to meet threshold:  2^{LOG2_P_THRESHOLD:.1f} = {P_THRESHOLD:.2e} machines")
print(f"Total T-gates at threshold:    2^{LOG2_TOTAL_AT_THRESHOLD:.1f}")
print(f"Factor worse than naive (P=1): 2^{LOG2_TOTAL_AT_THRESHOLD - LOG2_T_BASE:.1f}")