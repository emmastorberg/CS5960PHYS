"""
Parallel magic state distillation factory model.

Given a fixed total physical qubit budget, computes how many 15-to-1 distillation
factories each code can run in parallel, and the resulting wall-clock time to
execute 2^80 T-gates — the T-gate count for a Grover-based attack on AES-128
as cited in thesis section 3.4 (Grassl et al. [thesis ref]).

The central question: does the BB code's factor-of-12 qubit advantage per
factory translate into a meaningful runtime improvement at any plausible scale?

LOWER BOUND DISCLAIMER: same as t_gate_cost_model.py.  All qubit estimates
omit routing, scheduling, ancilla, and classical control overhead.  The cycle
time assumption is particularly important here; see CYCLE_TIME_S below.

Shared parameters are imported directly from t_gate_cost_model.py so both
scripts stay consistent when parameters change.
"""

import os
import sys
import io
import math
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Import shared parameters and functions from t_gate_cost_model.py ─────────
# The module has top-level code (plots, CSV, print) that runs on import.
# We suppress stdout to avoid interleaving its summary with ours.

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "t_gate_cost_model",
    os.path.join(_here, "t_gate_cost_model.py"),
)
_tgcm = importlib.util.module_from_spec(_spec)
_saved_stdout = sys.stdout
sys.stdout = io.StringIO()
_spec.loader.exec_module(_tgcm)
sys.stdout = _saved_stdout

# Shared constants
P_PHYS              = _tgcm.P_PHYS
P_TH_SURF           = _tgcm.P_TH_SURF
P_TH_BB             = _tgcm.P_TH_BB
BB_N                = _tgcm.BB_N
BB_K                = _tgcm.BB_K
BB_D                = _tgcm.BB_D
BB_ENCODING_RATIO   = _tgcm.BB_ENCODING_RATIO
DISTILL_LOGICAL_INPUTS = _tgcm.DISTILL_LOGICAL_INPUTS
COLOR_SURF          = _tgcm.COLOR_SURF
COLOR_BB            = _tgcm.COLOR_BB

# Factory qubit and cycle costs at the representative d=12 operating point.
# Both codes share d=12, so cycle counts are equal; the qubit difference is
# the encoding-ratio factor of 12 established in t_gate_cost_model.py.
Q_FACTORY_SURF  = _tgcm.factory_qubits_surf(_tgcm.BB_D)   # 15 × 12² = 2160
Q_FACTORY_BB    = _tgcm.factory_qubits_bb()                # 15 × 12  = 180
T_CYCLES_SURF   = _tgcm.factory_cycles_surf(_tgcm.BB_D)    # 12
T_CYCLES_BB     = _tgcm.factory_cycles_bb()                 # 12

# ── Parameters specific to this experiment ────────────────────────────────────

# T-gate count for a Grover-based attack on AES-128.
# Sourced from section 3.4 of the thesis, citing Grassl et al. (2015):
# "on the order of 2^80 T-gates would be required".
T_GATE_COUNT = 2 ** 80

# Code cycle time in seconds.
# ASSUMPTION: 1 microsecond per syndrome measurement cycle, a standard
# estimate for superconducting qubit architectures.  Surface code cycle times
# of ~1 µs are widely cited in the literature and are consistent with the
# hardware context in Jaques' argument.  This assumption is applied equally
# to both codes; if BB code cycles are slower due to long-range connectivity,
# the BB code's advantage would be reduced.
CYCLE_TIME_S = 1e-6   # seconds per code cycle

SECS_PER_YEAR = 365.25 * 24 * 3600   # 3.156 × 10^7 s

# Total physical qubit budget range.
#
# Lower bound: current state of the art in superconducting quantum processors.
# The thesis (section 3.4) states "the largest quantum processors contain on
# the order of a few thousand physical qubits".  We use 5,000 as a round
# conservative lower bound.
# ASSUMPTION: 5,000 physical qubits as of 2025 state-of-the-art.
# Flag: if a more precise figure is available from a hardware citation, replace
# this value.
BUDGET_SOTA = 5_000

# Upper bound: 3.9 million physical qubits, drawn from the estimate in
# section 3.4 of the thesis ("our 13,000 machines require 13,000 × 300 ≈
# 3.9 million [qubits]").  We treat this as a physical qubit count at the
# BB code's 12:1 encoding ratio, representing a plausible near-to-medium-term
# engineering target.
BUDGET_THESIS = 3_900_000

# Runtime thresholds for the "how many qubits does it take" analysis.
# These are chosen to bracket the range from "still completely intractable"
# (1,000 years) to "arguably within civilizational planning horizons" (50 years).
RUNTIME_THRESHOLDS_YEARS = [1_000, 100, 50]

# Physical qubit budget sweep: 50 log-spaced points from SOTA to thesis budget.
N_BUDGET_POINTS = 200
budgets = np.logspace(
    np.log10(BUDGET_SOTA),
    np.log10(BUDGET_THESIS),
    N_BUDGET_POINTS,
)

# ── Core computations ─────────────────────────────────────────────────────────

def n_parallel_factories(total_qubits: float, qubits_per_factory: int) -> int:
    """
    Number of distillation factories that fit within a total physical qubit
    budget.  Each factory requires qubits_per_factory physical qubits;
    factories run in parallel so each active factory contributes one T-gate
    per distillation round.
    """
    return max(1, int(total_qubits // qubits_per_factory))


def runtime_years(n_factories: int, t_cycles_per_tgate: int) -> float:
    """
    Wall-clock time in years to execute T_GATE_COUNT T-gates with n_factories
    factories running in parallel.

    Each factory produces one T-gate every t_cycles_per_tgate code cycles.
    With n_factories running simultaneously, the total number of serial
    T-gate rounds required is ceil(T_GATE_COUNT / n_factories).
    Total wall-clock time = rounds × t_cycles_per_tgate × CYCLE_TIME_S.
    """
    rounds = math.ceil(T_GATE_COUNT / n_factories)
    total_seconds = rounds * t_cycles_per_tgate * CYCLE_TIME_S
    return total_seconds / SECS_PER_YEAR


def qubits_for_runtime_threshold(threshold_years: float,
                                  t_cycles_per_tgate: int,
                                  qubits_per_factory: int) -> float:
    """
    Total physical qubits needed for the wall-clock time to fall below
    threshold_years.

    Rearranging runtime_years: n_factories_needed = T_GATE_COUNT × cycles ×
    CYCLE_TIME_S / (threshold_years × SECS_PER_YEAR).
    Physical qubits = n_factories_needed × qubits_per_factory.
    """
    n_needed = math.ceil(
        T_GATE_COUNT * t_cycles_per_tgate * CYCLE_TIME_S
        / (threshold_years * SECS_PER_YEAR)
    )
    return n_needed * qubits_per_factory


# ── Generate results table ────────────────────────────────────────────────────

rows = []
for Q in budgets:
    nf_surf = n_parallel_factories(Q, Q_FACTORY_SURF)
    nf_bb   = n_parallel_factories(Q, Q_FACTORY_BB)
    rt_surf = runtime_years(nf_surf, T_CYCLES_SURF)
    rt_bb   = runtime_years(nf_bb,   T_CYCLES_BB)
    speedup = rt_surf / rt_bb   # > 1 means BB code is faster
    rows.append(dict(
        total_physical_qubits=Q,
        n_factories_surf=nf_surf,
        n_factories_bb=nf_bb,
        runtime_years_surf=rt_surf,
        runtime_years_bb=rt_bb,
        speedup_bb_over_surf=speedup,
    ))

df = pd.DataFrame(rows)

# ── Compute summary quantities ────────────────────────────────────────────────

# Speedup at the thesis's 3.9 million qubit budget.
thesis_row = df.iloc[(df["total_physical_qubits"] - BUDGET_THESIS).abs().argsort().iloc[0]]
nf_surf_thesis = int(thesis_row["n_factories_surf"])
nf_bb_thesis   = int(thesis_row["n_factories_bb"])
rt_surf_thesis = thesis_row["runtime_years_surf"]
rt_bb_thesis   = thesis_row["runtime_years_bb"]
speedup_thesis = thesis_row["speedup_bb_over_surf"]

# Serial runtime (1 factory each): baseline with no parallelism.
rt_surf_serial = runtime_years(1, T_CYCLES_SURF)
rt_bb_serial   = runtime_years(1, T_CYCLES_BB)

# Physical qubit requirements to cross each runtime threshold.
threshold_results = {}
for thr in RUNTIME_THRESHOLDS_YEARS:
    q_surf = qubits_for_runtime_threshold(thr, T_CYCLES_SURF, Q_FACTORY_SURF)
    q_bb   = qubits_for_runtime_threshold(thr, T_CYCLES_BB,   Q_FACTORY_BB)
    threshold_results[thr] = dict(q_surf=q_surf, q_bb=q_bb)

# ── Save CSV ──────────────────────────────────────────────────────────────────

out_dir = os.path.join(_here, "results")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "parallel_factory_comparison.csv")
df.to_csv(csv_path, index=False, float_format="%.6e")

# ── Plotting ──────────────────────────────────────────────────────────────────

fig_dir = os.path.join(_here, "figures")
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# ── Figure 1: Wall-clock runtime in years vs. total physical qubit budget ────

fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.loglog(df["total_physical_qubits"], df["runtime_years_surf"],
           color=COLOR_SURF, lw=2, label=r"Toric surface code")
ax1.loglog(df["total_physical_qubits"], df["runtime_years_bb"],
           color=COLOR_BB,   lw=2, label=r"BB gross code $[\![144,12,12]\!]$")

# Reference threshold lines with direct labels.
ref_colors = ["#444444", "#666666", "#999999"]
for thr, col in zip(RUNTIME_THRESHOLDS_YEARS, ref_colors):
    ax1.axhline(thr, color=col, lw=1, ls="--", alpha=0.8)
    ax1.text(BUDGET_THESIS * 1.02, thr, rf"${thr}$ yr",
             va="center", ha="left", fontsize=9, color=col)

# Mark the thesis budget.
ax1.axvline(BUDGET_THESIS, color="gray", lw=1, ls=":", alpha=0.6)
ax1.text(BUDGET_THESIS * 1.02, ax1.get_ylim()[0] * 3,
         r"$3.9\!\times\!10^6$ (thesis)", fontsize=8, color="gray",
         rotation=90, va="bottom")

ax1.set_xlabel(r"Total physical qubit budget")
ax1.set_ylabel(r"Wall-clock runtime (years)")
ax1.set_title(
    r"Runtime to execute $2^{80}$ T-gates with parallel distillation factories"
    "\n"
    r"Toric surface code vs.\ BB gross code $[\![144,12,12]\!]$,"
    r" $p_{\mathrm{phys}}=0.001$, cycle time $= 1\,\mu$s",
    fontsize=11,
)
ax1.legend(loc="upper right")
ax1.set_xlim(BUDGET_SOTA * 0.8, BUDGET_THESIS * 5)
fig1.tight_layout()
fig1.savefig(os.path.join(fig_dir, "parallel_runtime.pdf"), bbox_inches="tight")
plt.close(fig1)

# ── Figure 2: Number of parallel factories vs. total physical qubit budget ───

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.loglog(df["total_physical_qubits"], df["n_factories_surf"],
           color=COLOR_SURF, lw=2, label=r"Toric surface code")
ax2.loglog(df["total_physical_qubits"], df["n_factories_bb"],
           color=COLOR_BB,   lw=2, label=r"BB gross code $[\![144,12,12]\!]$")

ax2.axvline(BUDGET_THESIS, color="gray", lw=1, ls=":", alpha=0.6)

# Annotate the speedup factor at the thesis budget.
ax2.annotate(
    rf"$\times {speedup_thesis:.0f}$ speedup at $3.9\!\times\!10^6$ qubits",
    xy=(BUDGET_THESIS, nf_bb_thesis),
    xytext=(BUDGET_THESIS / 10, nf_bb_thesis * 2),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
)

ax2.set_xlabel(r"Total physical qubit budget")
ax2.set_ylabel(r"Number of parallel T-gate factories")
ax2.set_title(
    r"Parallel distillation factories vs.\ physical qubit budget"
    "\n"
    r"Toric surface code vs.\ BB gross code $[\![144,12,12]\!]$,"
    r" $p_{\mathrm{phys}}=0.001$",
    fontsize=11,
)
ax2.legend(loc="upper left")
ax2.set_xlim(BUDGET_SOTA * 0.8, BUDGET_THESIS * 5)
fig2.tight_layout()
fig2.savefig(os.path.join(fig_dir, "parallel_factories.pdf"), bbox_inches="tight")
plt.close(fig2)

# ── Print summary ─────────────────────────────────────────────────────────────

W = 72

def hr():
    print("=" * W)

def sec(title):
    print()
    print(title)
    print("-" * len(title))

hr()
print("PARALLEL FACTORY MODEL")
print("Toric surface code vs. BB gross code [[144, 12, 12]]")
print(f"2^80 T-gates | cycle time = {CYCLE_TIME_S*1e6:.0f} µs | p_phys = {P_PHYS}")
hr()

sec("MODEL PARAMETERS (inherited from t_gate_cost_model.py)")
print(f"  Surface code factory qubits:   {Q_FACTORY_SURF}  (15 × 12² at d=12)")
print(f"  BB gross code factory qubits:  {Q_FACTORY_BB}  (15 × 12)")
print(f"  Surface code factory cycles:   {T_CYCLES_SURF}")
print(f"  BB gross code factory cycles:  {T_CYCLES_BB}")
print(f"  T-gate count (Grassl et al.):  2^80 ≈ {T_GATE_COUNT:.3e}")
print(f"  Code cycle time (ASSUMPTION):  {CYCLE_TIME_S*1e6:.0f} µs  [flag: superconducting estimate]")
print(f"  SOTA qubit budget (ASSUMPTION):{BUDGET_SOTA:>12,}  [flag: conservative 2025 estimate]")
print(f"  Thesis qubit budget:           {BUDGET_THESIS:>12,}  (section 3.4)")

sec("SERIAL RUNTIME BASELINE  (one factory, no parallelism)")
print(f"  Toric surface code:   {rt_surf_serial:.3e} years")
print(f"  BB gross code:        {rt_bb_serial:.3e} years")
print(f"  (Equal because both codes use d=12 cycles per T-gate.)")

sec(f"AT THE THESIS BUDGET  ({BUDGET_THESIS:,} total physical qubits)")
print(f"  Parallel factories (surface):  {nf_surf_thesis:,}")
print(f"  Parallel factories (BB code):  {nf_bb_thesis:,}")
print(f"  Runtime (surface):             {rt_surf_thesis:.3e} years")
print(f"  Runtime (BB code):             {rt_bb_thesis:.3e} years")
print(f"  Speedup (BB over surface):     {speedup_thesis:.1f}×")

sec("PHYSICAL QUBITS REQUIRED TO CROSS RUNTIME THRESHOLDS")
print(f"  {'Threshold':<15} {'Surface code':>20} {'BB gross code':>20}")
print(f"  {'-'*55}")
for thr in RUNTIME_THRESHOLDS_YEARS:
    q_s = threshold_results[thr]["q_surf"]
    q_b = threshold_results[thr]["q_bb"]
    print(f"  {thr} years{'':<7} {q_s:>20,.0f} {q_b:>20,.0f}")
print()
print("  For context: current SOTA ≈ 5,000 qubits; thesis budget = 3.9 M qubits.")
print("  All threshold-crossing qubit counts far exceed any plausible near-term")
print("  or medium-term hardware projection.")

print()
print("-" * W)
print("THESIS PLACEHOLDER VALUE: at matched physical qubit budget, BB code")
print(f"reduces wall-clock runtime by a factor of approximately "
      f"{int(round(speedup_thesis))} relative to")
print("the toric surface code.")
print()
print("NOTE: This speedup equals the encoding-ratio factor (12) from the")
print("first experiment because, at matched d=12, cycle counts are equal")
print("and the only advantage of the BB code is fitting more factories into")
print("the same qubit budget.  A factor-of-12 improvement on a runtime of")
print(f"~{rt_surf_thesis:.0e} years yields ~{rt_bb_thesis:.0e} years — still completely")
print("intractable.  The BB code's qubit advantage does not translate into")
print("a meaningful runtime improvement at any physically plausible scale.")
print("-" * W)

sec("OUTPUT FILES")
print(f"  CSV:    {os.path.abspath(csv_path)}")
print(f"  Plot 1: {os.path.abspath(os.path.join(fig_dir, 'parallel_runtime.pdf'))}")
print(f"  Plot 2: {os.path.abspath(os.path.join(fig_dir, 'parallel_factories.pdf'))}")
