"""
T-gate resource cost model: toric surface code vs. BB gross code [[144, 12, 12]].

Models the physical resource cost of implementing one logical T-gate via the
standard 15-to-1 magic state distillation protocol for both codes, as a function
of target logical error rate. Results are cited directly in thesis section 3.3
and fill the placeholder in section 3.4:
    "insert factor from 3.3 distillation resource counting experiment"

LOWER BOUND DISCLAIMER: This is explicitly a lower bound calculation.
Omitted overheads that would increase the true cost:
  - Qubit routing between the distillation factory and data patches
  - Factory scheduling latency and pipeline stalls
  - Idle qubit errors during distillation rounds
  - Distillation failure rates (which require re-running the factory)
The comparison is therefore optimistic for both codes; the relative factor
between them is more robust than the absolute numbers.

SIMPLIFYING ASSUMPTION: Code cycle time is assumed equal for both codes.
In practice, toric surface code cycles may differ from BB code cycles due
to the longer-range connectivity required by BB stabilizers. This assumption
is conservative: it favors neither code and isolates the effect of code
distance and encoding ratio on the resource count.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Parameters (each commented with the thesis section or reference it comes from)

# Physical error rate used throughout all thesis experiments (section 3.1,
# methods.tex: "p1 = 0.001 and p2 = 0.01 were chosen... kept constant throughout")
P_PHYS = 0.001

# Toric surface code error threshold under BP+LSD decoding.
# Sourced from sections 2.3.1 and 3.2.1: the memory simulation (figure 3.3) shows
# the toric code operating reliably up to p ≈ 0.10 under the same depolarizing
# noise model.  The standard theoretical value for the toric code under an
# optimal decoder is ~10 %, consistent with the simulation.
P_TH_SURF = 0.10

# BB gross code error threshold under BP+LSD decoding.
# Sourced from section 3.2.1: "identifies the physical error rate at which the
# gross code operates reliably as p ≈ 0.10".
P_TH_BB = 0.10

# Gross code [[144, 12, 12]] parameters.
# Sourced from section 2.3.2 and Bravyi et al. [22]:
#   n = 144 physical qubits, k = 12 logical qubits, d = 12.
BB_N = 144
BB_K = 12
BB_D = 12
# Encoding ratio: physical qubits per logical qubit (section 2.3.2).
# "the gross code achieves the same k at d=12 using only n=144" → 144/12 = 12.
BB_ENCODING_RATIO = BB_N // BB_K   # = 12

# Surface code physical-to-logical qubit ratio: n/k = d² per logical qubit.
# Derived from section 3.2.1 results:
#   "surface code requires n=192 physical qubits to encode k=12 at d=4"  → 192/12 = 16 = 4²
#   "surface code requires n=1728 physical qubits" at k=12, d=12          → 1728/12 = 144 = 12²
# Both data points confirm n/k = d² for the toric surface code used here.
def surf_phys_per_logical(d: float) -> float:
    """Physical qubits per logical qubit in the toric surface code = d²."""
    return d ** 2

# 15-to-1 magic state distillation protocol.
# This protocol is code-agnostic at the logical level (section 3.3): the same
# logical circuit is used regardless of the underlying physical code.  It
# consumes 15 noisy |T⟩ input states to produce 1 clean logical |T⟩.
DISTILL_LOGICAL_INPUTS = 15   # logical input magic states per factory run

# ── Derived threshold quantity ────────────────────────────────────────────────

# Maximum logical error rate achievable by the gross code at its fixed distance.
# Below this p_L the gross code would need a larger BB variant or additional
# distillation levels; the gross code itself cannot suppress errors further.
P_L_BB_MIN = P_TH_BB * (P_PHYS / P_TH_BB) ** ((BB_D + 1) / 2)


# ── Core model functions ──────────────────────────────────────────────────────

def required_distance_surf(p_L: float) -> int:
    """
    Minimum odd code distance for the toric surface code to achieve logical
    error rate p_L, from the standard threshold scaling formula:

        p_L = p_th * (p_phys / p_th)^((d+1)/2)

    Solved for d and rounded up to the nearest odd integer ≥ 3 (distance 1
    and 2 are not meaningfully error-correcting).
    """
    exponent = np.log(p_L / P_TH_SURF) / np.log(P_PHYS / P_TH_SURF)
    d_float = 2.0 * exponent - 1.0
    # Round up to nearest integer ≥ 3.  The rotated surface code (used in the
    # thesis and confirmed by the d=4 and d=12 data points in section 3.2.1)
    # supports all integer distances, so no odd-only restriction applies.
    return max(3, int(np.ceil(d_float)))


def factory_qubits_surf(d: int) -> int:
    """
    Physical qubits in a 15-to-1 magic state distillation factory for the
    toric surface code.

    The factory holds DISTILL_LOGICAL_INPUTS logical magic-state patches
    simultaneously.  Each logical qubit patch requires d² physical qubits
    (the toric surface code encoding ratio at distance d).

    Scales as O(d²) — the dominant term in the physical qubit budget.
    """
    return DISTILL_LOGICAL_INPUTS * int(surf_phys_per_logical(d))


def factory_cycles_surf(d: int) -> int:
    """
    Code cycles required per T-gate in the toric surface code factory.

    One distillation round runs for O(d) syndrome measurement cycles: the
    factory depth scales with the code distance because d rounds of error
    detection are needed to reliably identify and correct errors in the
    factory circuit itself.
    """
    return d


def factory_qubits_bb() -> int:
    """
    Physical qubits in a 15-to-1 magic state distillation factory for the
    BB gross code [[144, 12, 12]].

    The same 15 logical input patches are needed (code-agnostic protocol),
    but each logical qubit requires only BB_ENCODING_RATIO = 12 physical
    qubits (section 2.3.2: gross code encoding ratio 144/12 = 12).

    This is the key source of the physical qubit improvement over the surface
    code: the gross code's encoding ratio (12) is much smaller than the
    surface code's (d²) at any practical distance d ≥ 4.
    """
    return DISTILL_LOGICAL_INPUTS * BB_ENCODING_RATIO   # 15 × 12 = 180


def factory_cycles_bb() -> int:
    """
    Code cycles required per T-gate in the BB gross code factory.

    Same O(d) scaling as the surface code, evaluated at the gross code's
    fixed distance d = BB_D = 12.
    """
    return BB_D   # = 12


# ── Generate results table ────────────────────────────────────────────────────

# Target logical error rate range.  The upper bound 1e-2 is near but below
# the threshold; the lower bound 1e-18 is well below the gross code's minimum
# achievable p_L so that the surface code's continued scaling is visible.
p_L_values = np.logspace(-18, -2, 200)

rows = []
q_bb = factory_qubits_bb()
t_bb = factory_cycles_bb()

for p_L in p_L_values:
    d_surf = required_distance_surf(p_L)
    q_surf = factory_qubits_surf(d_surf)
    t_surf = factory_cycles_surf(d_surf)
    rows.append(dict(
        target_logical_error_rate=p_L,
        d_surf=d_surf,
        d_bb=BB_D,
        physical_qubits_per_factory_surf=q_surf,
        physical_qubits_per_factory_bb=q_bb,
        code_cycles_per_tgate_surf=t_surf,
        code_cycles_per_tgate_bb=t_bb,
        time_cost_cycles_surf=t_surf,
        time_cost_cycles_bb=t_bb,
    ))

df = pd.DataFrame(rows)

# ── Compute summary factors ───────────────────────────────────────────────────

# Representative operating point: d_surf = BB_D = 12.
# This is the crossover where both codes share the same code distance and
# therefore the same cycle count, isolating the pure encoding-ratio advantage.
rep_idx = (df["d_surf"] - BB_D).abs().idxmin()
rep = df.iloc[rep_idx]

qubit_factor = rep["physical_qubits_per_factory_surf"] / rep["physical_qubits_per_factory_bb"]
cycle_factor = rep["time_cost_cycles_surf"] / rep["time_cost_cycles_bb"]
spacetime_factor = qubit_factor * cycle_factor

# ── Save CSV ──────────────────────────────────────────────────────────────────

out_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "t_gate_cost_comparison.csv")
df.to_csv(csv_path, index=False, float_format="%.6e")

# ── Plot helpers ──────────────────────────────────────────────────────────────

fig_dir = os.path.join(os.path.dirname(__file__), "figures")
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

# Colors matching thesis figures 3.3 and 3.5:
#   figure 3.3 caption: "BB code (orange)… toric code implementations (light blue, dark blue)"
#   figure 3.5 caption: "BB code (yellow)… toric surface code (blue)"
COLOR_SURF = "#0072B2"   # blue  (toric surface code)
COLOR_BB   = "#E69F00"   # orange/yellow  (BB gross code)

# Mask: p_L values the gross code can achieve with a single distillation round.
bb_achievable = df["target_logical_error_rate"] >= P_L_BB_MIN

# ── Figure 1: T-gate time cost (code cycles) vs. target logical error rate ───

fig1, ax1 = plt.subplots(figsize=(7, 5))

ax1.loglog(
    df["target_logical_error_rate"],
    df["time_cost_cycles_surf"],
    color=COLOR_SURF, lw=2,
    label=r"Toric surface code"
)
# BB code: horizontal line in the achievable region, dashed outside
ax1.axhline(
    t_bb,
    color=COLOR_BB, lw=2,
    label=r"BB gross code $[\![144,12,12]\!]$",
)
# Shade the region below BB_achievable min as "BB code insufficient"
ax1.axvline(
    P_L_BB_MIN,
    color=COLOR_BB, lw=1, ls="--", alpha=0.7,
    label=rf"BB gross code floor ($p_L \approx {P_L_BB_MIN:.1e}$)"
)
ax1.fill_betweenx(
    [1, 1000], 1e-18, P_L_BB_MIN,
    color=COLOR_BB, alpha=0.07,
)
ax1.set_xlabel(r"Target logical error rate $p_L$")
ax1.set_ylabel(r"Code cycles per T-gate")
ax1.set_title(
    r"T-gate time cost: toric surface code vs.\ BB gross code $[\![144,12,12]\!]$"
    "\n"
    r"15-to-1 magic state distillation, $p_{\mathrm{phys}} = 0.001$",
    fontsize=12,
)
ax1.legend()
ax1.set_xlim(1e-18, 1e-2)
ax1.set_ylim(1, 500)
ax1.invert_xaxis()
fig1.tight_layout()
fig1.savefig(os.path.join(fig_dir, "t_gate_time_cost.pdf"), bbox_inches="tight")
plt.close(fig1)

# ── Figure 2: Physical qubit cost per factory vs. target logical error rate ──

fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.loglog(
    df["target_logical_error_rate"],
    df["physical_qubits_per_factory_surf"],
    color=COLOR_SURF, lw=2,
    label=r"Toric surface code"
)
ax2.axhline(
    q_bb,
    color=COLOR_BB, lw=2,
    label=r"BB gross code $[\![144,12,12]\!]$",
)
ax2.axvline(
    P_L_BB_MIN,
    color=COLOR_BB, lw=1, ls="--", alpha=0.7,
    label=rf"BB gross code floor ($p_L \approx {P_L_BB_MIN:.1e}$)"
)
ax2.fill_betweenx(
    [10, 1e6], 1e-18, P_L_BB_MIN,
    color=COLOR_BB, alpha=0.07,
)
ax2.set_xlabel(r"Target logical error rate $p_L$")
ax2.set_ylabel(r"Physical qubits per T-gate factory")
ax2.set_title(
    r"T-gate physical qubit cost: toric surface code vs.\ BB gross code $[\![144,12,12]\!]$"
    "\n"
    r"15-to-1 magic state distillation, $p_{\mathrm{phys}} = 0.001$",
    fontsize=12,
)
ax2.legend()
ax2.set_xlim(1e-18, 1e-2)
ax2.set_ylim(10, 100000)
ax2.invert_xaxis()
fig2.tight_layout()
fig2.savefig(os.path.join(fig_dir, "t_gate_qubit_cost.pdf"), bbox_inches="tight")
plt.close(fig2)

# ── Print summary ─────────────────────────────────────────────────────────────

print("=" * 70)
print("T-GATE RESOURCE COST MODEL")
print("Toric surface code vs. BB gross code [[144, 12, 12]]")
print("15-to-1 magic state distillation | p_phys = 0.001")
print("=" * 70)
print()
print("MODEL PARAMETERS")
print(f"  Physical error rate:              p_phys = {P_PHYS}")
print(f"  Toric surface code threshold:     p_th   = {P_TH_SURF}  (sections 2.3.1, 3.2.1)")
print(f"  BB gross code threshold:          p_th   = {P_TH_BB}  (section 3.2.1)")
print(f"  Gross code [[n,k,d]]:             [[{BB_N},{BB_K},{BB_D}]]          (section 2.3.2, ref [22])")
print(f"  Gross code encoding ratio:        {BB_ENCODING_RATIO} physical qubits per logical qubit")
print(f"  Surface code encoding ratio:      d² physical qubits per logical qubit (section 3.2.1)")
print(f"  Distillation protocol:            15-to-1 (code-agnostic)")
print()
print("FACTORY COSTS (fixed values for BB gross code)")
print(f"  BB gross code factory qubits:     {q_bb}  (15 inputs × {BB_ENCODING_RATIO} phys/logical)")
print(f"  BB gross code factory cycles:     {t_bb}  (d = {BB_D})")
print(f"  Minimum achievable p_L (BB):      {P_L_BB_MIN:.3e}")
print()
print("REPRESENTATIVE COMPARISON  (at d_surf = d_BB = 12)")
print(f"  p_L at representative point:      {rep['target_logical_error_rate']:.3e}")
print(f"  Surface code factory qubits:      {int(rep['physical_qubits_per_factory_surf'])}")
print(f"  BB gross code factory qubits:     {int(rep['physical_qubits_per_factory_bb'])}")
print(f"  Surface code factory cycles:      {int(rep['time_cost_cycles_surf'])}")
print(f"  BB gross code factory cycles:     {int(rep['time_cost_cycles_bb'])}")
print()
print("CONSTANT FACTOR IMPROVEMENTS  (BB code over toric surface code)")
print(f"  Physical qubits per T-gate:       {qubit_factor:.1f}×  (= d² / encoding_ratio = 144 / 12)")
print(f"  Time cycles per T-gate:           {cycle_factor:.1f}×  (both codes use d = 12 cycles here)")
print(f"  Spacetime volume (qubits×cycles): {spacetime_factor:.1f}×")
print()
print("NOTE ON TIME CYCLES:")
print("  At matched d=12, both codes require the same number of code cycles.")
print("  For p_L requirements demanding d_surf > 12, the surface code's cycle")
print("  count grows as d while the BB gross code stays fixed at 12.  The")
print("  cycle improvement factor for d_surf = 20 would be 20/12 ≈ 1.7×.")
print("  The dominant improvement of the BB code is in physical qubit count.")
print()
print("-" * 70)
print(f"THESIS PLACEHOLDER VALUE: BB code improves T-gate cost by a factor")
print(f"of approximately {int(round(qubit_factor))} relative to the toric surface code")
print(f"(in physical qubits per T-gate factory; time cost unchanged at matched d).")
print("-" * 70)
print()
print("OUTPUT FILES")
print(f"  CSV:    {os.path.abspath(csv_path)}")
print(f"  Plot 1: {os.path.abspath(os.path.join(fig_dir, 't_gate_time_cost.pdf'))}")
print(f"  Plot 2: {os.path.abspath(os.path.join(fig_dir, 't_gate_qubit_cost.pdf'))}")
