"""
main.py — Run all experiments and generate all figures for the thesis.

Usage (from repo root):
    python main.py

Figures are written to figures/.
Simulation data is cached to data/ and reused on subsequent runs.
Delete data/ to force a full re-run of the simulations.
"""

import sys
import os

# Make src/ importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import panqec_simulations.BBcode_classes as BBcode
from panqec_simulations.memory_simulation import simulate_code
from panqec_simulations import plotting


# ── Shared configurations ─────────────────────────────────────────────────────

DEPOLARISING = {
    'name': 'GaussianPauliErrorModel',
    'parameters': [{'r_x': 1/3, 'r_y': 1/3, 'r_z': 1/3}],
}

def bp_lsd(gaussian=True, max_bp_iter=1000, lsd_order=10):
    return {
        'name': 'BeliefPropagationLSDDecoder',
        'parameters': [{'max_bp_iter': max_bp_iter, 'lsd_order': lsd_order,
                        'gaussian': gaussian}],
    }


# ── Gross code [[144,12,12]] vs toric code ────────────────────────────────────
#
# Noise model: depolarizing (r_x=r_y=r_z=1/3), the most realistic general case.
#
# Gross code: BBcode_A312_B312 at L_x=12, L_y=6  →  n=144, k=12, d=12
#
# Toric baseline 1 (matched k):
#   6 × toric at L_x=4, L_y=4  →  n_total=192, k=12, d=4
#   Same number of logical qubits, but 33% more physical qubits and d=4 vs d=12.
#
# Toric baseline 2 (matched k and d):
#   6 × toric at L_x=12, L_y=12  →  n_total=1728, k=12, d=12
#   Same k and d, but 12× more physical qubits.

def run_gross_vs_toric():

    gross, gross_input, _ = simulate_code(
        BBclass=BBcode.BBcode_A312_B312,
        decoder_dict=bp_lsd(gaussian=True),
        error_model_dict=DEPOLARISING,
        n_trials=500,
        grids=[{'L_x': 12, 'L_y': 6}],
        p_range=(0, 0.3, 60),
        ask_overwrite=False,
    )

    # Toric matched k=12: 6 patches of L=4  (n_total=192, d=4)
    toric_k, toric_k_input, _ = simulate_code(
        BBclass=BBcode.BBcode_Toric,
        decoder_dict=bp_lsd(gaussian=True),
        error_model_dict=DEPOLARISING,
        n_trials=500,
        grids=6 * [{'L_x': 4, 'L_y': 4}],
        p_range=(0, 0.3, 60),
        ask_overwrite=False,
    )

    # Toric matched k=12, d=12: 6 patches of L=12  (n_total=1728, d=12)
    toric_kd, toric_kd_input, _ = simulate_code(
        BBclass=BBcode.BBcode_Toric,
        decoder_dict=bp_lsd(gaussian=True),
        error_model_dict=DEPOLARISING,
        n_trials=500,
        grids=6 * [{'L_x': 12, 'L_y': 12}],
        p_range=(0, 0.3, 60),
        ask_overwrite=False,
    )

    plotting.plot_gross_vs_toric(
        (gross, gross_input),
        (toric_k, toric_k_input),
        (toric_kd, toric_kd_input),
        savefig=True,
        filename_colored=os.path.join('figures', 'gross_vs_toric.pdf'),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Running all experiments. Figures will be saved to figures/.")
    print("Simulation data is cached in data/ and reused if present.\n")

    # Proof-of-concept experiments with small codes
    # (implemented separately in src/smaller_codes/)
    ...

    # PanQEC simulation results
    print("=== Gross code [[144,12,12]] vs toric code ===")
    run_gross_vs_toric()

    # T-gate implementation
    # (implemented separately in src/t-gate_implementation/)
    ...


if __name__ == "__main__":
    main()
