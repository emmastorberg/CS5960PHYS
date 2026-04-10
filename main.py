"""
main.py — Run all experiments and generate all figures for the thesis.

Usage (from repo root):
    python main.py

Figures are written to figures/.
Simulation data is cached to data/ and reused on subsequent runs.
Delete data/ to force a full re-run.
"""

import sys
import os

# Make src/ importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import panqec_simulations.BBcode_classes as BBcode
from panqec_simulations.memory_simulation import simulate_code
from panqec_simulations import plotting


# ── Shared simulation parameters ──────────────────────────────────────────────

DEPOLARISING_ERROR_MODEL = {
    'name': 'GaussianPauliErrorModel',
    'parameters': [{'r_x': 1/3, 'r_y': 1/3, 'r_z': 1/3}],
}

DECODER = {
    'name': 'BeliefPropagationLSDDecoder',
    'parameters': [{'max_bp_iter': 1000, 'lsd_order': 10,
                    'channel_update': False, 'bp_method': 'minimum_sum'}],
}


# ── Experiment 1: Quantum memory threshold ────────────────────────────────────

def run_memory_threshold():
    """Simulate and plot the logical error rate vs physical error rate threshold
    for the BB code under depolarising noise."""

    analysis, input_data, _ = simulate_code(
        BBclass=BBcode.BBcode_Toric,
        error_model_dict=DEPOLARISING_ERROR_MODEL,
        decoder_dict=DECODER,
        n_trials=100,
        grids=[{'L_x': 6, 'L_y': 6}, {'L_x': 8, 'L_y': 8}, {'L_x': 10, 'L_y': 10}],
        p_range=(0.05, 0.20, 20),
        ask_overwrite=False,
    )

    plotting.plot_error_rates(
        analysis,
        savefig=True,
        filename=os.path.join('figures', 'memory_threshold.pdf'),
    )

    return analysis, input_data


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Running all experiments. Figures will be saved to figures/.")
    print("Simulation data is cached in data/ and reused if present.\n")

    # Proof-of-concept experiments with small codes
    # (implemented separately in src/smaller_codes/)
    ...

    # PanQEC simulation results
    print("=== Experiment 1: Quantum memory threshold ===")
    run_memory_threshold()

    # T-gate implementation
    # (implemented separately in src/t-gate_implementation/)
    ...


if __name__ == "__main__":
    main()
