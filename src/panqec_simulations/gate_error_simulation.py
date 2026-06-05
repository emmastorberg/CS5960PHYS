"""
Gate error injection experiment: transversal X_L gate vs quantum memory.

Section 3.2.3 — Simulating Transversal Logical Operators with PanQEC.

When a transversal X_L gate is applied to a code block, the gate touches
a specific subset of physical qubits — the gate support. In a realistic
device, each of those qubits can acquire an additional X error during the
gate operation, with some probability p_gate. This differs fundamentally
from quantum memory noise, where X errors are spread uniformly over all
n physical qubits.

This experiment asks: at the same per-qubit error rate p, is it harder
or easier for the decoder to correct gate-concentrated errors versus
memory-spread errors? And how do the BB code and toric surface code compare?

Gate support is the minimum-weight representative of the X_L logical class,
taken from the first row of the bposd logical operator matrix (lx) with the
smallest Hamming weight. This is the most efficient physical realisation of
the transversal X_L gate.

Codes compared at equal n = 2·Lx·Ly physical qubits:
  BBcode_ArXiV_example — BB code [[n, k_BB, d_BB]]
  BBcode_Toric         — toric surface code [[n, 2, Lx]]

Figures produced:
  figures/gate_support_visualization.pdf  — lattice diagram of X_L gate support
  figures/logical_operator_weights.pdf    — weight of each X_L over all logical qubits
  figures/gate_error_vs_memory.pdf        — p_L vs p: gate errors vs memory noise
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import issparse

sys.path.insert(0, os.path.dirname(__file__))
from BBcode_classes import BBcode_ArXiV_example, BBcode_Toric


# ── Utilities ─────────────────────────────────────────────────────────────────

def to_dense(M):
    """Convert sparse or dense matrix to a dense numpy int array."""
    if issparse(M):
        return np.asarray(M.todense(), dtype=int)
    return np.asarray(M, dtype=int)


def all_logical_weights(lx_matrix):
    """Return array of X_L Hamming weights for each logical qubit."""
    lx = to_dense(lx_matrix)
    return lx.sum(axis=1)


def min_weight_support(lx_matrix):
    """
    Return the physical qubit indices of the minimum-weight X_L operator.

    Among all rows of lx (one per logical qubit), picks the row with the
    smallest Hamming weight. This gives the most qubit-efficient gate.
    """
    lx = to_dense(lx_matrix)
    weights = lx.sum(axis=1)
    row = int(np.argmin(weights))
    return np.flatnonzero(lx[row])


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate_x_failures(HZ, lz, active_qubits, p_values, n_trials,
                        max_bp_iter=100, lsd_order=5, seed=42):
    """
    Sweep X error rates over `active_qubits` and measure logical failure.

    For each p in p_values: sample independent X errors on active_qubits
    with probability p, compute the Z-check syndrome, decode with BP-LSD,
    and check whether any residual error anticommutes with a Z-type logical.

    Parameters
    ----------
    HZ            : (m, n) int array  — Z-type parity check matrix
    lz            : (k, n) int array  — Z-type logical operator matrix
    active_qubits : 1-D int array     — qubits that can have X errors
    p_values      : 1-D float array   — error rates to sweep
    n_trials      : int               — Monte Carlo trials per rate

    Returns
    -------
    failure_probs : 1-D float array, shape (len(p_values),)
    """
    HZ = to_dense(HZ)
    lz = to_dense(lz)
    n = HZ.shape[1]
    w = len(active_qubits)

    rng = np.random.default_rng(seed)
    failure_probs = np.zeros(len(p_values))

    for ip, p in enumerate(tqdm(p_values, desc='   sweep p', leave=False)):
        # Decoder is told uniform error rate p — it does not know errors are
        # concentrated on active_qubits. This is the conservative (realistic)
        # scenario where the decoder has no knowledge of the gate structure.
        decoder = BpLsdDecoder(
            HZ, error_rate=float(p),
            max_iter=max_bp_iter,
            bp_method='minimum_sum',
            ms_scaling_factor=0.,
            schedule='serial',
            lsd_method='lsd_0',
            lsd_order=lsd_order,
        )
        n_fail = 0
        for _ in range(n_trials):
            # Sample X errors on active qubits only
            x_err = np.zeros(n, dtype=int)
            x_err[active_qubits] = (rng.random(w) < p).astype(int)

            # Z-syndrome: Z-checks detect X errors
            sz = HZ @ x_err % 2

            # Decode and apply correction
            x_corr = decoder.decode(sz)
            eff = (x_err + x_corr) % 2

            # Logical failure: residual error anticommutes with any Z logical
            if np.any(lz @ eff % 2):
                n_fail += 1

        failure_probs[ip] = n_fail / n_trials

    return failure_probs


# ── Visualisations ────────────────────────────────────────────────────────────

def _fig_dir():
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
    os.makedirs(path, exist_ok=True)
    return path


def fig_gate_support(bb, sc, bb_support, sc_support, bb_d='?', save=True):
    """
    Figure 1: lattice diagram showing which physical qubits X_L acts on.
    Left panel = BB code, right panel = toric surface code.

    Parameters
    ----------
    bb_d : int or str
        Known code distance for the BB code, or '?' if unknown.
    """
    plt.style.use('seaborn-v0_8-white')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rcParams.update({
        'axes.labelsize': 18, 'axes.titlesize': 18,
        'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'legend.fontsize': 16, 'font.size': 16,
        'figure.titlesize': 19,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={'wspace': 0.02})
    n = bb.HX.shape[1]

    GATE_COLOR_BB = '#E69F00'   # Okabe-Ito golden yellow (matches gross/BB code in fig 3.3)
    GATE_COLOR_SC = '#0072B2'   # Okabe-Ito blue (matches toric code in fig 3.3)
    IDLE_COLOR    = '#d5d8dc'

    lx, ly = bb.size
    sc_d = lx  # toric code distance equals Lx
    configs = [
        (bb, bb_support, GATE_COLOR_BB,
         f'BB code  $[[{n},\\,{bb.num_logical_qubits},\\,{bb_d}]]$'),
        (sc, sc_support, GATE_COLOR_SC,
         f'Toric code  $[[{n},\\,{sc.num_logical_qubits},\\,{sc_d}]]$'),
    ]
    legend_handles = []
    for ax, (code, support, color, title) in zip(axes, configs):
        coords = code.get_qubit_coordinates()
        gate_set = set(support.tolist())
        rest = [i for i in range(len(coords)) if i not in gate_set]

        h_idle = ax.scatter(
            [coords[i][0] for i in rest],
            [coords[i][1] for i in rest],
            c=IDLE_COLOR, s=30, zorder=2,
            label=f'idle  ({len(rest)} qubits)',
        )
        h_gate = ax.scatter(
            [coords[i][0] for i in support],
            [coords[i][1] for i in support],
            c=color, s=130, zorder=3, marker='*',
            label=f'$\\bar{{X}}$ support  ($w = {len(support)}$)',
        )
        legend_handles.append((h_idle, h_gate))
        ax.set_title(title, fontsize=18, pad=10)
        ax.set_aspect('equal')
        ax.axis('off')

    # Per-panel legends placed outside the lattice, below each axis
    for ax, (h_idle, h_gate) in zip(axes, legend_handles):
        ax.legend(
            handles=[h_idle, h_gate],
            fontsize=16,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.04),
            ncol=2,
            frameon=True,
            borderpad=0.5,
            handletextpad=0.4,
            columnspacing=1.0,
        )

    fig.suptitle(
        f'Physical qubits in the transversal $\\bar{{X}}$ gate support'
        f'\\quad ($n = {n}$,  $L_x = {lx}$,  $L_y = {ly}$)',
        fontsize=19, y=1.02,
    )
    plt.tight_layout()
    if save:
        path = os.path.join(_fig_dir(), 'gate_support_visualization.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved: {os.path.abspath(path)}')
    plt.close(fig)


def fig_logical_weights(bb, sc, save=True):
    """
    Figure 2: weight of every X_L logical operator for each code.
    One dot per logical qubit, jittered for readability.
    """
    bb_weights = all_logical_weights(bb.lx)
    sc_weights = all_logical_weights(sc.lx)

    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(0)

    for xi, (weights, color, label) in enumerate([
        (bb_weights, '#e07b39', f'BB code  (k = {bb.num_logical_qubits})'),
        (sc_weights, '#2980b9', f'Toric code  (k = {sc.num_logical_qubits})'),
    ]):
        jitter = rng.uniform(-0.15, 0.15, size=len(weights))
        ax.scatter(
            np.full(len(weights), xi) + jitter,
            weights,
            color=color, alpha=0.8, s=60, label=label,
        )
        ax.hlines(
            np.mean(weights), xi - 0.3, xi + 0.3,
            color=color, linewidth=2, linestyle='--',
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f'BB code\n(k = {bb.num_logical_qubits})',
         f'Toric code\n(k = {sc.num_logical_qubits})'],
        fontsize=11,
    )
    ax.set_ylabel('$\\bar{X}$ operator weight', fontsize=11)
    ax.set_title('Weight of each logical $\\bar{X}$ operator\n(dashed line = mean)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save:
        path = os.path.join(_fig_dir(), 'logical_operator_weights.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved: {os.path.abspath(path)}')
    plt.close(fig)


def _power_law_extrapolation(p_values, pL, p_extrap, min_points=4):
    """
    Fit log(pL) = log(C) + t*log(p) to observed (non-zero) data and evaluate
    at p_extrap. Returns (p_extrap_filtered, pL_extrap) or (None, None) if
    there are fewer than min_points real observations.
    """
    real = pL > 0
    if real.sum() < min_points:
        return None, None
    log_p = np.log(p_values[real])
    log_pL = np.log(pL[real])
    t, log_C = np.polyfit(log_p, log_pL, 1)
    # Only extrapolate to p values below the first real observation
    p_first_real = p_values[real][0]
    p_ex = p_extrap[p_extrap < p_first_real]
    if len(p_ex) == 0:
        return None, None
    pL_ex = np.exp(log_C) * p_ex ** t
    return p_ex, pL_ex


def fig_gate_vs_memory(p_values, results, bb, sc, bb_d='?', save=True):
    """
    Figure 3: logical X failure probability vs gate error rate p.

    Compares the two codes under gate errors only. The key question:
    for a transversal X_L gate with per-qubit error rate p, which code
    has lower logical failure probability? The BB code has a larger gate
    support (w_BB > w_SC) but higher distance, so there is a crossover:
    BB wins at low p (realistic regime), SC wins at high p.

    Parameters
    ----------
    bb_d : int or str
        Known code distance for the BB code (from the literature), or '?'
        if unknown.
    """
    plt.style.use('seaborn-v0_8-white')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rcParams.update({
        'axes.labelsize': 17, 'axes.titlesize': 17,
        'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'legend.fontsize': 14, 'font.size': 14,
        'figure.titlesize': 18,
        'axes.grid': True, 'grid.linewidth': 0.5, 'grid.alpha': 0.5,
    })

    n = bb.HX.shape[1]
    bb_w = results['bb_gate_w']
    sc_w = results['sc_gate_w']
    sc_d = bb.size[0]   # toric code distance = Lx

    fig, ax = plt.subplots(figsize=(8, 5))

    n_trials = results.get('n_trials', 2000)
    laplace_floor = 0.5 / n_trials   # lower bound when zero failures observed

    bb_real = results['bb_gate'] > 0
    sc_real = results['sc_gate'] > 0

    ax.plot(p_values[bb_real], results['bb_gate'][bb_real], 'o-',
            color='#E69F00', lw=2,
            label=f'BB code $[[{n},\\,{bb.num_logical_qubits},\\,{bb_d}]]$'
                  f'  ($w = {bb_w}$)')

    ax.plot(p_values[sc_real], results['sc_gate'][sc_real], 's-',
            color='#0072B2', lw=2,
            label=f'Toric code $[[{n},\\,{sc.num_logical_qubits},\\,{sc_d}]]$'
                  f'  ($w = {sc_w}$)')

    # Power-law extrapolation into the unresolved low-p regime
    p_dense = np.geomspace(p_values[0], p_values[-1], 300)
    for pL_data, color in [
        (results['bb_gate'], '#E69F00'),
        (results['sc_gate'], '#0072B2'),
    ]:
        p_ex, pL_ex = _power_law_extrapolation(p_values, pL_data, p_dense)
        if p_ex is not None:
            ax.plot(p_ex, pL_ex, '--', color=color, lw=1.5, alpha=0.6)

    # Find and annotate the crossover point.
    # Only consider points where at least one curve is above a noise floor,
    # to avoid spurious sign-changes from Monte Carlo noise at zero failures.
    noise_floor = 1.5 / results.get('n_trials', 2000)
    meaningful = (results['bb_gate'] > noise_floor) | (results['sc_gate'] > noise_floor)
    if meaningful.any():
        diff = results['bb_gate'] - results['sc_gate']
        diff_m = diff[meaningful]
        p_m = p_values[meaningful]
        sign_changes = np.where(np.diff(np.sign(diff_m)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            p0, p1 = p_m[idx], p_m[idx + 1]
            d0, d1 = diff_m[idx], diff_m[idx + 1]
            p_cross = p0 - d0 * (p1 - p0) / (d1 - d0)
            ax.axvline(p_cross, color='gray', linestyle=':', lw=1.5)
            ax.text(p_cross + 0.004, 0.3,
                    f'crossover\n$p \\approx {p_cross:.2f}$',
                    fontsize=13, color='gray', va='center',
                    transform=ax.get_xaxis_transform())

    # Shade the "realistic" gate quality region
    p_realistic = 0.01
    ax.axvspan(0, p_realistic, alpha=0.08, color='green')
    ax.text(0.0005, 0.92,
            'realistic\ngate quality',
            fontsize=12, color='green', va='top',
            transform=ax.get_xaxis_transform())

    # Dummy handle so the legend explains the dashed extrapolation lines
    ax.plot([], [], 'k--', lw=1.5, alpha=0.6, label='power-law extrapolation')

    ax.set_xlabel('Per-qubit gate error rate $p$')
    ax.set_ylabel('Logical $\\bar{X}$ failure probability')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-8)
    ax.set_xlim(left=0, right=p_values[-1])
    ax.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(_fig_dir(), 'gate_error_vs_memory.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved: {os.path.abspath(path)}')
    plt.close(fig)


# ── Results cache (save / load) ───────────────────────────────────────────────

def _results_path(lx_size, ly_size, n_trials, p_range):
    """Canonical .npz path for a given set of run parameters."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    p_min, p_max, n_pts = p_range
    fname = (f'gate_error_lx{lx_size}_ly{ly_size}'
             f'_trials{n_trials}_p{p_min:.4f}-{p_max:.4f}-{n_pts}.npz')
    return os.path.join(results_dir, fname)


def save_results(path, p_values, results):
    np.savez(path, p_values=p_values, **results)
    print(f'  Results saved: {os.path.abspath(path)}')


def load_results(path):
    data = np.load(path)
    p_values = data['p_values']
    results = {k: data[k] for k in data.files if k != 'p_values'}
    print(f'  Results loaded: {os.path.abspath(path)}')
    return p_values, results


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_experiment(
    lx_size: int = 7,
    ly_size: int = 7,
    p_range: tuple = (0.01, 0.25, 30),
    n_trials: int = 2000,
    save_figures: bool = True,
    plot_only: bool = False,
):
    """
    Run the full gate error injection experiment and produce all three figures.

    Parameters
    ----------
    lx_size, ly_size : int
        Code grid dimensions. Both codes use the same (Lx, Ly) so n is equal.
    p_range : (p_min, p_max, n_points)
        Physical error rate sweep.
    n_trials : int
        Monte Carlo trials per error rate per condition.
    save_figures : bool
        Write PDFs to figures/.
    plot_only : bool
        If True, skip Monte Carlo and load results from the cached .npz file.
        Raises FileNotFoundError if no cache exists for these parameters.
    """
    # ── Instantiate codes ──────────────────────────────────────────────────
    print(f'Instantiating codes  (Lx={lx_size}, Ly={ly_size})...')
    bb = BBcode_ArXiV_example(lx_size, ly_size)
    sc = BBcode_Toric(lx_size, ly_size)
    n = bb.HX.shape[1]
    assert sc.HX.shape[1] == n, 'Both codes must have the same n'

    print(f'\n  BB code:     n={n},  k={bb.num_logical_qubits}')
    print(f'  Toric code:  n={n},  k={sc.num_logical_qubits}')

    bb_weights = all_logical_weights(bb.lx)
    sc_weights = all_logical_weights(sc.lx)
    print(f'\n  BB   X_L weights: {sorted(bb_weights.tolist())}')
    print(f'  Surf X_L weights: {sorted(sc_weights.tolist())}')

    bb_support = min_weight_support(bb.lx)
    sc_support = min_weight_support(sc.lx)
    print(f'\n  BB   minimum-weight X_L: w = {len(bb_support)}')
    print(f'  Surf minimum-weight X_L: w = {len(sc_support)}')

    # ── Figures 1 & 2: structural visualisations (no simulation needed) ────
    BB_KNOWN_D = {7: 12, 9: 12}
    bb_d = BB_KNOWN_D.get(lx_size, '?')
    print('\nGenerating structural figures...')
    fig_gate_support(bb, sc, bb_support, sc_support, bb_d=bb_d, save=save_figures)
    fig_logical_weights(bb, sc, save=save_figures)

    # ── Simulation or cache load ───────────────────────────────────────────
    cache_path = _results_path(lx_size, ly_size, n_trials, p_range)

    if plot_only:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f'No cached results found at {cache_path}.\n'
                f'Run without --plot-only first to generate them.'
            )
        print('\nLoading cached results (skipping Monte Carlo)...')
        p_values, results = load_results(cache_path)
    else:
        p_min, p_max, n_pts = p_range
        p_values = np.linspace(p_min, p_max, n_pts)
        all_q = np.arange(n)

        HZ_bb = to_dense(bb.HZ);  lz_bb = to_dense(bb.lz)
        HZ_sc = to_dense(sc.HZ);  lz_sc = to_dense(sc.lz)

        print(f'\nRunning Monte Carlo  (n_trials={n_trials}, '
              f'p ∈ [{p_min:.4f}, {p_max:.4f}], {n_pts} points)...')

        print('  BB code — gate errors:')
        bb_gate = simulate_x_failures(HZ_bb, lz_bb, bb_support, p_values, n_trials, seed=10)
        print('  BB code — memory (X only):')
        bb_mem  = simulate_x_failures(HZ_bb, lz_bb, all_q,      p_values, n_trials, seed=11)
        print('  Toric code — gate errors:')
        sc_gate = simulate_x_failures(HZ_sc, lz_sc, sc_support, p_values, n_trials, seed=12)
        print('  Toric code — memory (X only):')
        sc_mem  = simulate_x_failures(HZ_sc, lz_sc, all_q,      p_values, n_trials, seed=13)

        results = dict(
            bb_gate=bb_gate, bb_mem=bb_mem,
            sc_gate=sc_gate, sc_mem=sc_mem,
            bb_gate_w=np.int64(len(bb_support)),
            sc_gate_w=np.int64(len(sc_support)),
            n_trials=np.int64(n_trials),
        )
        save_results(cache_path, p_values, results)

    # ── Figure 3: main result ──────────────────────────────────────────────
    print('\nGenerating Figure 3...')
    fig_gate_vs_memory(p_values, results, bb, sc, bb_d=bb_d, save=save_figures)

    # ── Summary table ──────────────────────────────────────────────────────
    bb_gate = results['bb_gate']
    sc_gate = results['sc_gate']
    bb_mem  = results['bb_mem']
    sc_mem  = results['sc_mem']
    n_pts   = len(p_values)
    print(f'\n{"p":>6}  {"BB gate":>9}  {"BB mem":>9}  '
          f'{"SC gate":>9}  {"SC mem":>9}')
    print('─' * 52)
    step = max(1, n_pts // 10)
    for i in range(0, n_pts, step):
        print(f'{p_values[i]:6.4f}  {bb_gate[i]:9.4f}  {bb_mem[i]:9.4f}  '
              f'{sc_gate[i]:9.4f}  {sc_mem[i]:9.4f}')

    return p_values, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip Monte Carlo; load cached results and re-plot.')
    args = parser.parse_args()

    run_experiment(
        lx_size=7,
        ly_size=7,
        p_range=(0.0005, 0.10, 60),
        n_trials=1000000,
        plot_only=args.plot_only,
    )
