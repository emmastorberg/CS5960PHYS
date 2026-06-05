"""
Steane [[7,1,3]] code: error correction and Grover's algorithm.

Source: notebooks/qiskit_grovers_test.ipynb (Qiskit 2.x)

Demonstrates:
  1. Encode → inject error → syndrome measurement → decode (error correction)
  2. Grover's algorithm with 2 logical qubits, with:
       - Depolarizing noise model (single- and two-qubit gate errors)
       - Mid-circuit syndrome measurement and feed-forward correction after
         each logical gate layer
       - Fault-tolerant logical Z readout using the full weight-7 Z_L operator
       - Side-by-side comparison of noiseless and noisy+EC runs

Simulation notes:
  - Uses AerSimulator with method='stabilizer', which exploits the fact that
    the circuit contains only Clifford gates and Pauli noise. This allows
    classical simulation in polynomial memory rather than the exponential
    memory required by a full statevector simulator.
  - T gates would break the stabilizer formalism and require statevector or
    tensor-network simulation. They do not appear here because the Grover
    oracle for a 2-element search space is implementable with a CZ gate alone.
    In a cryptographically relevant implementation the oracle would be the AES
    circuit, whose Toffoli gates decompose into T gates under Clifford+T.
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ── Steane code constants ─────────────────────────────────────────────────────

# Parity-check matrix H of the [7,4,3] Hamming code.
# Each row is the support of one stabilizer generator.
# Both X-type and Z-type stabilizers share the same support sets (CSS code).
#
#   qubit:         0  1  2  3  4  5  6
#   stabilizer 0:  1  0  1  0  1  0  1  → {0, 2, 4, 6}
#   stabilizer 1:  0  1  1  0  0  1  1  → {1, 2, 5, 6}
#   stabilizer 2:  0  0  0  1  1  1  1  → {3, 4, 5, 6}
STAB_SUPPORT = [
    [0, 2, 4, 6],
    [1, 2, 5, 6],
    [3, 4, 5, 6],
]

# Syndrome lookup: 3-bit integer → data qubit index (-1 = no error).
# Syndrome = Σ_i ancilla[i] * 2^i (ancilla[i] = 1 iff stabilizer i anticommuted).
# Each column of H read as a binary number uniquely identifies the qubit:
#   qubit 0 → 0b001, qubit 1 → 0b010, ..., qubit 6 → 0b111
SYNDROME_TO_QUBIT = {
    0b000: -1,
    0b001:  0,
    0b010:  1,
    0b011:  2,
    0b100:  3,
    0b101:  4,
    0b110:  5,
    0b111:  6,
}

simulator = AerSimulator(method="stabilizer")


# ── Noise model ───────────────────────────────────────────────────────────────

def build_noise_model(p1=0.001, p2=0.01, p_idle=None):
    """
    Depolarizing noise model.
      p1:     single-qubit gate error probability
      p2:     two-qubit gate error probability
      p_idle: idle qubit error probability per identity gate (default: p1)

    Depolarizing noise applies a random Pauli error (X, Y, or Z) after each gate
    with the given probability. Idle qubits receive the same noise via explicit
    identity gates inserted during syndrome measurement rounds.
    """
    if p_idle is None:
        p_idle = p1
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p1, 1), ['h', 'x', 'z', 'y', 's', 'sdg', 't', 'tdg']
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p2, 2), ['cx', 'cz']
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p_idle, 1), ['id']
    )
    return noise_model


# ── Error Correction ──────────────────────────────────────────────────────────

def build_encoding_circuit(logical_state='0'):
    """
    Steane [[7,1,3]] encoding circuit. Encodes logical qubit into 7 physical qubits.
    The logical state lives on qubit 6; all others start in |0>.
    """
    qc = QuantumCircuit(7, name='Encode')
    if logical_state == '1':
        qc.x(6)
    qc.h([0, 1, 3])
    # CNOT fan-out following columns of H
    qc.cx(0, 2); qc.cx(0, 4); qc.cx(0, 6)
    qc.cx(1, 2); qc.cx(1, 5); qc.cx(1, 6)
    qc.cx(3, 4); qc.cx(3, 5); qc.cx(3, 6)
    return qc


def build_decoding_circuit():
    """Inverse of encoding. Logical state is returned to qubit 6."""
    qc = QuantumCircuit(7, name='Decode')
    qc.cx(3, 6); qc.cx(3, 5); qc.cx(3, 4)
    qc.cx(1, 6); qc.cx(1, 5); qc.cx(1, 2)
    qc.cx(0, 6); qc.cx(0, 4); qc.cx(0, 2)
    qc.h([0, 1, 3])
    return qc


def build_syndrome_circuit():
    """
    Syndrome measurement circuit.

    Registers:
        d[0..6]   — 7 data qubits
        az[0..2]  — 3 ancilla for Z-type stabilizers (detect X/bit-flip errors)
        ax[0..2]  — 3 ancilla for X-type stabilizers (detect Z/phase-flip errors)
        sz[0..2]  — classical: Z-stabilizer syndrome
        sx[0..2]  — classical: X-stabilizer syndrome
    """
    data  = QuantumRegister(7, 'd')
    anc_z = QuantumRegister(3, 'az')
    anc_x = QuantumRegister(3, 'ax')
    syn_z = ClassicalRegister(3, 'sz')
    syn_x = ClassicalRegister(3, 'sx')
    qc = QuantumCircuit(data, anc_z, anc_x, syn_z, syn_x, name='Syndrome')

    # Z-stabilizer measurement (detect X errors): CNOT data → ancilla
    for i, support in enumerate(STAB_SUPPORT):
        for q in support:
            qc.cx(data[q], anc_z[i])
    qc.barrier()

    # X-stabilizer measurement (detect Z errors): H, CNOT ancilla → data, H
    for i, support in enumerate(STAB_SUPPORT):
        qc.h(anc_x[i])
        for q in support:
            qc.cx(anc_x[i], data[q])
        qc.h(anc_x[i])
    qc.barrier()

    qc.measure(anc_z, syn_z)
    qc.measure(anc_x, syn_x)
    return qc


def build_full_ec_circuit(error_qubit=None, error_type='X', logical_state='0'):
    """
    Full error-correction circuit: encode → (optional error) → syndrome measurement.

    Classical output string format: '{sx_bits} {sz_bits}'
    (Qiskit puts last-declared register first in the counts key)
    """
    data  = QuantumRegister(7, 'd')
    anc_z = QuantumRegister(3, 'az')
    anc_x = QuantumRegister(3, 'ax')
    syn_z = ClassicalRegister(3, 'sz')
    syn_x = ClassicalRegister(3, 'sx')
    qc = QuantumCircuit(data, anc_z, anc_x, syn_z, syn_x)

    qc.compose(build_encoding_circuit(logical_state), qubits=list(range(7)), inplace=True)
    qc.barrier()

    if error_qubit is not None:
        if error_type == 'X':   qc.x(data[error_qubit])
        elif error_type == 'Z': qc.z(data[error_qubit])
        elif error_type == 'Y': qc.y(data[error_qubit])
    qc.barrier()

    for i, support in enumerate(STAB_SUPPORT):
        for q in support:
            qc.cx(data[q], anc_z[i])
    qc.barrier()

    for i, support in enumerate(STAB_SUPPORT):
        qc.h(anc_x[i])
        for q in support:
            qc.cx(anc_x[i], data[q])
        qc.h(anc_x[i])
    qc.barrier()

    qc.measure(anc_z, syn_z)
    qc.measure(anc_x, syn_x)
    return qc


def parse_syndrome(counts_str):
    """
    Parse Qiskit counts key → (sz_val, sx_val).
    Qiskit string format: '{sx_bits} {sz_bits}' (last-declared register first).
    """
    parts = counts_str.split(' ')
    sx_str, sz_str = parts[0], parts[1]
    return int(sz_str, 2), int(sx_str, 2)


# ── Logical gates ─────────────────────────────────────────────────────────────

def logical_x_circuit():
    """Logical X: transversal X on all 7 qubits. Effect: |0_L> ↔ |1_L>."""
    qc = QuantumCircuit(7, name='X_L')
    qc.x(range(7))
    return qc


def logical_z_circuit():
    """Logical Z: transversal Z on all 7 qubits."""
    qc = QuantumCircuit(7, name='Z_L')
    qc.z(range(7))
    return qc


def logical_h_circuit():
    """
    Logical H: transversal H on all 7 qubits.
    Works because the Steane code is self-dual (H^7 maps X-stabs to Z-stabs).
    """
    qc = QuantumCircuit(7, name='H_L')
    qc.h(range(7))
    return qc


# ── Grover's algorithm ────────────────────────────────────────────────────────

def _apply_H_L(qc, block_offset):
    qc.h(range(block_offset, block_offset + 7))

def _apply_X_L(qc, block_offset):
    qc.x(range(block_offset, block_offset + 7))

def _apply_CZ_L(qc, offset1, offset2):
    """Transversal CZ between two logical blocks. Implements logical CZ."""
    for i in range(7):
        qc.cz(offset1 + i, offset2 + i)

def _apply_encode(qc, block_offset, logical_state='0'):
    qc.compose(build_encoding_circuit(logical_state),
               qubits=list(range(block_offset, block_offset + 7)), inplace=True)

def _measure_syndrome_once(qc, data_qubits, anc_z, anc_x, syn_z, syn_x):
    """One round of syndrome measurement into syn_z and syn_x.

    Explicit `id` gates are inserted on data qubits that are idle during each
    stabilizer's CNOT sequence so that idle-qubit decoherence fires on them.
    """
    qc.reset(anc_z)
    qc.reset(anc_x)
    all_data = set(range(7))
    for i, support in enumerate(STAB_SUPPORT):
        idle = all_data - set(support)
        for q in idle:
            qc.id(data_qubits[q])
        for q in support:
            qc.cx(data_qubits[q], anc_z[i])
    qc.barrier()
    for i, support in enumerate(STAB_SUPPORT):
        idle = all_data - set(support)
        for q in idle:
            qc.id(data_qubits[q])
        qc.h(anc_x[i])
        for q in support:
            qc.cx(anc_x[i], data_qubits[q])
        qc.h(anc_x[i])
    qc.barrier()
    qc.measure(anc_z, syn_z)
    qc.measure(anc_x, syn_x)


def _apply_mid_circuit_correction(qc, data_qubits,
                                  anc_z, anc_x,
                                  syn_z, syn_x,
                                  correct=True):
    """
    Single-round syndrome measurement with optional feed-forward correction.

    Always performs one full round of syndrome measurement (so gate overhead
    is identical regardless of the `correct` flag). When correct=True, applies
    Pauli corrections for any non-zero syndrome. When correct=False, the
    syndrome data are discarded — error rates reflect uncorrected noise on
    the same gate sequence.

    Args:
        data_qubits: list of 7 data qubit objects
        anc_z:       list of 3 ancilla qubits for Z-stabilizer measurement
        anc_x:       list of 3 ancilla qubits for X-stabilizer measurement
        syn_z:       ClassicalRegister(3) for Z syndrome
        syn_x:       ClassicalRegister(3) for X syndrome
        correct:     if True, apply feed-forward corrections
    """
    _measure_syndrome_once(qc, data_qubits, anc_z, anc_x, syn_z, syn_x)

    if correct:
        for syndrome_val, qubit_idx in SYNDROME_TO_QUBIT.items():
            if qubit_idx == -1:
                continue
            with qc.if_test((syn_z, syndrome_val)):
                qc.x(data_qubits[qubit_idx])
            with qc.if_test((syn_x, syndrome_val)):
                qc.z(data_qubits[qubit_idx])

    qc.barrier()


def _apply_logical_z_measurement(qc, data_qubits, anc_qubit, out_bit):
    """
    Fault-tolerant logical Z measurement for the Steane code.

    The logical Z operator Z_L = Z_0 Z_1 Z_2 Z_3 Z_4 Z_5 Z_6 is the tensor
    product of Z on all 7 physical qubits. Its eigenvalue (+1 or -1) determines
    whether the logical qubit is |0_L> or |1_L>.

    We measure this by computing the parity of all 7 data qubits into a single
    ancilla qubit via CNOT gates, then measuring the ancilla. This uses the full
    weight-7 Z_L representative rather than a weight-3 shortcut, making the
    readout robust to single physical qubit errors.

    Args:
        qc:          QuantumCircuit
        data_qubits: list of 7 qubit indices for the data block
        anc_qubit:   single ancilla qubit index for readout
        out_bit:     single classical bit to store the result
    """
    qc.reset(anc_qubit)
    for q in data_qubits:
        qc.cx(q, anc_qubit)
    qc.measure(anc_qubit, out_bit)


def build_logical_grover_2qubit(target='11', n_iterations=1, ec=True):
    """
    Grover's algorithm with 2 Steane-encoded logical qubits.

    Args:
        target:       2-char string, e.g. '11'. target[0] = logical qubit 0.
        n_iterations: number of Grover iterations.
        ec:           if True, apply two-round syndrome measurement with
                      agreement filtering after each logical gate layer.
                      If False, noise accumulates uncorrected (no-EC baseline).

    Qubit layout:
        Block 0:  qubits  0-6   (logical qubit 0)
        Block 1:  qubits  7-13  (logical qubit 1)
        anc_z0/1: 3 qubits each — Z-stabilizer ancilla per block
        anc_x0/1: 3 qubits each — X-stabilizer ancilla per block
        readout:  2 qubits
        Total: 28 qubits

    Classical registers:
        sz0, sx0: syndrome registers for block 0
        sz1, sx1: syndrome registers for block 1
        out: 2-bit measurement result
    """
    qr      = QuantumRegister(14, 'q')
    anc_z0  = QuantumRegister(3,  'az0')
    anc_x0  = QuantumRegister(3,  'ax0')
    anc_z1  = QuantumRegister(3,  'az1')
    anc_x1  = QuantumRegister(3,  'ax1')
    readout = QuantumRegister(2,  'ro')

    sz0 = ClassicalRegister(3, 'sz0')
    sx0 = ClassicalRegister(3, 'sx0')
    sz1 = ClassicalRegister(3, 'sz1')
    sx1 = ClassicalRegister(3, 'sx1')
    out = ClassicalRegister(2, 'out')

    qc = QuantumCircuit(qr, anc_z0, anc_x0, anc_z1, anc_x1, readout,
                        sz0, sx0, sz1, sx1, out)

    data0 = list(qr[:7])
    data1 = list(qr[7:])

    def correct_both():
        _apply_mid_circuit_correction(
            qc, data0, list(anc_z0), list(anc_x0),
            sz0, sx0, correct=ec)
        _apply_mid_circuit_correction(
            qc, data1, list(anc_z1), list(anc_x1),
            sz1, sx1, correct=ec)

    # 1. Encode both blocks as |0_L>
    _apply_encode(qc, block_offset=0, logical_state='0')
    _apply_encode(qc, block_offset=7, logical_state='0')
    qc.barrier()

    # 2. Superposition: H_L on both blocks, then syndrome (always)
    _apply_H_L(qc, 0)
    _apply_H_L(qc, 7)
    qc.barrier()
    correct_both()

    for _ in range(n_iterations):
        # 3. Oracle: phase-kick the target state, then syndrome
        if target[0] == '0': _apply_X_L(qc, 0)
        if target[1] == '0': _apply_X_L(qc, 7)
        _apply_CZ_L(qc, 0, 7)
        if target[0] == '0': _apply_X_L(qc, 0)
        if target[1] == '0': _apply_X_L(qc, 7)
        qc.barrier()
        correct_both()

        # 4. Grover diffusion: D = H_L X_L CZ_L X_L H_L, then syndrome
        _apply_H_L(qc, 0);  _apply_H_L(qc, 7)
        _apply_X_L(qc, 0);  _apply_X_L(qc, 7)
        _apply_CZ_L(qc, 0, 7)
        _apply_X_L(qc, 0);  _apply_X_L(qc, 7)
        _apply_H_L(qc, 0);  _apply_H_L(qc, 7)
        qc.barrier()
        correct_both()

    # 5. Fault-tolerant logical Z measurement: parity of all 7 qubits per block.
    # Qiskit count string is out[1]out[0], so block0→out[1], block1→out[0]
    # so the output string matches the target string directly.
    _apply_logical_z_measurement(qc, data0, readout[0], out[1])
    _apply_logical_z_measurement(qc, data1, readout[1], out[0])

    return qc


def run_logical_grover(target='11', shots=1024, noise_model=None, ec=True):
    """Run Grover's algorithm for the given target and return measurement counts.

    Uses result.get_counts(qc) and filters to only the 'out' classical register
    by passing the circuit object, then marginalises over all other registers
    using Qiskit's marginal_counts utility.
    """
    from qiskit.result import marginal_counts

    qc = build_logical_grover_2qubit(target=target, n_iterations=1, ec=ec)
    backend = AerSimulator(method='stabilizer', noise_model=noise_model,
                           seed_simulator=42) \
              if noise_model else simulator
    job    = backend.run(transpile(qc, backend), shots=shots, seed_simulator=42)
    result = job.result()

    # 'out' is declared last → leftmost 2 bits → indices 12 and 13
    # (total classical bits: sz0=3, sx0=3, sz1=3, sx1=3, out=2 → 14 bits)
    raw = result.get_counts()
    marginalised = marginal_counts(raw, indices=[12, 13])
    return dict(marginalised)


# ── Main ──────────────────────────────────────────────────────────────────────

def demo_error_correction():
    """Verify syndrome detection for all 7 single-qubit X errors."""
    print("=== Steane [[7,1,3]] error correction demo ===\n")
    print("Injected error  →  sz syndrome  →  decoded qubit")
    print("-" * 48)
    all_ok = True
    for err_q in range(7):
        qc = build_full_ec_circuit(error_qubit=err_q, error_type='X')
        result = simulator.run(transpile(qc, simulator), shots=256).result()
        syndrome_str, _ = max(result.get_counts().items(), key=lambda x: x[1])
        sz_val, _ = parse_syndrome(syndrome_str)
        decoded = SYNDROME_TO_QUBIT.get(sz_val, '?')
        ok = decoded == err_q
        if not ok:
            all_ok = False
        print(f"  X on qubit {err_q}  →  sz={sz_val:03b} ({sz_val})  →  qubit {decoded}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\nAll tests passed.\n" if all_ok else "\nSome tests FAILED.\n")


def demo_grovers(shots=1024, p1=0.001, p2=0.01, p_idle=1e-5, save_figures=True):
    """
    Run Grover's algorithm for all 4 targets under depolarizing noise, with
    and without mid-circuit error correction, and save a grouped histogram.

    Args:
        shots:        number of simulation shots per target per condition
        p1:           single-qubit depolarizing error probability
        p2:           two-qubit depolarizing error probability
        p_idle:       idle qubit depolarizing error probability (default: p1/100,
                      reflecting that T1/T2 times are ~100x longer than gate times
                      in typical superconducting hardware)
        save_figures: whether to save the histogram to disk
    """
    print("=== Grover's algorithm: 2 logical qubits (Steane [[7,1,3]] code) ===\n")
    print(f"Search space: N=4  |  Optimal iterations: 1  |  Shots: {shots}")
    print(f"Noise model: p1={p1} (single-qubit), p2={p2} (two-qubit), p_idle={p_idle} (idle)\n")

    noise_model  = build_noise_model(p1=p1, p2=p2, p_idle=p_idle)
    targets      = ['00', '01', '10', '11']
    all_outcomes = ['00', '01', '10', '11']

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
    })

    # ColorBrewer RdBu: blue for EC, red for no-EC; dark=hit, light=miss
    EC_HIT    = "#2166AC"   # deep blue
    EC_MISS   = "#92C5DE"   # light blue
    NOEC_HIT  = "#D6604D"   # deep red
    NOEC_MISS = "#F4A582"   # light red

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()
    fig.suptitle(
        r"Grover's algorithm on Steane $[\![7,1,3]\!]$ code" + "\n"
        r"2 logical qubits, 1 iteration, depolarizing noise" + "\n"
        rf"$p_1={p1}$,\ $p_2={p2}$,\ $p_{{\mathrm{{idle}}}}={p_idle}$",
        fontsize=16, x=0.5, ha='center')

    print(f"{'Target':<10} {'EC prob':<12} {'No-EC prob':<12}")
    print("-" * 34)

    x = np.arange(len(all_outcomes))
    width = 0.38

    all_counts_ec   = {}
    all_counts_noec = {}
    for tgt in targets:
        with tqdm(total=2, desc=f"Target |{tgt}_L>", unit="run", leave=False) as pbar:
            all_counts_ec[tgt] = run_logical_grover(
                target=tgt, shots=shots, noise_model=noise_model, ec=True)
            pbar.update(1)
            all_counts_noec[tgt] = run_logical_grover(
                target=tgt, shots=shots, noise_model=noise_model, ec=False)
            pbar.update(1)

    for idx, tgt in enumerate(targets):
        counts_ec   = all_counts_ec[tgt]
        counts_noec = all_counts_noec[tgt]

        prob_ec   = counts_ec.get(tgt, 0) / shots
        prob_noec = counts_noec.get(tgt, 0) / shots
        print(f"|{tgt}_L>  EC prob={prob_ec:.2f}  no-EC prob={prob_noec:.2f}")
        print(f"  EC counts:    { {k: counts_ec.get(k,0)   for k in all_outcomes} }")
        print(f"  no-EC counts: { {k: counts_noec.get(k,0) for k in all_outcomes} }")

        vals_ec   = [counts_ec.get(k, 0)   / shots for k in all_outcomes]
        vals_noec = [counts_noec.get(k, 0) / shots for k in all_outcomes]

        colors_ec   = [EC_HIT   if k == tgt else EC_MISS   for k in all_outcomes]
        colors_noec = [NOEC_HIT if k == tgt else NOEC_MISS for k in all_outcomes]

        ax = axes[idx]
        bars_ec   = ax.bar(x - width/2, vals_ec,   width, color=colors_ec,   label='With EC')
        bars_noec = ax.bar(x + width/2, vals_noec, width, color=colors_noec, label='No EC')

        def label_bars(bars, is_target_list):
            for bar, is_target in zip(bars, is_target_list):
                h = bar.get_height()
                color = 'white' if is_target else 'black'
                ax.text(bar.get_x() + bar.get_width() / 2, h - 0.015,
                        f'{h:.2f}', ha='center', va='top', fontsize=11, color=color)

        label_bars(bars_ec,   [k == tgt for k in all_outcomes])
        label_bars(bars_noec, [k == tgt for k in all_outcomes])
        ax.set_title(rf"Target $|\,{tgt}_L\rangle$", fontsize=16)
        ax.set_xlabel("Measurement outcome", fontsize=14)
        ax.set_ylabel("Probability", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(all_outcomes)
        ax.tick_params(axis='both', labelsize=13)
        ax.yaxis.set_tick_params(labelleft=True)
        if idx == 0:
            ax.legend(fontsize=12)

    print()
    plt.tight_layout()

    if save_figures:
        fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, 'steane_grovers_histogram.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Histogram saved to {os.path.abspath(path)}")
    plt.close()


if __name__ == '__main__':
    demo_error_correction()
    demo_grovers()