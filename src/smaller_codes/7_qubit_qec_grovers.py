"""
Steane [[7,1,3]] code: error correction and Grover's algorithm.

Source: notebooks/qiskit_grovers_test.ipynb (Qiskit 2.x)

Demonstrates:
  1. Encode → inject error → syndrome measurement → decode (error correction)
  2. Grover's algorithm with 2 logical qubits (14 physical qubits, 1 iteration)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer

try:
    from qiskit.visualization import plot_histogram
    _HAS_QISKIT_VIZ = True
except ImportError:
    _HAS_QISKIT_VIZ = False

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

simulator = Aer.get_backend('qasm_simulator')


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

def build_logical_grover_2qubit(target='11', n_iterations=1):
    """
    Grover's algorithm with 2 Steane-encoded logical qubits.

    Search space: N=4 logical basis states. Optimal iterations: ⌊π/4 √4⌋ = 1.

    Qubit layout:
        Block 0: qubits  0-6   (logical qubit 0, corresponds to target[0])
        Block 1: qubits  7-13  (logical qubit 1, corresponds to target[1])
        Ancilla: qubits 14-15  (parity readout ancilla, one per block)

    Logical Z readout: parity of physical qubits {0,1,2} (block 0) and {7,8,9}
    (block 1). For this encoding, Z_L = Z_0 Z_1 Z_2 is a valid weight-3
    representative: eigenvalue +1 on |0_L>, -1 on |1_L> = X^7|0_L>.

    Args:
        target:       2-char string, e.g. '11'. target[0] = logical qubit 0.
        n_iterations: number of Grover iterations.

    Returns:
        QuantumCircuit with 16 qubits (14 data + 2 ancilla) and 2 classical bits.
    """
    qr  = QuantumRegister(14, 'q')
    anc = QuantumRegister(2, 'anc')
    out = ClassicalRegister(2, 'out')
    qc  = QuantumCircuit(qr, anc, out)

    # 1. Encode both blocks as |0_L>
    _apply_encode(qc, block_offset=0, logical_state='0')
    _apply_encode(qc, block_offset=7, logical_state='0')
    qc.barrier()

    # 2. Superposition: H_L on both blocks
    _apply_H_L(qc, 0)
    _apply_H_L(qc, 7)
    qc.barrier()

    for _ in range(n_iterations):
        # 3. Oracle: phase-kick the target state
        # Flip '0'-bits so target looks like |11>, apply CZ_L, then unflip.
        if target[0] == '0': _apply_X_L(qc, 0)
        if target[1] == '0': _apply_X_L(qc, 7)
        _apply_CZ_L(qc, 0, 7)
        if target[0] == '0': _apply_X_L(qc, 0)
        if target[1] == '0': _apply_X_L(qc, 7)
        qc.barrier()

        # 4. Grover diffusion: D = H_L X_L CZ_L X_L H_L
        _apply_H_L(qc, 0);  _apply_H_L(qc, 7)
        _apply_X_L(qc, 0);  _apply_X_L(qc, 7)
        _apply_CZ_L(qc, 0, 7)
        _apply_X_L(qc, 0);  _apply_X_L(qc, 7)
        _apply_H_L(qc, 0);  _apply_H_L(qc, 7)
        qc.barrier()

    # 5. Logical Z measurement via parity of {0,1,2} (block 0) and {7,8,9} (block 1).
    # Qiskit count string is out[1]out[0], so assign block0→out[1], block1→out[0]
    # so the output string matches the target string directly.
    qc.cx(qr[0], anc[0]); qc.cx(qr[1], anc[0]); qc.cx(qr[2], anc[0])
    qc.measure(anc[0], out[1])   # block 0 parity → out[1] (left in string)

    qc.cx(qr[7], anc[1]); qc.cx(qr[8], anc[1]); qc.cx(qr[9], anc[1])
    qc.measure(anc[1], out[0])   # block 1 parity → out[0] (right in string)

    return qc


def run_logical_grover(target='11', shots=1024):
    """Run Grover's algorithm for the given target and return measurement counts."""
    qc = build_logical_grover_2qubit(target=target, n_iterations=1)
    result = simulator.run(transpile(qc, simulator), shots=shots).result()
    return result.get_counts()


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


def demo_grovers(shots=1024, save_figures=True):
    """Run Grover's algorithm for all 4 targets, print results, and save histogram."""
    print("=== Grover's algorithm: 2 logical qubits (Steane [[7,1,3]] code) ===\n")
    print(f"Search space: N=4  |  Optimal iterations: 1  |  Shots: {shots}\n")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    for ax, tgt in zip(axes, ['00', '01', '10', '11']):
        counts = run_logical_grover(target=tgt, shots=shots)
        top = max(counts, key=counts.get)
        top_prob = counts[top] / shots
        print(f"  Target |{tgt}_L>:  top='{top}'  (prob≈{top_prob:.2f})  "
              f"counts={dict(sorted(counts.items()))}")

        ax.bar(counts.keys(), counts.values(), color='steelblue')
        ax.set_title(f"Target $|{{{tgt}}}_L\\rangle$")
        ax.set_xlabel("Measurement")
        ax.set_ylabel("Counts")
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(
        "Grover's algorithm on Steane [[7,1,3]] code — 2 logical qubits, 1 iteration",
        fontsize=11)
    plt.tight_layout()

    if save_figures:
        import os
        fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, 'steane_grovers_histogram.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"\nHistogram saved to {os.path.abspath(path)}")
    plt.close()


if __name__ == '__main__':
    demo_error_correction()
    demo_grovers()
