import sys
import os
import numpy as np
import jax.numpy as jnp
from functools import reduce

sys.path.insert(0, os.path.dirname(__file__))
from gates import pauli_x_gate, pauli_y_gate, pauli_z_gate, identity_gate, hadamard_gate

"""
Source: https://www.physics.unlv.edu/~bernard/MATH_book/Chap9/Notebook9_3.pdf

Demonstrates error correction on the [[5,1,3]] code using a simplified
inject-and-correct pattern: prepare logical codeword → inject single Pauli
error → measure syndrome → look up correction → apply → verify.
"""

tol = 1e-12

# Syndrome lookup: syndrome integer → (Pauli label, ops-list index)
# ops-list index 0 = leftmost in tensor product = actual data qubit 4
# ops-list index 4 = rightmost = actual data qubit 0
SYNDROME_TO_ERROR = {
    9:  ("X", 0),
    15: ("Y", 0),
    6:  ("Z", 0),
    4:  ("X", 1),
    7:  ("Y", 1),
    3:  ("Z", 1),
    10: ("X", 2),
    11: ("Y", 2),
    1:  ("Z", 2),
    5:  ("X", 3),
    13: ("Y", 3),
    8:  ("Z", 3),
    2:  ("X", 4),
    14: ("Y", 4),
    12: ("Z", 4),
}

ket0 = jnp.array([1, 0])
ket1 = jnp.array([0, 1])

X = pauli_x_gate()
Y = pauli_y_gate()
Z = pauli_z_gate()
I = identity_gate()
H = hadamard_gate()


def _setup():
    """Build stabilizers, projector, codewords, and syndrome circuit."""
    # Stabilizer generators in qubit order [q4, q3, q2, q1, q0]
    M0_def = [Z, X, X, Z, I]
    M1_def = [X, X, Z, I, Z]
    M2_def = [X, Z, I, Z, X]
    M3_def = [Z, I, Z, X, X]
    Unit_def = [I, I, I, I, I]

    M0 = reduce(jnp.kron, jnp.stack(M0_def))
    M1 = reduce(jnp.kron, jnp.stack(M1_def))
    M2 = reduce(jnp.kron, jnp.stack(M2_def))
    M3 = reduce(jnp.kron, jnp.stack(M3_def))
    Unit = jnp.eye(*jnp.shape(M0))

    projector = (1 / 4) * (Unit + M0) @ (Unit + M1) @ (Unit + M2) @ (Unit + M3)

    codeword0 = projector @ jnp.ravel(reduce(jnp.kron, jnp.stack([ket0] * 5)))
    codeword0 = codeword0 / jnp.linalg.norm(codeword0)
    codeword1 = projector @ jnp.ravel(reduce(jnp.kron, jnp.stack([ket1] * 5)))
    codeword1 = codeword1 / jnp.linalg.norm(codeword1)

    # Syndrome circuit: ancilla (4 qubits) ⊗ data (5 qubits) = 9 qubits
    ctrl0 = jnp.array([[1, 0], [0, 0]])
    ctrl1 = jnp.array([[0, 0], [0, 1]])

    gate0 = jnp.round(reduce(jnp.kron, jnp.stack([H, H, H, H] + Unit_def)), 5)
    gate1 = (reduce(jnp.kron, jnp.stack([I, I, I, ctrl0] + Unit_def))
             + reduce(jnp.kron, jnp.stack([I, I, I, ctrl1] + M0_def)))
    gate2 = (reduce(jnp.kron, jnp.stack([I, I, ctrl0, I] + Unit_def))
             + reduce(jnp.kron, jnp.stack([I, I, ctrl1, I] + M1_def)))
    gate3 = (reduce(jnp.kron, jnp.stack([I, ctrl0, I, I] + Unit_def))
             + reduce(jnp.kron, jnp.stack([I, ctrl1, I, I] + M2_def)))
    gate4 = (reduce(jnp.kron, jnp.stack([ctrl0, I, I, I] + Unit_def))
             + reduce(jnp.kron, jnp.stack([ctrl1, I, I, I] + M3_def)))

    syndrome_circuit = gate0 @ gate4 @ gate3 @ gate2 @ gate1 @ gate0

    return codeword0, codeword1, syndrome_circuit


def inject_single_pauli(codeword, pauli, qubit):
    """
    Inject a single-qubit Pauli error on the logical state.
    qubit: actual data qubit index (0 = rightmost, 4 = leftmost in tensor product)
    Returns the 9-qubit state (4 ancilla |0000⟩ ⊗ corrupted codeword).
    """
    ops = [I] * 5
    ops[4 - qubit] = pauli  # map actual qubit to ops-list index
    error_op = reduce(jnp.kron, jnp.stack([I, I, I, I] + ops))

    ancilla = jnp.ravel(reduce(jnp.kron, jnp.stack([ket0] * 4)))
    state = jnp.kron(ancilla, codeword)
    return error_op @ state


def measure_ancillas(state, n_anc=4, n_data=5):
    """Return the most likely ancilla measurement outcome as an integer."""
    probs = np.zeros(2**n_anc)
    N = n_anc + n_data
    for a in range(2**n_anc):
        total = 0.0
        for i in range(2**N):
            if (i >> n_data) == a:
                total += float(jnp.abs(state[i])**2)
        probs[a] = total
    return int(np.argmax(probs))


def apply_correction(state, syndrome):
    """Apply the Pauli correction indicated by the syndrome. Returns corrected state."""
    if syndrome == 0:
        print("No error detected.")
        return state

    if syndrome not in SYNDROME_TO_ERROR:
        raise ValueError(f"Unknown syndrome {syndrome:04b}")

    pauli_label, qubit_idx = SYNDROME_TO_ERROR[syndrome]
    gate_map = {"X": X, "Y": Y, "Z": Z}
    P = gate_map[pauli_label]
    print(f"Syndrome {syndrome:04b}: correcting {pauli_label} on ops-index {qubit_idx} "
          f"(actual data qubit {4 - qubit_idx})")

    ops = [I] * 5
    ops[qubit_idx] = P
    correction = reduce(jnp.kron, jnp.stack([I, I, I, I] + ops))
    return correction @ state


def run_demo():
    """Inject each Pauli error on each qubit and verify the code corrects it."""
    codeword0, codeword1, syndrome_circuit = _setup()

    pauli_map = {"X": X, "Y": Y, "Z": Z}
    all_passed = True

    for logical_label, codeword in [("0_L", codeword0), ("1_L", codeword1)]:
        for qubit in range(5):
            for pauli_label, pauli in pauli_map.items():
                # Inject error
                corrupted = inject_single_pauli(codeword, pauli, qubit)

                # Run syndrome circuit
                after_syndrome = syndrome_circuit @ corrupted

                # Measure
                syndrome = measure_ancillas(after_syndrome)

                # Correct (operates on the full 9-qubit state)
                corrected = apply_correction(after_syndrome, syndrome)

                # After syndrome circuit, state = |syndrome⟩_anc ⊗ E|codeword⟩.
                # After correction, state = |syndrome⟩_anc ⊗ |codeword⟩.
                # Extract data by projecting onto |syndrome⟩ (NOT |0000⟩).
                n_data = 5
                start = syndrome * 2**n_data
                corrected_data = corrected[start:start + 2**n_data]

                fidelity = float(jnp.abs(jnp.dot(jnp.conj(codeword), corrected_data))**2)
                passed = fidelity > 1 - tol
                status = "PASS" if passed else "FAIL"
                if not passed:
                    all_passed = False
                print(f"  |{logical_label}⟩ + {pauli_label} on qubit {qubit}: "
                      f"fidelity={fidelity:.6f} [{status}]")

    print()
    print("All tests passed." if all_passed else "Some tests FAILED.")


if __name__ == "__main__":
    run_demo()
