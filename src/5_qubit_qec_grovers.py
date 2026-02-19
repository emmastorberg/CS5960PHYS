import numpy as np
import jax
import jax.numpy as jnp
from functools import reduce
from gates import pauli_x_gate, pauli_y_gate, pauli_z_gate, identity_gate, hadamard_gate

# source: https://www.physics.unlv.edu/~bernard/MATH_book/Chap9/Notebook9_3.pdf

"""Might not errors occurring throughout the error correction, as long as the circuit is actually able to correct errors!"""

tol = 1e-12
p_error = 0.01
error_count = 0

SYNDROME_TO_ERROR = {
    9: ("X", 0),
    15: ("Y", 0),
    6: ("Z", 0),
    4: ("X", 1),
    7: ("Y", 1),
    3: ("Z", 1),
    10: ("X", 2),
    11: ("Y", 2),
    1: ("Z", 2),
    5: ("X", 3),
    13: ("Y", 3),
    8: ("Z", 3),
    2: ("X", 4),
    14: ("Y", 4),
    12: ("Z", 4),
}

# SYNDROME_TO_ERROR = {
#     1: ("X", 1),
#     8: ("X", 2),
#     12: ("X", 3),
#     6: ("X", 4),
#     3: ("X", 5),
# }

ket0 = jnp.array([1, 0])
ket1 = jnp.array([0, 1])

k = jnp.kron
X = pauli_x_gate()
Y = pauli_y_gate()
Z = pauli_z_gate()
I = identity_gate()
H = hadamard_gate()

error_log = {}


def multikron(q5, q4, q3, q2, q1):
    # starting from the bottom of the circuit with q1
    return k(q5, k(q4, k(q3, k(q2, q1))))


def error_gate_specification(error_def, index: int) -> list[jnp.ndarray]:
    error_def[4 - index] = X
    return error_def


def potential_error(current_tick: int, duration: int = 1):
    error_def = [I, I, I, I, I]
    for j in range(duration):
        for i in range(5):
            if (
                np.random.rand()
                < 0.5 * current_tick * p_error  # FIXME: arbitrary choice!!!
            ):  # probability of error grows as time goes on
                global error_count
                error_count += 1
                error_def = error_gate_specification(error_def, i)
                error_log[current_tick] = i

    error_gate = reduce(
        jnp.kron, jnp.stack([I, I, I, I] + error_def)
    )  # FIXME: currently, no error can occur on ancilla qubits
    return error_gate


# def inject_single_pauli(pauli, qubit):
#     ops = [I] * 5
#     ops[4 - qubit] = pauli  # match fixed indexing
#     return reduce(jnp.kron, jnp.stack([I, I, I, I] + ops))


def advance_tick(state, tick, num_ticks: int = 1):
    for i in range(num_ticks):
        error = potential_error(tick)
        state = error @ state
        tick += 1
    return state, tick


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

M = jnp.stack([M0, M1, M2, M3], axis=0)

# define the operator that maps us into the code space based on legal code words:
projector = (
    1 / 4 * (Unit + M0) @ (Unit + M1) @ (Unit + M2) @ (Unit + M3)
)  # TODO: find out how we arrive at this mathematically

codeword0 = projector @ jnp.ravel(
    reduce(jnp.kron, jnp.stack([ket0, ket0, ket0, ket0, ket0]))
)
codeword1 = projector @ jnp.ravel(
    reduce(jnp.kron, jnp.stack([ket1, ket1, ket1, ket1, ket1]))
)

# # Exercise 5: Demonstrate that the code words mapped into the code space are eigenstates of the stabilizers Mi.
# for i in range(4):
#     eigs = jnp.linalg.eigvals(M[i])
#     Av0 = M[i] @ codeword0
#     Av1 = M[i] @ codeword1

#     residuals0 = jnp.linalg.norm(
#         Av0[None, :] - eigs[:, None] * codeword0[None, :], axis=1
#     )
#     residuals1 = jnp.linalg.norm(
#         Av1[None, :] - eigs[:, None] * codeword1[None, :], axis=1
#     )

#     assert jnp.any(residuals0 < tol)
#     assert jnp.any(residuals1 < tol)

# build components of circuit and calculate syndrome
ctrl0 = jnp.array([[1, 0], [0, 0]])
ctrl1 = jnp.array([[0, 0], [0, 1]])

ancilla = jnp.ravel(reduce(jnp.kron, jnp.stack([ket0, ket0, ket0, ket0])))

gate0 = jnp.round(
    reduce(jnp.kron, jnp.stack([H, H, H, H] + Unit_def)), 5
)  # rounded off because of floating point errors, hopefully no issues arise
gate1 = reduce(jnp.kron, jnp.stack([I, I, I, ctrl0] + Unit_def)) + reduce(
    jnp.kron, jnp.stack([I, I, I, ctrl1] + M0_def)
)
gate2 = reduce(jnp.kron, jnp.stack([I, I, ctrl0, I] + Unit_def)) + reduce(
    jnp.kron, jnp.stack([I, I, ctrl1, I] + M1_def)
)
gate3 = reduce(jnp.kron, jnp.stack([I, ctrl0, I, I] + Unit_def)) + reduce(
    jnp.kron, jnp.stack([I, ctrl1, I, I] + M2_def)
)
gate4 = reduce(jnp.kron, jnp.stack([ctrl0, I, I, I] + Unit_def)) + reduce(
    jnp.kron, jnp.stack([ctrl1, I, I, I] + M3_def)
)

syndrome = gate0 @ gate4 @ gate3 @ gate2 @ gate1 @ gate0

state0 = jnp.ravel(
    jnp.kron(reduce(jnp.kron, jnp.stack([ket0, ket0, ket0, ket0])), codeword0)
)
state1 = jnp.ravel(
    jnp.kron(reduce(jnp.kron, jnp.stack([ket0, ket0, ket0, ket0])), codeword1)
)

# now we can measure the ancilla qubits. depending on the results, we know where an error has occurred and what to do.

# EX2 = reduce(jnp.kron, jnp.stack([I, I, I, I, I, I, I, X, I]))


def apply_correction(state, syndrome):
    if syndrome == 0:
        print("No error detected.")
        return state

    if syndrome not in SYNDROME_TO_ERROR:
        raise ValueError(f"Unknown syndrome {syndrome:04b}")

    pauli_label, qubit = SYNDROME_TO_ERROR[syndrome]
    print(f"Applying correction: {pauli_label} on data qubit {qubit}")

    # Map label -> gate
    if pauli_label == "X":
        P = X
    elif pauli_label == "Y":
        P = Y
    elif pauli_label == "Z":
        P = Z
    else:
        raise ValueError("Invalid Pauli label")

    # Build correction operator
    ops = [I] * 5
    ops[qubit] = P

    correction = reduce(jnp.kron, jnp.stack([I, I, I, I] + ops))

    return correction @ state


def run_correction_circuit(
    gate_sequence: list[jnp.ndarray] = [gate0, gate1, gate2, gate3, gate4, gate0]
):
    # initialize tick and state
    state = state0
    tick = 0

    print(state)

    for i in range(len(gate_sequence)):
        state, tick = advance_tick(state, tick)
        state = gate_sequence[i] @ state

    print(state)
    print("error count", error_count)

    def measure_ancillas(state, n_anc=4, n_data=5):
        """
        Returns length-16 array of ancilla outcome probabilities.
        Works regardless of ancilla/data ordering in state vector.
        """
        state = jnp.asarray(state)
        probs = np.zeros(2**n_anc)
        N = n_anc + n_data

        for a in range(2**n_anc):
            total = 0.0
            for i in range(2**N):
                anc_bits = i >> n_data       # top n_anc bits
                if anc_bits == a:
                    total += float(jnp.abs(state[i])**2)
            probs[a] = total

        return probs

    probs = measure_ancillas(state)
    print("Ancilla probabilities:")
    for a, p in enumerate(probs):
        if p > 1e-6:
            print(f"{a:04b}  ->  {p:.6f}")

    measured_syndrome = int(np.argmax(probs))
    print("Measured syndrome:", format(measured_syndrome, "04b"))

    state = apply_correction(state, measured_syndrome)

    print(error_log)


run_correction_circuit()

"""now, we need to read the measure the ancilla qubits to tell us where errors have occured.
we can probably in this case just have a lookup table or something. 
then we correct those errors and we're back in business, ready for a new round of stabilization/error correction!
"""

# next up: make two of these bad boys and do computations.
