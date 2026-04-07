import numpy as np

# ------------------------
# Utilities for vectorized operations
# ------------------------

def apply_X(state, qubit):
    """Apply X gate to a single qubit in the state vector"""
    n = int(np.log2(len(state)))
    new_state = state.copy()
    for i in range(len(state)):
        j = i ^ (1 << (n-1-qubit))
        new_state[j] = state[i]
    return new_state

def apply_Z(state, qubit):
    """Apply Z gate to a single qubit in the state vector"""
    n = int(np.log2(len(state)))
    for i in range(len(state)):
        if (i >> (n-1-qubit)) & 1:
            state[i] *= -1
    return state

def apply_H(state, qubit):
    """Apply Hadamard gate to a single qubit in the state vector"""
    n = int(np.log2(len(state)))
    new_state = state.copy()
    for i in range(len(state)):
        if ((i >> (n-1-qubit)) & 1) == 0:
            j = i | (1 << (n-1-qubit))
            new_state[i] = (state[i] + state[j]) / np.sqrt(2)
            new_state[j] = (state[i] - state[j]) / np.sqrt(2)
    return new_state

# ------------------------
# 1. Initialize 18-qubit state |0>^18
# ------------------------
n_qubits = 18
state = np.zeros(2**n_qubits, dtype=complex)
state[0] = 1.0

# ------------------------
# 2. Logical qubit definitions (indices)
# ------------------------
patch_A = list(range(0, 9))
patch_B = list(range(9, 18))

logical_A_Z = [0, 3, 6]  # vertical
logical_B_Z = [9, 12, 15]

# ------------------------
# 3. Apply Hadamard to all qubits in each patch
# ------------------------
for q in patch_A:
    state = apply_H(state, q)
for q in patch_B:
    state = apply_H(state, q)

# ------------------------
# 4. Grover oracle: flip phase of |11>_L
# ------------------------
def measure_logical_Z(state, logical_indices):
    """Return +1 or -1 based on majority vote along vertical Z line"""
    n = int(np.log2(len(state)))
    val = 0
    for i in range(len(state)):
        amp = abs(state[i])**2
        sign = 1
        for q in logical_indices:
            if (i >> (n-1-q)) & 1:
                sign *= -1
        val += sign * amp
    return 1 if val >= 0 else -1

val_A = measure_logical_Z(state, logical_A_Z)
val_B = measure_logical_Z(state, logical_B_Z)
if val_A == -1 and val_B == -1:
    state *= -1  # phase flip |11>_L

# ------------------------
# 5. Grover diffusion
# ------------------------
# Apply Hadamard again
for q in patch_A:
    state = apply_H(state, q)
for q in patch_B:
    state = apply_H(state, q)

# Flip |00>_L phase
val_A = measure_logical_Z(state, logical_A_Z)
val_B = measure_logical_Z(state, logical_B_Z)
if val_A == 1 and val_B == 1:
    state *= -1

# Return to original basis
for q in patch_A:
    state = apply_H(state, q)
for q in patch_B:
    state = apply_H(state, q)

# ------------------------
# 6. Measure logical qubits
# ------------------------
def decode_logical(state, logical_indices):
    val = measure_logical_Z(state, logical_indices)
    return 0 if val == 1 else 1

result_A = decode_logical(state, logical_A_Z)
result_B = decode_logical(state, logical_B_Z)

print("Logical measurement result:", (result_A, result_B))
