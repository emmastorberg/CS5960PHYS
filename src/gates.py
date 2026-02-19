import numpy as np

def identity_gate() -> np.ndarray:
    return np.array([[1, 0], [0, 1]])


def pauli_x_gate() -> np.ndarray:
    return np.array([[0, 1], [1, 0]])


def pauli_y_gate() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]])


def pauli_z_gate() -> np.ndarray:
    return np.array([[1, 0], [0, -1]])


def hadamard_gate() -> np.ndarray:
    return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])


def phase_gate() -> np.ndarray:
    return np.array([[1, 0], [0, 1j]])


def cnot_gate(first_is_control: bool = True) -> np.ndarray:
    if first_is_control:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    else:
        return np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   


def RY_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
        ])


def RX_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ])

def RZ_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ])


def CX_10_gate() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])

def SWAP_gate() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])



def multi_kron(*mats: np.ndarray) -> np.ndarray:
    """Tensor product of multiple matrices."""
    result = mats[0]
    for mat in mats[1:]:
        result = np.kron(result, mat)
    return result