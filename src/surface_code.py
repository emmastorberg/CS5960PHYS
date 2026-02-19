import numpy as np

class DataQubit:
    def __init__(self, idx):
        self.idx = idx
        self.errors = {"X": 0, "Z": 0}  # bit flags for current error state

class Stabilizer:
    def __init__(self, stype, qubits):
        # stype = 'X' or 'Z'
        # qubits = list of data-qubit indices it acts on
        self.stype = stype
        self.qubits = qubits
        self.syndrome = 0  # last measured value (0 for +1, 1 for -1)

class SurfaceCodePatch:
    def __init__(self, width: int, height: int, p_error: int = 1e-3) -> None:
        self.grid = [DataQubit(i) for i in range(width * height)]
        self.width = width
        self.height = height
        self.p_error = p_error

        self._assign_stabilizers()

    def _square(self, x, y):
        # find the square face around an index
        q = lambda a,b: a*self.height + b
        return [q(x,y), q(x+1,y), q(x,y+1), q(x+1,y+1)]

    def _assign_stabilizers(self):
        self.stabilizers = []
        for x in range(self.width-1):
            for y in range(self.height-1):
                # X-type star
                self.stabilizers.append(
                    Stabilizer("X", qubits=self._square(x, y))
                )
                # Z-type plaquette
                self.stabilizers.append(
                    Stabilizer("Z", qubits=self._square(x, y))
                )

    def _inject_errors(self):
        for dq in self.grid:
            if np.random.random() < self.p_error:
                err_type = np.random.choice(["X","Z"])
                dq.errors[err_type] ^= 1

    def _measure_stabilizers(self):
        for stab in self.stabilizers:
            parity = 0
            for qidx in stab.qubits:
                dq = self.grid[qidx]
                # flip syndrome if error anticommutes
                if stab.stype == "X" and dq.errors["Z"]:
                    parity ^= 1
                if stab.stype == "Z" and dq.errors["X"]:
                    parity ^= 1
            stab.syndrome = parity
    
    def _decode_and_correct(self):
        for stab in self.stabilizers:
            if stab.syndrome == 1:
                # apply correction of the same type
                for qidx in stab.qubits:
                    dq = self.grid[qidx]
                    dq.errors[stab.stype] ^= 1  # flip back

    def stabilizer_cycle(self):
        self._inject_errors()
        self._measure_stabilizers()
        self._decode_and_correct()

    def logical_error_occurred(self):
        # check if an uncorrectable chain spans the patch
        # (toy check: if > d//2 qubits have same-type error)
        count_X = sum(dq.errors["X"] for dq in self.grid)
        return count_X > self.width//2

if __name__ == "__main__":
    # create one patch (logical qubit)
    patch = SurfaceCodePatch(width=3, height=3, p_error=1e-4)

    # run a few correction cycles
    for _ in range(1000):
        patch.stabilizer_cycle()
        if patch.logical_error_occurred():
            print(_)
            print("Logical failure!")
            break