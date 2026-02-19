from panqec.codes import StabilizerCode
from panqec.gui import GUI
from bposd.css import css_code
import numpy as np
from scipy.sparse import issparse
import os 
import json

"""
* Code written by Anton Brekke * 

This file consists of 
 - One superclass 'BB2DCode' as a subclass of the PanQEC class "StabilizerCode", 
   implementing the Bivariate Bicycle (BB) code in 2D.
 - A range of subclasses of 'BB2DCode' making different examples of BB codes.
If file is run as the main-file, you can view selected codes in a GUI. 
"""

# Write own superclass for BBCode as a subclass of StabilizerCode
class BB2DCode(StabilizerCode):
    dimension = 2
    deformation_names = ['XZZX']

    def __init__(self, L_x, L_y=None):
        if L_y is None: L_y = L_x
        # L_x, L_y = 8, 6
        super().__init__(L_x, L_y)

        ell, m = L_x, L_y
        # code length
        n = 2*m*ell

        # Compute check matrices of X- and Z-checks
        I_ell = np.identity(ell, dtype=int)
        I_m = np.identity(m, dtype=int)
        self.I = np.identity(ell*m, dtype=int) 
        # Neat way to make polynomial powers of cycle matrix
        self.x = {}
        self.y = {}
        # x =  S_ell \otimes I_m
        for i in range(ell):
            self.x[i] = np.kron(np.roll(I_ell, i, axis=1), I_m)
        # y =  I_ell \otimes S_m
        for i in range(m):
            self.y[i] = np.kron(I_ell, np.roll(I_m, i, axis=1))

        # Method get_AB defines by subclass 
        A, B = self.get_AB()

        AT = np.transpose(A)
        BT = np.transpose(B)

        # Each row of HX defines an X-type check operator X(v) 
        self.HX = np.hstack((A,B))
        # Each row of HZ defines a Z-type check operator Z(v)
        self.HZ = np.hstack((BT,AT))

        # Rank of matrix over field F_2
        def rank2(A):
            rows,n = A.shape
            X = np.identity(n, dtype=int)

            for i in range(rows):
                y = np.dot(A[i,:], X) % 2
                not_y = (y + 1) % 2
                good = X[:, np.nonzero(not_y)]
                good = good[:,0,:]
                bad = X[:, np.nonzero(y)]
                bad = bad[:,0,:]
                if bad.shape[1]>0 :
                    bad = np.add(bad,  np.roll(bad, 1, axis=1) ) 
                    bad = bad % 2
                    bad = np.delete(bad, 0, axis=1)
                    X = np.concatenate((good, bad), axis=1)
            # now columns of X span the binary null-space of A
            return n - X.shape[1]
        
        # Number of logical qubits, must be non-zero
        self.num_logical_qubits = n - rank2(self.HX) - rank2(self.HZ)
        # Is zero of rank2(H) = n/2 <=> k = 2*dim(ker(A)\cap ker(B)) e.g. no overlap between A and B kernel
        # print(f'Number of logical qubits: {self.num_logical_qubits}')

        qcode=css_code(self.HX, self.HZ)
        self.qcode = qcode
        # Logical check matrices -- for logical operators later
        self.lx = self.qcode.lx
        self.lz = self.qcode.lz


    @property
    def label(self):
        return 'My Toric {}x{}'.format(*self.size)

    def get_qubit_coordinates(self):
        coordinates = []
        Lx, Ly = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                coordinates.append((x, y))

        # Qubits along e_y
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                coordinates.append((x, y))

        return coordinates

    def get_stabilizer_coordinates(self):
        coordinates = []
        Lx, Ly = self.size

        # Vertices
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                coordinates.append((x, y))

        # Face in xy plane
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                coordinates.append((x, y))

        return coordinates

    def stabilizer_type(self, location):
        x, y = location
        if x % 2 == 0 and y % 2 == 0:
            return 'vertex'
        else:
            return 'face'

    def get_stabilizer(self, location):
        """
        Return the X- or Z-stabilizer at panqec grid coordinate `location = (x,y)`.
        Works if self.HX / self.HZ is either a NumPy array or a SciPy sparse matrix.
        For labelling, see 
        https://arxiv.org/pdf/2308.07915 Sec.4, paragraph above Lemma 2.
        """
        x, y = location
        Lx, Ly = self.size
        n2 = Lx*Ly    # n2 = n/2 

        # Pick the right matrix and Pauli
        if self.stabilizer_type(location) == 'vertex':  # Z–check
            H, pauli = self.HZ, 'Z'
        else:                                             # X–check
            H, pauli = self.HX, 'X'

        # 1) Map the doubled-grid (x,y) back to cell coords (cx,cy)
        # x in [0, 2Lx-1], y in [0, 2Ly-1] --> cx in [0, Lx-1], cy in [0, Ly-1]
        cx, cy = x // 2, y // 2
        # row number in flattened coordinates. row in [0, ell*m-1]
        row = cy + cx*Ly

        # 2) Extract that row as a 1D vector, dense or sparse
        rowvec = H[row, :]  # shape (2*n2,)

        # 3) Find the 1-entries (indices, signify the column of H)
        cols = np.flatnonzero(rowvec) # Alternatively use np.nonzero(rowvec)[0]
        operator = {}
        for j in cols:
            # determine left/right half (A or B)
            if j < n2:
                idx, layer = j, 'left'
            else:
                idx, layer = j - n2, 'right'

            # turn idx -> cell-offset
            cj = idx // Ly
            ci = idx % Ly

            """
            * lift back to doubled panqec grid
            * Can split into horizontal and vertical. See e.g. 
            * https://quantumcomputing.stackexchange.com/questions/39376/how-are-stabilizers-defined-with-monomial-labellings-for-bivariate-bicycle-codes/39529#39529
            NB: To make comparable with stabilizer defined in article, change layer == 'left' to layer == 'right' below. 
            However, this definition is not equal to the one PanQEC makes. Print self.HX (our) vs self.Hx (PanQEC) to compare.
            Equivalent to change A <--> B (or more like x <--> y)
            """
            if layer == 'left':
                # (odd, even) = horizontal edge 
                qx = (2*cj+1) % (2*Lx)  # odd x, even y -> horizontal
                qy = (2*ci) % (2*Ly)  # even y
            else:
                # (even, odd) = vertical edge
                qx = (2*cj) % (2*Lx)  # even x, odd y -> vertical
                qy = (2*ci+1) % (2*Ly)  # odd y

            if self.is_qubit((qx, qy)):
                operator[(qx, qy)] = pauli
                # print(x, y, operator)
        # Check for correct weight
        # print(len(operator))
        return operator


    def qubit_axis(self, location):
        x, y = location

        if (x % 2 == 1) and (y % 2 == 0):
            axis = 'x'
        elif (x % 2 == 0) and (y % 2 == 1):
            axis = 'y'

        return axis

    def get_logicals_x(self):
        Lx, Ly = self.size
        n2 = Lx*Ly
        logicals = []

        # self.lx should be shape (k, 2*n2)
        for raw_row in self.lx:
            # 1) convert to a flat 1D numpy array of 0/1
            # if issparse(raw_row):
            row_vec = raw_row.toarray().ravel()
            # else:
            #     row_vec = np.asarray(raw_row).ravel()

            # 2) find only the positions with bit == 1 (indices)
            ones = np.flatnonzero(row_vec)

            op = {}
            for j in ones:
                # decide left vs. right half
                if j < n2:
                    idx, layer = j, 'left'
                else:
                    idx, layer = j - n2, 'right'

                # decode into cell‐coords
                cj = idx // Ly
                ci = idx % Ly

                # lift back onto the doubled 2L×2L grid
                if layer == 'left':
                    qx = (2*cj+1) % (2*Lx)  # odd x, even y -> horizontal
                    qy = (2*ci) % (2*Ly)
                else:
                    qx = (2*cj) % (2*Lx)  # even x, odd y -> vertical
                    qy = (2*ci+1) % (2*Ly)

                op[(qx, qy)] = 'X'

            logicals.append(op)

        return logicals


    def get_logicals_z(self):
        Lx, Ly = self.size
        n2 = Lx * Ly
        logicals = []

        for raw_row in self.lz:
            if issparse(raw_row):
                row_vec = raw_row.toarray().ravel()
            else:
                row_vec = np.asarray(raw_row).ravel()

            ones = np.flatnonzero(row_vec)
            op = {}

            for j in ones:
                if j < n2:
                    idx, layer = j, 'left'
                else:
                    idx, layer = j - n2, 'right'

                cj = idx // Ly
                ci = idx % Ly

                if layer == 'left':
                    qx = (2*cj+1) % (2*Lx)  # odd x, even y -> horizontal
                    qy = (2*ci) % (2*Ly)
                else:
                    qx = (2*cj) % (2*Lx)    # even x, odd y -> vertical
                    qy = (2*ci+1) % (2*Ly)

                op[(qx, qy)] = 'Z'

            logicals.append(op)

        return logicals

    def get_deformation(self, location, deformation_name, deformation_axis='y'):
        if deformation_name == 'XZZX':
            undeformed_dict = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
            deformed_dict = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}

            if self.qubit_axis(location) == deformation_axis:
                deformation = deformed_dict
            else:
                deformation = undeformed_dict

        else:
            raise ValueError(f"The deformation {deformation_name}"
                            "does not exist")

        # return undeformed_dict
        return deformation

    def stabilizer_representation(self, location, rotated_picture=False):
        # representation = super().stabilizer_representation(location, rotated_picture, json_file='BBcode.json')

        json_file = os.path.join(os.path.dirname(__file__), '..', 'js', 'BBcode.json')
        if json_file is None:
            json_file = os.path.join(
                os.environ['PANQEC_ROOT_DIR'], 'codes', 'gui-config.json'
            )

        stab_type = self.stabilizer_type(location)

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Want to overwrite code_name so sane .json file can be used for all subclasses 
        # code_name = self.id
        code_name = 'BB2DCode'
        picture = 'rotated' if rotated_picture else 'kitaev'

        representation = data[code_name]['stabilizers'][picture][stab_type]
        representation['type'] = stab_type
        representation['location'] = location

        for activation in ['activated', 'deactivated']:
            color_name = representation['color'][activation]
            representation['color'][activation] = self.colormap[color_name]

        return representation

    def qubit_representation(self, location, rotated_picture=False):
        # representation = super().qubit_representation(location, rotated_picture, json_file='BBcode.json')
        json_file = os.path.join(os.path.dirname(__file__), '..', 'js', 'BBcode.json')
        if json_file is None:
            json_file = os.path.join(
                os.environ['PANQEC_ROOT_DIR'], 'codes', 'gui-config.json'
            )

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Want to overwrite code_name so sane .json file can be used for all subclasses 
        # code_name = self.id
        code_name = 'BB2DCode'
        picture = 'rotated' if rotated_picture else 'kitaev'

        representation = data[code_name]['qubits'][picture]
        representation['params']['axis'] = self.qubit_axis(location)
        representation['location'] = location

        for pauli in ['I', 'X', 'Y', 'Z']:
            color_name = representation['color'][pauli]
            representation['color'][pauli] = self.colormap[color_name]

        return representation
    

# Define subclasses for specific BB codes 
# Takes Lx, Ly as parameters inherited from BB2DCode
class BBcode_Toric(BB2DCode):
    """
    Make method 'get_AB' and define A and B matrices
    """
    def get_AB(self):
        # Get I, x, y from superclass 
        I, x, y = self.I, self.x, self.y
        A = (I + y[1]) % 2
        B = (I + x[1]) % 2

        return A, B
    
class BBcode_ArXiV_example(BB2DCode):
    """
    Make method 'get_AB' and define A and B matrices
    This example works in the dual lattice, as they assign 'left'<--> A to vertical rather than horizontal. 
    To make comparable with stabilizer defined in article, change layer == 'left' to layer == 'right' below. 
    However, this definition is not equal to the one PanQEC makes. Print self.HX (our) vs self.Hx (PanQEC) to compare. 
    Equivalent to change A <--> B (or more like x <--> y)
    """
    def get_AB(self):
        # Get I, x, y from superclass 
        x, y = self.x, self.y
        # Compare with Fig.1 from https://arxiv.org/pdf/2407.03973 in Lx, Ly = 7,7
        # See also https://quantumcomputing.stackexchange.com/questions/39376/how-are-stabilizers-defined-with-monomial-labellings-for-bivariate-bicycle-codes
        # They define 
        A = (x[1] + y[3] + y[4]) % 2
        B = (y[1] + x[3] + x[4]) % 2
        return A, B

class BBcode_A312_B312(BB2DCode):
    # If you want to change grid to rectangular in GUI, define own init and overwrite
    def __init__(self, L_x, L_y=None):
        if L_y is None: L_y = L_x
        # L_x, L_y = 12, 6
        super().__init__(L_x, L_y)
    """
    Make method 'get_AB' and define A and B matrices
    """
    def get_AB(self):
        a1,a2,a3 = 3,1,2
        b1,b2,b3 = 3,1,2
        # Get x, y from superclass 
        x, y = self.x, self.y
        A = (x[a1] + y[a2] + y[a3]) % 2
        B = (y[b1] + x[b2] + x[b3]) % 2
        return A, B
    
class BBcode_Ay3x1x2_Bx3y7y2(BB2DCode):
    """
    Make method 'get_AB' and define A and B matrices
    """
    def get_AB(self):
        a1,a2,a3 = 3,1,2
        b1,b2,b3 = 3,7,2
        # Get x, y from superclass 
        x, y = self.x, self.y
        A = (x[a1] + y[a2] + y[a3]) % 2
        B = (y[b1] + x[b2] + x[b3]) % 2
        return A, B

if __name__ == '__main__':

    def prime_factors(n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return '*'.join(map(str, factors))

    # Visualize the code in the GUI

    from decoder_classes import BeliefPropagationLSDDecoder
    from errormodel_classes import GaussianPauliErrorModel

    gui = GUI()
    gui.add_code(BBcode_Toric, 'BBcode Toric')
    gui.add_code(BBcode_ArXiV_example, 'BBcode ArXiV example')
    gui.add_code(BBcode_A312_B312, 'BBcode-A312-B312')
    gui.add_code(BBcode_Ay3x1x2_Bx3y7y2, 'BBcode-Ay3x1x2-Bx3y7y2')

    gui.add_decoder(BeliefPropagationLSDDecoder, 'BeliefPropagationLSDDecoder')

    BBcode_A312_B312(13, 13)

    num_toric_logicals = 2 
    for i in range(6, 30):
        # Symmetric in (i,j) 
        for j in range(i, 30):
            c = BBcode_A312_B312(i, j)
            if c.num_logical_qubits % 2 == 0 and c.num_logical_qubits > 0:
            # if c.num_logical_qubits == 8:
                num_toric_codes = c.num_logical_qubits//num_toric_logicals
                num_qubit_BB = 2*i*j
                num_qubit_pr_toric = num_qubit_BB / num_toric_codes
                if num_qubit_pr_toric % 2 == 0:
                    Lx_times_Ly_toric = prime_factors(int(num_qubit_pr_toric//2))
                    print((i,j), num_toric_codes, num_qubit_BB, int(num_qubit_pr_toric//2), Lx_times_Ly_toric)
                    print(f'Number of logical qubits: {c.num_logical_qubits}')
                
    gui.run(port=5000)

