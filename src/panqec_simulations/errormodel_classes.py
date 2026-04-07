import functools
from typing import Tuple, Optional
import numpy as np
from panqec.codes import StabilizerCode
from panqec.error_models import BaseErrorModel
from panqec.bpauli import pauli_to_bsf
import random

from scipy.integrate import quad 
from scipy.optimize import newton
from scipy.special import erf, erfinv
from scipy.stats import truncnorm

"""
* Code written by Anton Brekke * 

This file consists of 
 - The decoder class "GaussianPauliErrorModel" as a subclass of PanQEC's "BaseErrorModel" class. 

This class generates Pauli-errors by sampling displacements in position and momentum from a Gaussian distribution,
inspired by the analog QEC from the GKP qubit. 
"""

# Define global variables
th = np.sqrt(np.pi)/2

# Utility functions 
def gaussian(x, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-x**2/(2*sigma**2)) if sigma != 0.0 else 0.0

def root_func(sigma, p):
    return quad(gaussian, -th, th, args=(sigma,))[0] + p - 1

def fast_choice(options, probs, rng=None):
    """Found on stack overflow to accelerate np.random.choice"""
    if rng is None:
        x = random.random()
    else:
        x = rng.random()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]

def sample_outside_tau(sigma, tau):
    alpha = tau/sigma
    # Phi(alpha) using the error function:
    Phi = 0.5*(1 + erf(alpha/np.sqrt(2)))
    tail_mass = 1 - Phi  # = P(X>tau) = P(X< -tau)

    # 1) choose left or right tail with prob 1/2 each
    if np.random.rand() < 0.5:
        # left tail: U mapped to [0, Phi(-alpha)=tail_mass]
        u = np.random.rand()*tail_mass
    else:
        # right tail: U mapped to [Phi(alpha), 1]
        u = Phi + np.random.rand()*tail_mass

    # 2) invert standard normal CDF:
    #    Z = Phi^{-1}(u) = sqrt(2)*erfinv(2u-1)
    Z = np.sqrt(2) * erfinv(2*u - 1)

    # 3) scale to variance sigma^2
    return sigma * Z

class GaussianPauliErrorModel(BaseErrorModel):

    def __init__(self,
                 r_x: float, r_y: float, r_z: float,
                 deformation_name: Optional[str] = None, 
                 deformation_kwargs: Optional[dict] = None):

        if not np.isclose(r_x + r_y + r_z, 1):
            raise ValueError(
                f'Noise direction ({r_x}, {r_y}, {r_z}) does not sum to 1.0'
            )
        self._direction = r_x, r_y, r_z
        self._deformation_name = deformation_name

        if deformation_kwargs is not None:
            self._deformation_kwargs = deformation_kwargs
        else:
            self._deformation_kwargs = {}

    @property
    def direction(self) -> Tuple[float, float, float]:
        """Rate of X, Y and Z errors, as given when initializing the
        error model

        Returns
        -------
        (r_x, r_y, r_z): Tuple[float]
            Rate of X, Y and Z errors
        """
        return self._direction

    @property
    def label(self):
        label = 'Pauli X{:.4f}Y{:.4f}Z{:.4f}'.format(*self.direction)
        if self._deformation_name:
            label = 'Deformed ' + self._deformation_name + ' ' + label

        return label

    @property
    def params(self) -> dict:
        """List of class arguments (as a dictionary), that can be saved
        and reused to instantiate the same code"""
        return {
            'r_x': self.direction[0],
            'r_y': self.direction[1],
            'r_z': self.direction[2],
            'deformation_name': self._deformation_name,
            'deformation_kwargs': self._deformation_kwargs
        }

    def generate(self, code: StabilizerCode, error_rate: float, rng=None):
        rng = np.random.default_rng() if rng is None else rng

        p_i, p_x, p_y, p_z = self.probability_distribution(code, error_rate)
        pi = p_i[0]
        px = p_x[0]
        py = p_y[0]
        pz = p_z[0]
        self.gaussian_error_data_X = []
        self.gaussian_error_data_Z = []
        error_pauli = ''

        # X and Z equally probable
        # p = 1 - np.sqrt(1 - error_rate)
        # sigma = np.sqrt(np.pi/8) * 1/(erfinv(1-p))
        # for _ in range(code.n):
        #     delta_x = rng.normal(0, sigma)
        #     delta_z = rng.normal(0, sigma)
        #     if delta_x > th and delta_z < th:
        #         error_pauli += 'X'
        #         delta_x = np.sqrt(np.pi) - delta_x
        #     elif delta_x < th and delta_z > th:
        #         error_pauli += 'Z'
        #         delta_z = np.sqrt(np.pi) - delta_z
        #     elif delta_x > th and delta_z > th:
        #         error_pauli += 'Y'
        #         delta_x = np.sqrt(np.pi) - delta_x
        #         delta_z = np.sqrt(np.pi) - delta_z
        #     else:
        #         error_pauli += 'I'

        """
        See https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054
        error_rate = px + pz + py = qx*(1-qz) + qz*(1-qx) + qx*qz
        px = qx*(1-qz) = qx - qx*qz = qx - py --> qx = x_prob = px + py  
        """
        x_prob = px + py
        z_prob = pz + py

        # sigma_x = newton(root_func, x0=1, args=(p,))
        sigma_x = np.sqrt(np.pi/8) * 1/(erfinv(1-x_prob))
        sigma_z = np.sqrt(np.pi/8) * 1/(erfinv(1-z_prob))

        # Uniform sample E = I, X, Y, Z and then force delta > sqrt(pi)/2
        # for _ in range(code.n):
        #     E = fast_choice(('I', 'X', 'Y', 'Z'), (pi, px, py, pz), rng=rng)
        #     error_pauli += E
        #     if E == 'X':
        #         delta_x = sample_outside_tau(sigma_x, th)
        #         delta_x = np.sqrt(np.pi) - abs(delta_x)
        #         delta_z = rng.normal(0, sigma_z)
        #     elif E == 'Z':
        #         delta_z = sample_outside_tau(sigma_z, th)
        #         delta_z = np.sqrt(np.pi) - abs(delta_z)
        #         delta_x = rng.normal(0, sigma_x)
        #     elif E == 'Y':
        #         delta_x = sample_outside_tau(sigma_x, th)
        #         delta_z = sample_outside_tau(sigma_z, th)
        #         delta_x = np.sqrt(np.pi) - abs(delta_x)
        #         delta_z = np.sqrt(np.pi) - abs(delta_z)
        #     else: 
        #         delta_x = rng.normal(0, sigma_x)
        #         delta_z = rng.normal(0, sigma_z)

        # Find E = I, X, Y, Z bases on gaussian sample of shift 
        for _ in range(code.n):
            delta_x = abs(rng.normal(0, sigma_x))
            delta_z = abs(rng.normal(0, sigma_z))
            if delta_x > th and delta_z < th:
                error_pauli += 'X'
                delta_x = np.sqrt(np.pi) - delta_x
            elif delta_x < th and delta_z > th:
                error_pauli += 'Z'
                delta_z = np.sqrt(np.pi) - delta_z
            elif delta_x > th and delta_z > th:
                error_pauli += 'Y'
                delta_x = np.sqrt(np.pi) - delta_x
                delta_z = np.sqrt(np.pi) - delta_z
            else:
                error_pauli += 'I'

            # If sigma -> 0, then delta -> 0. Hence f_inc = 0 and f_corr = infinity. 
            # Then p_correct = 1, and p_wrong = 0.
            likelihood_wrong_X = gaussian(np.sqrt(np.pi)-delta_x, sigma_x)
            likelihood_correct_X = gaussian(delta_x, sigma_x)
            normalization_X = likelihood_wrong_X + likelihood_correct_X
            p_wrong_X = likelihood_wrong_X/normalization_X if normalization_X != 0.0 else 0.0
            self.gaussian_error_data_X.append(p_wrong_X)

            likelihood_wrong_Z = gaussian(np.sqrt(np.pi)-delta_z, sigma_z)
            likelihood_correct_Z = gaussian(delta_z, sigma_z)
            normalization_Z = likelihood_wrong_Z + likelihood_correct_Z
            p_wrong_Z = likelihood_wrong_Z/normalization_Z if normalization_Z != 0.0 else 0.0
            self.gaussian_error_data_Z.append(p_wrong_Z)
            
            # num_X_error = error_pauli.count('X')
            # num_Y_error = error_pauli.count('Y')
            # num_Z_error = error_pauli.count('Z')
            # num_I_error = error_pauli.count('I')
            # n_trials = len(error_pauli)
            # PX = num_X_error / n_trials
            # PY = num_Y_error / n_trials
            # PZ = num_Z_error / n_trials
            # P_err = PX + PY + PZ
            # print(P_err, error_rate, n_trials)
            # Only consider X-error first 
            # if delta_x < th: 
            #     error_pauli += options[0]
            #     delta = delta_x
            # else: 
            #     error_pauli += options[1]
            #     delta = np.sqrt(np.pi) - delta_x

            # likelihood_wrong = gaussian(np.sqrt(np.pi)-delta, sigma_x)
            # likelihood_correct = gaussian(delta, sigma_x)
            # normalization = likelihood_wrong + likelihood_correct
            # # If sigma --> 0, delta --> 0 and hence f_inc = 0 and f_corr = infinity. Then p_correct = 1, and p_wrong = 0
            # p_wrong = likelihood_wrong/normalization if normalization != 0.0 else 0.0
            # self.gaussian_error_data_X.append(p_wrong)

        error = pauli_to_bsf(error_pauli)

        return error

    @functools.lru_cache()
    def probability_distribution(
        self, code: StabilizerCode, error_rate: float
        ) -> Tuple:
        n = code.n
        r_x, r_y, r_z = self.direction

        p: dict = {}
        p['I'] = (1 - error_rate) * np.ones(n)
        p['X'] = (r_x * error_rate) * np.ones(n)
        p['Y'] = (r_y * error_rate) * np.ones(n)
        p['Z'] = (r_z * error_rate) * np.ones(n)

        if self._deformation_name is not None:
            for i in range(code.n):
                deformation = code.get_deformation(
                    code.qubit_coordinates[i], self._deformation_name,
                    **self._deformation_kwargs
                )
                previous_p = {pauli: p[pauli][i] for pauli in ['X', 'Y', 'Z']}
                for pauli in ['X', 'Y', 'Z']:
                    p[pauli][i] = previous_p[deformation[pauli]]

        return p['I'], p['X'], p['Y'], p['Z']