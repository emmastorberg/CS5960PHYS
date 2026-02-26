from panqec.decoders import BaseDecoder
from panqec.codes import StabilizerCode
from panqec.error_models import BaseErrorModel

from ldpc.bp_decoder import BpDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bposd_decoder import BpOsdDecoder
import numpy as np

"""
* Code written by Anton Brekke * 

This file consists of 
 - The decoder class "BeliefPropagationLSDDecoder" as a subclass of PanQEC's "BaseDecoder" class. 
 
This class can use analog information from the "GaussianPauliErrorModel" class in the decoding process. 
"""

class BeliefPropagationLSDDecoder(BaseDecoder):
    label = 'BP-LSD decoder'
    allowed_codes = None  # all codes allowed

    def __init__(self,
                 code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float,
                 gaussian: bool = True,    # only works if error_model has attribute 'gaussian_error_data', otherwise ignored
                 max_bp_iter: int = 1000,
                 channel_update: bool = False,
                 lsd_order: int = 10,
                 bp_method: str = 'minimum_sum',
                 ):
        super().__init__(code, error_model, error_rate)
        self._max_bp_iter = max_bp_iter
        self._channel_update = channel_update
        self._osd_order = lsd_order
        self._bp_method = bp_method
        self.do_gaussian_decoding = gaussian

        # Do not initialize the decoder until we call the decode method.
        # This is required because during analysis, there is no need to
        # initialize the decoder every time.
        self._initialized = False

    @property
    def params(self) -> dict:
        return {
            'max_bp_iter': self._max_bp_iter,
            'channel_update': self._channel_update,
            'lsd_order': self._osd_order,
            'bp_method': self._bp_method,
            'gaussian': self.do_gaussian_decoding,
        }

    def get_probabilities(self):
        pi, px, py, pz = self.error_model.probability_distribution(self.code, self.error_rate)
        return pi, px, py, pz

    def update_probabilities(self, correction: np.ndarray,
                             px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                             direction: str = "x->z") -> np.ndarray:
        """Update X probabilities once a Z correction has been applied"""

        n_qubits = correction.shape[0]

        new_probs = np.zeros(n_qubits)

        if direction == "z->x":
            for i in range(n_qubits):
                if correction[i] == 1:
                    if pz[i] + py[i] != 0:
                        new_probs[i] = py[i] / (pz[i] + py[i])
                else:
                    new_probs[i] = px[i] / (1 - pz[i] - py[i])

        elif direction == "x->z":
            for i in range(n_qubits):
                if correction[i] == 1:
                    if px[i] + py[i] != 0:
                        new_probs[i] = py[i] / (px[i] + py[i])
                else:
                    new_probs[i] = pz[i] / (1 - px[i] - py[i])

        else:
            raise ValueError(
                f"Unrecognized direction {direction} when "
                "updating probabilities"
                )

        return new_probs

    def initialize_decoders(self):
        is_css = self.code.is_css

        if is_css:
            self.z_decoder = BpLsdDecoder(
                self.code.Hx,
                error_rate=self.error_rate,
                error_channel=self.error_channel_data_X,   # If both error_rate and error_channel is used, error_channel is prioritized by ldpc
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                schedule="serial",
                lsd_method="lsd_0",  # Choose from: "lsd_e", "lsd_cs", "lsd0"
                lsd_order=self._osd_order)

            self.x_decoder = BpLsdDecoder(
                self.code.Hz,
                error_rate=self.error_rate,
                error_channel=self.error_channel_data_Z,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                schedule="serial",
                lsd_method="lsd_0",  # Choose from: "lsd_e", "lsd_cs", "lsd0"
                lsd_order=self._osd_order)

        else:
            self.decoder = BpLsdDecoder(
                self.code.stabilizer_matrix,
                error_rate=self.error_rate,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                lsd_method="lsd_0",  # Choose from: "lsd_e", "lsd_cs", "lsd0"
                osd_order=self._osd_order)
        self._initialized = True

    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        if self.do_gaussian_decoding and hasattr(self.error_model, 'gaussian_error_data_X'):
            self.error_channel_data_X = self.error_model.gaussian_error_data_X
        else: 
            self.error_channel_data_X = None
            
        if self.do_gaussian_decoding and hasattr(self.error_model, 'gaussian_error_data_Z'):
            self.error_channel_data_Z = self.error_model.gaussian_error_data_Z
        else: 
            self.error_channel_data_Z = None

        if not self._initialized:
            self.initialize_decoders()

        is_css = self.code.is_css
        n_qubits = self.code.n
        syndrome = np.array(syndrome, dtype=int)

        if is_css:
            syndrome_z = self.code.extract_z_syndrome(syndrome)
            syndrome_x = self.code.extract_x_syndrome(syndrome)

        pi, px, py, pz = self.get_probabilities()

        if self.error_channel_data_X is not None:
            probabilities_x = self.error_channel_data_X
        else:
            # Y-errors also introduce X and Z-errors 
            probabilities_x = px + py

        if self.error_channel_data_Z is not None:
            probabilities_z = self.error_channel_data_Z
        else:
            # Y-errors also introduce X and Z-errors 
            probabilities_z = pz + py

        probabilities = np.hstack([probabilities_z, probabilities_x])

        if is_css:
            # Update probabilities (in case the distribution is new at each
            # iteration)
            self.x_decoder.update_channel_probs(probabilities_x)
            self.z_decoder.update_channel_probs(probabilities_z)

            # Decode Z errors
            z_correction = self.z_decoder.decode(syndrome_x)
            # Relic from BPOSD
            # z_correction = self.z_decoder.osdw_decoding

            # Bayes update of the probability
            if self._channel_update:
                new_x_probs = self.update_probabilities(z_correction, px, py, pz, direction="z->x")
                self.x_decoder.update_channel_probs(new_x_probs)

            # Decode X errors
            x_correction = self.x_decoder.decode(syndrome_z)
            # Relic from BPOSD
            # x_correction = self.x_decoder.osdw_decoding

            correction = np.concatenate([x_correction, z_correction])
        else:
            # Update probabilities (in case the distribution is new at each
            # iteration)
            self.decoder.update_channel_probs(probabilities)

            # Decode all errors
            correction = self.decoder.decode(syndrome)
            # Relic from BPOSD
            # correction = self.decoder.osdw_decoding
            correction = np.concatenate([correction[n_qubits:], correction[:n_qubits]])
            
        return correction
