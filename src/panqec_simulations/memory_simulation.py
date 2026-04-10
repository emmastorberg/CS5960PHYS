from panqec.config import CODES, DECODERS, ERROR_MODELS
from panqec.simulation import read_input_dict
from panqec.analysis import Analysis
import numpy as np
import os
from tqdm import tqdm
from typing import Union

import panqec_simulations.BBcode_classes as BBcode
from panqec_simulations.decoder_classes import BeliefPropagationLSDDecoder
from panqec_simulations.errormodel_classes import GaussianPauliErrorModel


def deduce_bias(
    error_model: dict, rtol: float = 0.1
) -> Union[str, float, int]:
    """Deduce the eta ratio from the noise model label.

    Parameters
    ----------
    error_model : dict
        The error model dict (with 'parameters' key containing r_x, r_y, r_z).
    rtol : float
        Relative tolerance to consider rounding eta value to int.

    Returns
    -------
    eta : Union[str, float, int]
        The eta value. If infinite, the string 'inf' is returned.
    """
    eta: Union[str, float, int] = 0

    common_eta_values = [0.5, 3, 10, 30, 100, 300, 1000]

    direction = (
        error_model['parameters']['r_x'],
        error_model['parameters']['r_y'],
        error_model['parameters']['r_z'],
    )
    r_max = np.max(direction)
    if r_max == 1:
        eta = 'inf'
    else:
        eta_f: float = r_max / (1 - r_max)
        common_matches = np.isclose(eta_f, common_eta_values, rtol=rtol)
        if any(common_matches):
            eta_f = common_eta_values[int(np.argwhere(common_matches).flat[0])]
        elif np.isclose(eta_f, np.round(eta_f), rtol=rtol):
            eta_f = int(np.round(eta_f))
        else:
            eta_f = np.round(eta_f, 3)
        eta = eta_f

    return eta


def fix_analysis_raw_data(analysis):
    """Post-process raw PanQEC analysis data.

    Assigns unique run_ids to distinguish repeated parameter combinations,
    aggregates trial data, and computes derived quantities (n_fail, bias,
    code parameters). Mutates analysis in place.
    """
    df = analysis.raw.assign(
        code_str        = analysis.raw['code'].astype(str),
        decoder_str     = analysis.raw['decoder'].astype(str),
        error_model_str = analysis.raw['error_model'].astype(str),
        method_str      = analysis.raw['method'].astype(str),
        error_rate      = analysis.raw['error_rate'],
    )
    df['run_id'] = df.groupby(
        ['code_str', 'decoder_str', 'error_model_str', 'method_str', 'error_rate'],
        sort=False
    ).cumcount()

    agg_dict = {
        'wall_time'      : 'sum',
        'n_trials'       : 'sum',
        'effective_error': lambda x: np.concatenate(x.values),
        'success'        : lambda x: np.concatenate(x.values),
        'codespace'      : lambda x: np.concatenate(x.values),
        'results_file'   : list,
        'code'           : 'first',
        'decoder'        : 'first',
        'error_model'    : 'first',
        'method'         : 'first',
        'error_rate'     : 'first',
    }

    analysis._results = (
        df
        .groupby(analysis.INPUT_KEYS + ['run_id'], as_index=False, sort=False)
        .agg(agg_dict)
    )

    analysis._results['n_fail'] = (
        analysis._results['n_trials']
        - analysis._results['success'].apply(sum)
    )
    analysis._results['bias'] = analysis._results['error_model'].apply(deduce_bias)

    for col in ['n', 'k', 'd']:
        analysis._results[col] = analysis._results['code'].apply(lambda x: x[col])

    for s in ['code', 'decoder', 'error_model', 'method']:
        analysis._results[f'{s}_params'] = analysis._results[s].apply(lambda x: x['parameters'])
        analysis._results[s]             = analysis._results[s].apply(lambda x: x['name'])

    analysis.apply_overrides()
    analysis.calculate_total_error_rates()
    analysis.calculate_word_error_rates()
    analysis.calculate_single_qubit_error_rates()
    analysis.assign_labels()
    analysis.reorder_columns()


def simulate_code(
    BBclass: BBcode.BB2DCode = BBcode.BBcode_Toric,
    error_model_dict: dict = {
        'name': 'GaussianPauliErrorModel',
        'parameters': [{'r_x': 1/3, 'r_y': 1/3, 'r_z': 1/3}]
    },
    decoder_dict: dict = {
        'name': 'BeliefPropagationLSDDecoder',
        'parameters': [{'max_bp_iter': 1e3, 'lsd_order': 10,
                        'channel_update': False, 'bp_method': 'minimum_sum'}]
    },
    n_trials: int = 1e2,
    grids: list[dict] = [{'L_x': 10, 'L_y': 10}],
    p_range: tuple = (0.1, 0.25, 40),
    ask_overwrite: bool = True,
) -> tuple[Analysis, dict, str]:
    """Run a quantum memory threshold simulation and return the analysis.

    Parameters
    ----------
    BBclass : BB2DCode subclass
        The code class to simulate.
    error_model_dict : dict
        PanQEC-format error model specification.
    decoder_dict : dict
        PanQEC-format decoder specification.
    n_trials : int
        Number of Monte Carlo trials per error rate.
    grids : list[dict]
        List of {'L_x': ..., 'L_y': ...} dicts defining code sizes to sweep.
    p_range : tuple
        (p_min, p_max, n_points) for the physical error rate sweep.
    ask_overwrite : bool
        If True, prompt before overwriting an existing data file.

    Returns
    -------
    analysis : Analysis
        Populated PanQEC Analysis object.
    input_data : dict
        The PanQEC input specification dict used for the simulation.
    filename : str
        Path to the JSON data file written to disk.
    """
    n_trials = int(n_trials)
    p_min, p_max, n_points = p_range
    p = np.linspace(p_min, p_max, n_points)

    code_class = BBclass
    code_name = code_class.__name__

    CODES[code_name] = code_class
    DECODERS['BeliefPropagationLSDDecoder'] = BeliefPropagationLSDDecoder
    ERROR_MODELS['GaussianPauliErrorModel'] = GaussianPauliErrorModel

    # Build filename from simulation parameters
    grids_count_list = []
    num_grid_count_list = []
    for g in grids:
        if g not in grids_count_list:
            grids_count_list.append(g)
            num_grid_count_list.append(grids.count(g))

    grids_list_str = [
        f'{i}{g}' if i != 1 else f'{g}'
        for g, i in zip(grids_count_list, num_grid_count_list)
    ]
    grids_str = (
        f'{grids_list_str}'
        .replace(' ', '').replace(':', ';').replace("'", '').replace('_', '').replace('"', '')
    )

    parameters_copy = error_model_dict['parameters'][0].copy()
    parameters_copy['r_x'] = f"{parameters_copy['r_x']:.2f}"
    parameters_copy['r_y'] = f"{parameters_copy['r_y']:.2f}"
    parameters_copy['r_z'] = f"{parameters_copy['r_z']:.2f}"
    parameters_str = f'[{parameters_copy}]'
    error_model_dict_str = (
        f'{error_model_dict}'
        .replace(f"{error_model_dict['parameters']}", parameters_str)
        .replace(' ', '').replace("'name':", '').replace(':', ';').replace("'", '').replace('_', '')
    )
    decoder_dict_str = (
        f'{decoder_dict}'
        .replace(' ', '').replace("'name':", '').replace(':', ';').replace("'", '')
    )

    os.makedirs('data', exist_ok=True)
    filename = os.path.join(
        'data', f'{code_name};{grids_str};{error_model_dict_str};{decoder_dict_str}.json'
    )

    rewrite_data = True
    if os.path.exists(filename):
        if ask_overwrite:
            advance = False
            while not advance:
                answer = input(
                    f'File {filename} already exists. Overwrite? (y/n): '
                )
                if answer.lower() == 'y':
                    advance = True
                elif answer.lower() == 'n':
                    rewrite_data = False
                    advance = True
        else:
            rewrite_data = False

    input_data = {
        'ranges': {
            'label': 'BB 2D Experiment',
            'code': {
                'name': code_name,
                'parameters': grids,
            },
            'error_model': error_model_dict,
            'decoder': decoder_dict,
            'error_rate': p.tolist(),
        }
    }

    if rewrite_data:
        if os.path.exists(filename):
            os.remove(filename)
        batch_sim = read_input_dict(input_data, output_file=filename)
        batch_sim.run(n_trials, progress=tqdm)

    analysis = Analysis(filename)
    fix_analysis_raw_data(analysis)

    return analysis, input_data, filename
