from panqec.config import CODES, DECODERS, ERROR_MODELS
import numpy as np
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd 
from typing import Union 

import panqec_simulations.BBcode_classes as BBcode
from panqec_simulations.decoder_classes import BeliefPropagationLSDDecoder
from panqec_simulations.errormodel_classes import GaussianPauliErrorModel

### Calculate threshold and get error-rate plot 
from panqec.simulation import read_input_dict
from panqec.analysis import Analysis

"""
* Code written by Anton Brekke * 

This file calculates the error threshold given a BBCode class from 'BBcode_classes.py'. 
"""

def deduce_bias(
    error_model: dict, rtol: float = 0.1
) -> Union[str, float, int]:
    """Deduce the eta ratio from the noise model label.

    Parameters
    ----------
    noise_model : str
        The noise model.
    rtol : float
        Relative tolerance to consider rounding eta value to int.

    Returns
    -------
    eta : Union[str, float, int]
        The eta value. If it's infinite then the string 'inf' is returned.
    """
    eta: Union[str, float, int] = 0

    # Commonly occuring eta values to snap to.
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
        eta_f: float = r_max/(1 - r_max)
        common_matches = np.isclose(eta_f, common_eta_values, rtol=rtol)
        if any(common_matches):
            eta_f = common_eta_values[
                int(np.argwhere(common_matches).flat[0])
            ]
        elif np.isclose(eta_f, np.round(eta_f), rtol=rtol):
            eta_f = int(np.round(eta_f))
        else:
            eta_f = np.round(eta_f, 3)
        eta = eta_f

    return eta

def fix_analysis_raw_data(analysis):
    # 1) Create string keys for grouping
    # df = analysis.raw.assign(
    #     code_str        = analysis.raw['code'].astype(str),
    #     decoder_str     = analysis.raw['decoder'].astype(str),
    #     error_model_str = analysis.raw['error_model'].astype(str),
    #     method_str      = analysis.raw['method'].astype(str)
    # )

    # # 2) Within each parameter-combination, give each row a unique run_id
    # #    cumcount() MUST be called on the DataFrameGroupBy, but we assign its result back to the original df
    # df['run_id'] = df.groupby(
    #     ['code_str', 'decoder_str','error_model_str','method_str']
    # ).cumcount()

    # # 3) Now group by the original INPUT_KEYS *plus* the new run_id
    # grouped = df.groupby(
    #     analysis.INPUT_KEYS + ['run_id']
    # )

    # # 4) Perform exactly the same aggregations as before
    # added_columns   = grouped[['wall_time','n_trials']].sum()
    # concat_columns  = grouped[['effective_error','success','codespace']].aggregate(
    #     lambda x: np.concatenate(x.values)
    # )
    # list_columns    = grouped[['results_file']].aggregate(list)
    # remaining_first = grouped[['code', 'error_rate', 'run_id','error_model','decoder','method']].first()

    # # 5) Stitch them back together
    # analysis._results = pd.concat([
    #     added_columns, concat_columns, list_columns, remaining_first
    # ], axis=1).reset_index(drop=True)

    # # 6) Recompute n_fail, drop helpers, compute biases & params, etc.
    # analysis._results['n_fail'] = (
    #     analysis._results['n_trials'] - analysis._results['success'].apply(sum)
    # )
    # analysis._results['bias'] = analysis._results['error_model'].apply(deduce_bias)

    # for col in ['n','k','d']:
    #     analysis._results[col] = analysis._results['code'].apply(lambda x: x[col])

    # for s in ['code','decoder','error_model','method']:
    #     analysis._results[f'{s}_params'] = analysis._results[s].apply(
    #         lambda x: x['parameters']
    #     )
    #     analysis._results[s] = analysis._results[s].apply(lambda x: x['name'])

    df = analysis.raw.assign(
        code_str        = analysis.raw['code'].astype(str),
        decoder_str     = analysis.raw['decoder'].astype(str),
        error_model_str = analysis.raw['error_model'].astype(str),
        method_str      = analysis.raw['method'].astype(str),
        error_rate      = analysis.raw['error_rate'],
    )
    # ensure unique runs get their own row
    df['run_id'] = df.groupby(
        ['code_str','decoder_str','error_model_str','method_str','error_rate'],
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

    # Group, aggregate, and get a flat DataFrame
    analysis._results = (
        df
        .groupby(analysis.INPUT_KEYS + ['run_id'], as_index=False, sort=False)
        .agg(agg_dict)
    )

    # Now your remaining post-processing is the same:
    analysis._results['n_fail']   = (
        analysis._results['n_trials']
        - analysis._results['success'].apply(sum)
    )
    analysis._results['bias']     = analysis._results['error_model'].apply(deduce_bias)

    for col in ['n','k','d']:
        analysis._results[col] = analysis._results['code'].apply(lambda x: x[col])

    for s in ['code','decoder','error_model','method']:
        analysis._results[f'{s}_params'] = analysis._results[s].apply(lambda x: x['parameters'])
        analysis._results[s]             = analysis._results[s].apply(lambda x: x['name'])

    analysis.apply_overrides()
    analysis.calculate_total_error_rates()
    analysis.calculate_word_error_rates()
    analysis.calculate_single_qubit_error_rates()
    analysis.assign_labels()
    analysis.reorder_columns()

def simulate_code(BBclass: BBcode.BB2DCode=BBcode.BBcode_Toric,
                  error_model_dict: dict = {'name': 'GaussianPauliErrorModel',  #  Class name of the error model
                                                  'parameters': [{'r_x': 1/3, 'r_y': 1/3, 'r_z': 1/3}]},
                  decoder_dict: dict = {'name': 'BeliefPropagationLSDDecoder',  #  Class name of the decoder
                                              'parameters': [{'max_bp_iter': 1e3, 'lsd_order': 10, 
                                              'channel_update': False, 'bp_method': 'minimum_sum'}]}, 
                  n_trials: int=1e2, 
                  grids: list[dict]=[{'L_x':10,'L_y':10}],
                  p_range: tuple=(0.1, 0.25, 40),
                  ask_overwrite: bool=True):

    n_trials = int(n_trials)  # Ensure n_trials is an integer
    p_min, p_max, n_points = p_range
    p = np.linspace(p_min, p_max, n_points)

    # Define which code-class to use 
    # code_class = BBcode.BBcode_Toric
    # code_class = BBcode.BBcode_ArXiV_example
    code_class = BBclass
    code_name = code_class.__name__
    decoder_name = decoder_dict['name']

    # Check if parity checks we implement are the same as in PanQEC 
    # test_code = code_class(4,4)
    # print(test_code.HX)
    # print(test_code.Hx.toarray())
    # print(np.all(test_code.HX == test_code.Hx.toarray()))
    # print(np.all(test_code.HZ == test_code.Hz.toarray()))

    # Must register the new code in panQEC 
    CODES[f'{code_name}'] = code_class
    DECODERS['BeliefPropagationLSDDecoder'] = BeliefPropagationLSDDecoder
    ERROR_MODELS['GaussianPauliErrorModel'] = GaussianPauliErrorModel

    save_frequency = 10  # Frequency of saving to file
    n_trials_str = f'{n_trials:.0e}'.replace('+0', '')
    # Avoid repeating same grid multiple times in filename
    grids_count_list = []
    num_grid_count_list = []
    num_grids = 0
    for g in grids:
        if g not in grids_count_list:
            num_grids = grids.count(g)
            num_grid_count_list.append(num_grids)
            grids_count_list.append(g)

    grids_list_str = [f'{i}{g}' if i != 1 else f'{g}' for g, i in zip(grids_count_list, num_grid_count_list)]

    grids_str = f'{grids_list_str}'.replace(' ', '').replace(':',';').replace(':', ';').replace("'", "").replace('_', '').replace('"', '')
    p_range_str = f'{p_range}'.replace(' ', '')
    # Must copy and edit parameters from error_model_dict in a SAFE way. Trivial .copy() and edit does not work due to deepcopy
    parameters_copy = error_model_dict['parameters'][0].copy()
    rx = parameters_copy['r_x']
    ry = parameters_copy['r_y']
    rz = parameters_copy['r_z']
    parameters_copy['r_x'] = f"{rx:.2f}"
    parameters_copy['r_y'] = f"{ry:.2f}"
    parameters_copy['r_z'] = f"{rz:.2f}"
    parameters_str = f'[{parameters_copy}]'
    error_model_dict_str = f'{error_model_dict}'.replace(f"{error_model_dict['parameters']}", parameters_str).replace(' ', '').replace("'name':", '').replace(':', ';').replace("'", "").replace('_', '')
    decoder_dict_str = f'{decoder_dict}'.replace(' ', '').replace("'name':", '').replace(':', ';').replace("'", "")

    filename = f"data\{code_name};{grids_str};{error_model_dict_str};{decoder_dict_str}.json"

    # This magically fixes the fact that the filename is too long... 
    filename = u"\\\\?\\" + os.path.abspath(filename)

    rewrite_data = True
    if os.path.exists(filename):
        if ask_overwrite: advance = False
        else: advance = True
        while not advance:
            answer = input(f'Filename {filename} already exists. Do you want to write over the existing one (y/n)? ')
            if answer.lower() == 'y':
                advance = True
            elif answer.lower() == 'n':
                rewrite_data = False
                advance = True 

    input_data = {
        'ranges': {
            'label': 'BB 2D Experiment',  # Can be any name you want
            'code': {
                'name': f'{code_name}',  # Class name of the code
                'parameters': grids  # List of dictionaries with code parameters
            },
            'error_model': error_model_dict,
            'decoder': decoder_dict,
            'error_rate': p.tolist()  #  List of physical error rates
        }
    }

    if rewrite_data:
        # If data-file already exists, rewrite the file by setting to 'True'
        if os.path.exists(filename):
            os.remove(filename)
        # We create a BatchSimulation by reading the input dictionary
        batch_sim = read_input_dict(input_data, output_file=filename)
        batch_sim.run(n_trials, progress=tqdm)
    analysis = Analysis(filename)

    # Have to give each code run a unique run_id in case multiple runs have identical parameters
    fix_analysis_raw_data(analysis)

    return analysis, input_data, filename 


def plot_error_rates(analysis, 
                     savefig: bool=False, 
                     filename: str=None, 
                     include_threshold_estimate: bool=True):
    
    if filename is None:
        filename = 'figures/no_filename.pdf'
    ### Plot resulting data 
    plt.style.use('seaborn-v0_8')
    # Comment back in to get LaTeX font 
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    params = {'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.title_fontsize': 10,
            'legend.fontsize': 10,
            'font.size': 10,
            'figure.titlesize': 16} # extend as needed
    # print(plt.rcParams.keys())
    plt.rcParams.update(params)

    # Get colors from https://en.wikipedia.org/wiki/Pantone#Color_of_the_Year
    # Very Peri, Fuchsia Rose, Mimosa, Emerald, Classic Blue, Chili Pepper
    custom_cycle = ["#009473", "#C74375", "#F0C05A", "#6667AB", '#0F4C81', '#9B1B30']
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=custom_cycle)

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    plt.sca(ax[0])
    analysis.plot_thresholds(include_threshold_estimate=include_threshold_estimate)
    plt.sca(ax[1])
    analysis.plot_thresholds(sector='X', include_threshold_estimate=include_threshold_estimate)
    plt.sca(ax[2])
    analysis.plot_thresholds(sector='Z', include_threshold_estimate=include_threshold_estimate)

    fig.tight_layout()
    plt.show()

    # analysis.make_collapse_plots()

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    results = analysis.get_results()        # same as analysis.trunc_results['total']
    # Count number of combinations of L_x and L_y
    dict_arr = np.array([[*L_dict.values()][:-1] for L_dict in results['code_params']])
    n_Ls = 0 
    com_list = []
    for d in dict_arr:
        lx, ly = d
        if (lx, ly) not in com_list:
            com_list.append((lx, ly))
            n_Ls += 1

    # Divide by total number of choices of L_x x L_y in parameters from input_data to only get each code param. one time 
    n_trials_pr = int(len(results['n_trials'])/n_Ls)

    code_names =  results['code'][::n_trials_pr]
    code_params = results['code_params'][::n_trials_pr]
    error_models = results['error_model'][::n_trials_pr]
    decoders = results['decoder'][::n_trials_pr]
    biases = results['bias'][::n_trials_pr]
    num_logical_qubits = results['k'][::n_trials_pr].values

    capsize = 5
    ms = 5
    legend_lines = []
    legend_labels = []
    p_phys_grid = results['error_rate'].to_numpy().reshape((n_Ls, int(len(results['error_rate'].to_numpy())/n_Ls))).T
    results_p_est_grid = results['p_est'].to_numpy().reshape((n_Ls, int(len(results['p_est'].to_numpy())/n_Ls))).T
    results_p_se_grid = results['p_se'].to_numpy().reshape((n_Ls, int(len(results['p_se'].to_numpy())/n_Ls))).T
    results_p_est_X_grid = results['p_est_X'].to_numpy().reshape((n_Ls, int(len(results['p_est_X'].to_numpy())/n_Ls))).T
    results_p_se_X_grid = results['p_se_X'].to_numpy().reshape((n_Ls, int(len(results['p_se_X'].to_numpy())/n_Ls))).T
    results_p_est_Z_grid = results['p_est_Z'].to_numpy().reshape((n_Ls, int(len(results['p_est_Z'].to_numpy())/n_Ls))).T
    results_p_se_Z_grid = results['p_se_Z'].to_numpy().reshape((n_Ls, int(len(results['p_se_Z'].to_numpy())/n_Ls))).T
    for i, Ls in enumerate(code_params):
        Lx, Ly, Lz = Ls.values()
        # code = eval('BBcode.' + results['code'][0] + f'({Lx}, {Ly})')
        k = num_logical_qubits[i]  
        p_phys = p_phys_grid[:, i]
        results_p_est = results_p_est_grid[:, i]
        results_p_se = results_p_se_grid[:, i]
        results_p_est_X = results_p_est_X_grid[:, i]
        results_p_se_X = results_p_se_X_grid[:, i]
        results_p_est_Z = results_p_est_Z_grid[:, i]
        results_p_se_Z = results_p_se_Z_grid[:, i]
        kp = 1 - (1 - p_phys)**k        # 1 - (1-p)^k = k*p to first order

        line = ax[0].errorbar(p_phys, results_p_est, results_p_se,
                    label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$', capsize=capsize, marker='o', ms=ms)
        linecolor = line[0].get_color()
        ax[0].plot(p_phys, kp, color=linecolor, linestyle=(0,(3,6)))
        ax[0].plot(p_phys, kp, color='k', linestyle=(4.5,(3,6)))

        ax[1].errorbar(p_phys, results_p_est_X, results_p_se_X,
                    label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$', capsize=capsize, marker='o', ms=ms)
        ax[2].errorbar(p_phys, results_p_est_Z, results_p_se_Z,
                    label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$', capsize=capsize, marker='o', ms=ms)
        
        legend_lines.append(line)
        legend_labels.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$')

    from matplotlib import lines

    th_line1 = lines.Line2D([], [], color='gray', linestyle=(0,(3,6)))
    th_line2 = lines.Line2D([], [], color='k', linestyle=(4.5,(3,6)))

    legend_lines.append((th_line1, th_line2))
    legend_labels.append('pseudo-threshold')

    result_X = analysis.sector_thresholds['X']
    result_Z = analysis.sector_thresholds['Z']
    if include_threshold_estimate:
        ax[0].axvline(analysis.thresholds.iloc[0]['p_th_fss'], color='red', linestyle='--')
        ax[0].axvspan(analysis.thresholds.iloc[0]['p_th_fss_left'], analysis.thresholds.iloc[0]['p_th_fss_right'],
                    alpha=0.5, color='pink')
        ax[1].axvline(result_X['p_th_fss'][0], color='red', linestyle='--')
        ax[1].axvspan(result_X['p_th_fss_left'][0], result_X['p_th_fss_right'][0],
                    alpha=0.5, color='pink')
        ax[2].axvline(result_Z['p_th_fss'][0], color='red', linestyle='--')
        ax[2].axvspan(result_Z['p_th_fss_left'][0], result_Z['p_th_fss_right'][0],
                    alpha=0.5, color='pink')

    pth_str_1 = r'$p_{\rm th}' + f' = ({100*analysis.thresholds.iloc[0]["p_th_fss"]:.2f}' + '\pm' + f'{100*analysis.thresholds.iloc[0]["p_th_fss_se"]:.2f})\%$'
    pth_str_2 = r'$p_{\rm th}' + f' = ({100*result_X["p_th_fss"][0]:.2f}' + '\pm' + f'{100*result_X["p_th_fss_se"][0]:.2f})\%$'
    pth_str_3 = r'$p_{\rm th}' + f' = ({100*result_Z["p_th_fss"][0]:.2f}' + '\pm' + f'{100*result_Z["p_th_fss_se"][0]:.2f})\%$'

    # ax[0].set_xscale('log')
    # ax[0].set_yscale('log')
    dist = p_phys.max() - p_phys.min()
    ax[0].set_xlim(p_phys.min()-0.05*dist, p_phys.max()+0.05*dist)
    ax[0].set_ylim(ymax=1.1)

    code_name = code_names[0]
    error_model = error_models[0]
    bias_label = str(biases[0]).replace('inf', '\\infty')
    decoder = decoders[0]
    fig.suptitle(f'{error_model} {code_name}\n'f'$\\eta_Z={bias_label}$, {decoder}\n')

    ax[0].set_title('All errors')
    ax[1].set_title('X errors')
    ax[2].set_title('Z errors')

    ax[0].set_xlabel('Physical error rate')
    ax[0].set_ylabel('Logical error rate')
    ax[0].legend(legend_lines, legend_labels, title=pth_str_1)

    ax[1].set_xlabel('Physical error rate')
    ax[1].set_ylabel('Logical error rate')
    ax[1].legend(title=pth_str_2)

    ax[2].set_xlabel('Physical error rate')
    ax[2].set_ylabel('Logical error rate')
    ax[2].legend(title=pth_str_3)

    fig.tight_layout()
    if savefig: plt.savefig(filename)
    plt.show()



def plot_compare_models(analysis_and_inputdata1, analysis_and_inputdata2, 
                        relevant_error_params: list=['r_x', 'r_y', 'r_z'], 
                        relevant_decoder_params: list=['max_bp_iter', 'lsd_order'],
                        collapse_1: bool=False, 
                        collapse_2: bool=False, 
                        savefig=False):
    
    analysis1, input_data1 = analysis_and_inputdata1
    analysis2, input_data2 = analysis_and_inputdata2

    ### Plot resulting data 
    plt.style.use('seaborn-v0_8')
    # Comment back in to get LaTeX font 
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    params = {'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.title_fontsize': 10,
            'legend.fontsize': 10,
            'font.size': 10,
            'figure.titlesize': 16} # extend as needed
    # print(plt.rcParams.keys())
    plt.rcParams.update(params)

    # Get colors from https://en.wikipedia.org/wiki/Pantone#Color_of_the_Year
    custom_cycle = ["#009473", "#C74375", "#F0C05A", "#6667AB", '#0F4C81', '#9B1B30']
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=custom_cycle)

    fig = plt.figure(figsize=(9, 5))
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0, :])

    results1 = analysis1.get_results()        # same as analysis.trunc_results['total']
    raw1 = analysis1.raw
    # count number of grids
    grids_1 = input_data1['ranges']['code']['parameters']
    n_Ls1 = len(grids_1)

    # Divide by total number of choices of L_x x L_y in parameters from input_data to only get each code param. one time 
    n_trials_pr1 = int(len(results1['n_trials'])/n_Ls1)

    code_names1 =  results1['code'][::n_trials_pr1]
    code_params1 = results1['code_params'][::n_trials_pr1]
    error_models1 = results1['error_model'][::n_trials_pr1]
    decoders1 = results1['decoder'][::n_trials_pr1]
    biases1 = results1['bias'][::n_trials_pr1]
    num_logical_qubits1 = results1['k'][::n_trials_pr1].values
    distance_1 = results1['d'][::n_trials_pr1].values

    results2 = analysis2.get_results()        # same as analysis.trunc_results['total']
    raw2 = analysis2.raw
    # Count number of grids
    grids_2 = input_data2['ranges']['code']['parameters']
    n_Ls2 = len(grids_2)

    # Divide by total number of choices of L_x x L_y in parameters from input_data to only get each code param. one time 
    n_trials_pr2 = int(len(results2['n_trials'])/n_Ls2)

    code_names2 =  results2['code'][::n_trials_pr2]
    code_params2 = results2['code_params'][::n_trials_pr2]
    error_models2 = results2['error_model'][::n_trials_pr2]
    decoders2 = results2['decoder'][::n_trials_pr2]
    biases2 = results2['bias'][::n_trials_pr2]
    num_logical_qubits2 = results2['k'][::n_trials_pr2].values
    distance_2 = results2['d'][::n_trials_pr2].values

    # print(distance_1, distance_2)

    code_name1 = code_names1[0]
    error_model1 = error_models1[0]
    bias_label1 = str(biases1[0]).replace('inf', '\\infty')
    decoder1 = decoders1[0]

    code_name2 = code_names2[0]
    error_model2 = error_models2[0]
    bias_label2 = str(biases2[0]).replace('inf', '\\infty')
    decoder2 = decoders2[0]

    relevant_error_params = ['r_x', 'r_y', 'r_z']
    error_params_dict1 = results1['error_model_params'][0]
    relevant_error_param_dict1 = dict([(key, value if value%1==0 else f'{value:.2f}') for key, value in zip(error_params_dict1.keys(), error_params_dict1.values()) if key in relevant_error_params])
    if len(relevant_error_param_dict1) == 0: relevant_error_param_dict_str1 = ''
    else: relevant_error_param_dict_str1 = f"{relevant_error_param_dict1}"

    error_params_dict2 = results2['error_model_params'][0]
    relevant_error_param_dict2 = dict([(key, value if value%1==0 else f'{value:.2f}') for key, value in zip(error_params_dict2.keys(), error_params_dict2.values()) if key in relevant_error_params])
    if len(relevant_error_param_dict2) == 0: relevant_error_param_dict_str2 = ''
    else: relevant_error_param_dict_str2 = f"{relevant_error_param_dict2}"

    relevant_decoder_params = ['gaussian']
    decoder_params_dict1 = results1['decoder_params'][0]
    relevant_decoder_param_dict1 = dict([(key, value) for key, value in zip(decoder_params_dict1.keys(), decoder_params_dict1.values()) if key in relevant_decoder_params])
    if len(relevant_decoder_param_dict1) == 0: relevant_decoder_param_dict_str1 = ''
    else: relevant_decoder_param_dict_str1 = f"{relevant_decoder_param_dict1}"

    decoder_params_dict2 = results2['decoder_params'][0]
    relevant_decoder_param_dict2 = dict([(key, value) for key, value in zip(decoder_params_dict2.keys(), decoder_params_dict2.values()) if key in relevant_decoder_params])
    if len(relevant_decoder_param_dict2) == 0: relevant_decoder_param_dict_str2 = ''
    else: relevant_decoder_param_dict_str2 = f"{relevant_decoder_param_dict2}"

    grid_num = 0 
    grid_count = 0 
    grids_1_str = ''
    for grid in grids_1:
        if grid_num < len(grids_1):
            grid_num = grids_1.count(grid)
            grids_1_str += f'{grid_num}{grid}' if grid_num != 1 else f'{grid}'
            grid_count += grid_num

    grid_num = 0 
    grid_count = 0 
    grids_2_str = ''
    for grid in grids_2:
        if grid_num < len(grids_2):
            grid_num = grids_2.count(grid)
            grids_2_str += f'{grid_num}{grid}' if grid_num != 1 else f'{grid}'
            grid_count += grid_num

    grids_1_str = f"{grids_1_str}".replace("'", '').replace(':', ';').replace(' ', '')
    grids_2_str = f"{grids_2_str}".replace("'", '').replace(':', ';').replace(' ', '')

    error_params_str1 = f"{relevant_error_param_dict_str1}".replace("'", '').replace(':', ';').replace(' ', '')
    error_params_str2 = f"{relevant_error_param_dict_str2}".replace("'", '').replace(':', ';').replace(' ', '')
    decoder_params_str1 = f"{relevant_decoder_param_dict_str1}".replace("'", '').replace(':', ';').replace(' ', '')
    decoder_params_str2 = f"{relevant_decoder_param_dict_str2}".replace("'", '').replace(':', ';').replace(' ', '')
    filename = f"figures\compare_{code_name1}_{error_params_str1}_{decoder_params_str1}_{grids_1_str}_{code_name2}_{error_params_str2}_{decoder_params_str2}_{grids_2_str}.pdf"

    filename = u"\\\\?\\" + os.path.abspath(filename)

    rewrite_plot = True
    if os.path.exists(filename) and savefig:
        advance = False
        while not advance:
            answer = input(f'Filename {filename} already exists. Do you want to write over the existing one (y/n)? ')
            if answer.lower() == 'y':
                advance = True
            elif answer.lower() == 'n':
                rewrite_plot = False
                advance = True 

    capsize = 5
    ms = 5
    from matplotlib import lines 
    analysis1_line = lines.Line2D([], [], color='gray', linestyle='solid')
    legend_lines1 = [analysis1_line]
    legend_labels1 = [f'Analysis 1']
    p_phys1_grid = results1['error_rate'].to_numpy().reshape((n_Ls1, int(len(results1['error_rate'].to_numpy())/n_Ls1))).T
    results1_p_est_grid = results1['p_est'].to_numpy().reshape((n_Ls1, int(len(results1['p_est'].to_numpy())/n_Ls1))).T
    results1_p_se_grid = results1['p_se'].to_numpy().reshape((n_Ls1, int(len(results1['p_se'].to_numpy())/n_Ls1))).T
    # print(p_phys1_grid.T)
    # print(results1_p_est_grid.T)
    # print(results1_p_se_grid.T)
    total_p_est1 = np.ones(results1_p_est_grid[:,0].size)
    k_tot = 0
    for i, Ls in enumerate(code_params1):
        Lx, Ly, Lz = Ls.values()
        # code = eval('BBcode.' + results1['code'][0] + f'({Lx}, {Ly})')
        k = num_logical_qubits1[i] 
        d = distance_1[i] 
        k_tot += k
        p_phys1 = p_phys1_grid[:, i]
        results1_p_est = results1_p_est_grid[:, i]
        results1_p_se = results1_p_se_grid[:, i]
        total_p_est1 *= (1 - results1_p_est)
        kp = 1 - (1 - p_phys1)**k        # 1 - (1-p)^k = k*p to first order

        if not collapse_1:
            line = ax.errorbar(p_phys1, results1_p_est, results1_p_se,
                        label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$, $d\!: {d}$', capsize=capsize, marker='o', ms=ms)
            linecolor = line[0].get_color()
            ax.plot(p_phys1, kp, color=linecolor, linestyle=(0,(3,6)))
            ax.plot(p_phys1, kp, color='k', linestyle=(4.5,(3,6)))
            
            legend_lines1.append(line)
            legend_labels1.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$, $d\!: {d}$')
    # print(plt.rcParams.keys())
    if collapse_1:
        kp = 1 - (1 - p_phys1)**k_tot        # 1 - (1-p)^k = k*p to first order
        line = ax.errorbar(p_phys1, 1-total_p_est1, 0, label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot}$,  $d\!: {np.min(distance_1)}$', linestyle=(0,(2,2)), capsize=capsize, marker='o', ms=ms)
        linecolor = line[0].get_color()
        ax.plot(p_phys1, kp, color=linecolor, linestyle=(0,(3,6)))
        ax.plot(p_phys1, kp, color='k', linestyle=(4.5,(3,6)))

        legend_lines1.append(line)
        legend_labels1.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot}$,  $d\!: {np.min(distance_1)}$')

    # Reset matplotlib color cycle 
    plt.gca().set_prop_cycle(plt.cycler(color=custom_cycle))

    analysis2_line = lines.Line2D([], [], color='gray', linestyle=(0,(2,2)))
    legend_lines2 = [analysis2_line]
    legend_labels2 = [f'Analysis 2']
    p_phys2_grid = results2['error_rate'].to_numpy().reshape((n_Ls2, int(len(results2['error_rate'].to_numpy())/n_Ls2))).T
    results2_p_est_grid = results2['p_est'].to_numpy().reshape((n_Ls2, int(len(results2['p_est'].to_numpy())/n_Ls2))).T
    results2_p_se_grid = results2['p_se'].to_numpy().reshape((n_Ls2, int(len(results2['p_se'].to_numpy())/n_Ls2))).T
    # print(p_phys2_grid.T)
    # print(results2_p_est_grid.T)
    # print(results2_p_se_grid.T)
    total_p_est2 = np.ones(results2_p_est_grid[:,0].size)
    k_tot = 0
    for i, Ls in enumerate(code_params2):
        Lx, Ly, Lz = Ls.values()
        # code = eval('BBcode.' + results2['code'][0] + f'({Lx}, {Ly})')
        k = num_logical_qubits2[i] 
        d = distance_2[i] 
        k_tot += k
        p_phys2 = p_phys2_grid[:, i]
        results2_p_est = results2_p_est_grid[:, i]
        results2_p_se = results2_p_se_grid[:, i]
        total_p_est2 *= (1 - results2_p_est)
        kp = 1 - (1 - p_phys2)**k        # 1 - (1-p)^k = k*p to first order

        if not collapse_2:
            line = ax.errorbar(p_phys2, results2_p_est, results2_p_se, label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$, $d\!: {d}$', linestyle=(0,(2,2)), capsize=capsize, marker='o', ms=ms)

            linecolor = line[0].get_color()
            ax.plot(p_phys2, kp, color=linecolor, linestyle=(0,(3,6)))
            ax.plot(p_phys2, kp, color='k', linestyle=(4.5,(3,6)))
            
            legend_lines2.append(line)
            legend_labels2.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$, $d\!: {d}$')

    if collapse_2:
        kp = 1 - (1 - p_phys2)**k_tot        # 1 - (1-p)^k = k*p to first order
        line = ax.errorbar(p_phys2, 1-total_p_est2, 0, label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot}$,  $d\!: {np.min(distance_2)}$', linestyle=(0,(2,2)), capsize=capsize, marker='o', ms=ms)
        linecolor = line[0].get_color()
        ax.plot(p_phys2, kp, color=linecolor, linestyle=(0,(3,6)))
        ax.plot(p_phys2, kp, color='k', linestyle=(4.5,(3,6)))
    
        legend_lines2.append(line)
        legend_labels2.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot}$,  $d\!: {np.min(distance_2)}$')

    th_line1 = lines.Line2D([], [], color='gray', linestyle=(0,(3,6)))
    th_line2 = lines.Line2D([], [], color='k', linestyle=(4.5,(3,6)))

    legend_lines2.append((th_line1, th_line2))
    legend_labels2.append('pseudo-threshold')

    dist = max(p_phys1.max() - p_phys1.min(), p_phys2.max() - p_phys2.min())
    p_phys_min = min(p_phys1.min(), p_phys2.min())
    p_phys_max = max(p_phys1.max(), p_phys2.max())
    ax.set_xlim(p_phys_min-0.05*dist, p_phys_max+0.05*dist)
    ax.set_ylim(ymax=1.1)

    if 'gaussian' in results1['decoder_params'][::n_trials_pr1].values[0].keys():
        gaussian_decoding1 = results1['decoder_params'][::n_trials_pr1].values[0]['gaussian']
        gauss_decoder_str1 = f'(Gaussian={gaussian_decoding1}) '
    else: 
        gauss_decoder_str1 = ''
    if 'gaussian' in results2['decoder_params'][::n_trials_pr2].values[0].keys():
        gaussian_decoding2 = results2['decoder_params'][::n_trials_pr2].values[0]['gaussian']
        gauss_decoder_str2 = f'(Gaussian={gaussian_decoding2}) '
    else: 
        gauss_decoder_str2 = ''

    # ax.set_title(f'Analysis 1: {error_model1} {code_name1},{decoder1}' + gauss_decoder_str1 + f'$\\eta_Z={bias_label1}$', loc='left', wrap=True)
    # ax.set_title(f'Analysis 2: {error_model2} {code_name2},{decoder2}' + gauss_decoder_str2 + f'$\\eta_Z={bias_label2}$', loc='right', wrap=True)
    title = (
        r"$\begin{array}{l}"
        r"\text{Analysis 1: }" + r"\text{" + f"{error_model1} {code_name1}, $\\eta_Z={bias_label1}$" + r"}\\"
        r"\text{\phantom{Analysis 1: }}" + r"\text{" + f"{decoder1}" + gauss_decoder_str1 + r"}\\"
        r"\text{Analysis 2: }" + r"\text{" + f"{error_model2} {code_name2}, $\\eta_Z={bias_label2}$" + r"}\\"
        r"\text{\phantom{Analysis 2: }}" + r"\text{" + f"{decoder2}" + gauss_decoder_str2 + r"}\\"
        r"\end{array}$"
    )
    title1 = (
        r"$\begin{array}{l}"
        r"\text{Analysis 1: }" + r"\text{" + f"{code_name1}" + r"}\\"
        r"\text{" + f"{error_model1}, $\\eta_Z={bias_label1}$" + r"}\\"
        r"\text{" + f"{decoder1}" + gauss_decoder_str1 + r"}"
        r"\end{array}$"
    )
    title2 = (
        r"$\begin{array}{l}"
        r"\text{Analysis 2: }" + r"\text{" + f"{code_name2}" + r"}\\"
        r"\text{" + f"{error_model2}, $\\eta_Z={bias_label2}$" + r"}\\"
        r"\text{" + f"{decoder2}" + gauss_decoder_str2 + r"}"
        r"\end{array}$"
    )
    # ax.set_title(f'Analysis 1: {error_model1} {code_name1},\n {decoder1}' + gauss_decoder_str1 + f'$\\eta_Z={bias_label1}$\n' + f'Analysis 2: {error_model2} {code_name2},\n{decoder2}' + gauss_decoder_str2 + f'$\\eta_Z={bias_label2}$')
    ax.set_title(title1, loc='left')
    ax.set_title(title2, loc='right')

    ax.set_xlabel('Physical error rate')
    ax.set_ylabel('Logical error rate')
    ax.legend(legend_lines1 + legend_lines2, legend_labels1 + legend_labels2)

    fig.tight_layout()
    if savefig and rewrite_plot: plt.savefig(filename)
    plt.show()


def plot_all(analysis_list, input_data_list):

    ### Plot resulting data 
    plt.style.use('seaborn-v0_8')
    # Comment back in to get LaTeX font 
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    params = {'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.title_fontsize': 10,
            'legend.fontsize': 10,
            'font.size': 10,
            'figure.titlesize': 16} # extend as needed
    # print(plt.rcParams.keys())
    plt.rcParams.update(params)

    # Get colors from https://en.wikipedia.org/wiki/Pantone#Color_of_the_Year
    custom_cycle = ["#009473", "#C74375", "#F0C05A", "#6667AB", '#0F4C81', '#9B1B30']
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=custom_cycle)

    fig = plt.figure(figsize=(9, 5))
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0, :])

    legend_lines1 = []
    legend_labels1 = []
    for analysis1, input_data1 in zip(analysis_list, input_data_list):

        results1 = analysis1.get_results()
        grids_1 = input_data1['ranges']['code']['parameters']
        n_Ls1 = len(grids_1)
        n_trials_pr1 = int(len(results1['n_trials'])/n_Ls1)
        code_params1 = results1['code_params'][::n_trials_pr1]
        num_logical_qubits1 = results1['k'][::n_trials_pr1].values
        code_names1 =  results1['code'][::n_trials_pr1]
        code_name1 = code_names1[0]
        distance_1 = results1['d'][::n_trials_pr1].values

        capsize = 5
        ms = 5
        from matplotlib import lines 
        p_phys1_grid = results1['error_rate'].to_numpy().reshape((n_Ls1, int(len(results1['error_rate'].to_numpy())/n_Ls1))).T
        results1_p_est_grid = results1['p_est'].to_numpy().reshape((n_Ls1, int(len(results1['p_est'].to_numpy())/n_Ls1))).T
        results1_p_se_grid = results1['p_se'].to_numpy().reshape((n_Ls1, int(len(results1['p_se'].to_numpy())/n_Ls1))).T
        # print(p_phys1_grid.T)
        # print(results1_p_est_grid.T)
        # print(results1_p_se_grid.T)
        total_p_est1 = np.ones(results1_p_est_grid[:,0].size)
        k_tot = 0
        for i, Ls in enumerate(code_params1):
            Lx, Ly, Lz = Ls.values()
            # code = eval('BBcode.' + results1['code'][0] + f'({Lx}, {Ly})')
            k = num_logical_qubits1[i] 
            d = distance_1[i] 
            k_tot += k
            p_phys1 = p_phys1_grid[:, i]
            results1_p_est = results1_p_est_grid[:, i]
            results1_p_se = results1_p_se_grid[:, i]
            total_p_est1 *= (1 - results1_p_est)
            kp = 1 - (1 - p_phys1)**k        # 1 - (1-p)^k = k*p to first order

        # print(plt.rcParams.keys())
        kp = 1 - (1 - p_phys1)**k_tot        # 1 - (1-p)^k = k*p to first order
        line = ax.errorbar(p_phys1, 1-total_p_est1, 0, label=rf'{code_name1}: $L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot}$,  $d\!: {np.min(distance_1)}$', capsize=capsize, marker='o', ms=ms)
        linecolor = line[0].get_color()
        ax.plot(p_phys1, kp, color=linecolor, linestyle=(0,(3,6)))
        ax.plot(p_phys1, kp, color='k', linestyle=(4.5,(3,6)))

        legend_lines1.append(line)
        legend_labels1.append(rf'{code_name1}: $L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot}$,  $d\!: {np.min(distance_1)}$')

    th_line1 = lines.Line2D([], [], color='gray', linestyle=(0,(3,6)))
    th_line2 = lines.Line2D([], [], color='k', linestyle=(4.5,(3,6)))

    legend_lines1.append((th_line1, th_line2))
    legend_labels1.append('pseudo-threshold')

    ax.set_xlabel('Physical error rate')
    ax.set_ylabel('Logical error rate')
    ax.legend(legend_lines1, legend_labels1)
    fig.tight_layout()
    plt.show()