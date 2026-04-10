import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import lines as mlines


# Shared style helpers

def _apply_style():
    plt.style.use('seaborn-v0_8')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rcParams.update({
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.title_fontsize': 10,
        'legend.fontsize': 10,
        'font.size': 10,
        'figure.titlesize': 16,
    })
    custom_cycle = ["#009473", "#C74375", "#F0C05A", "#6667AB", '#0F4C81', '#9B1B30']
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=custom_cycle)
    return custom_cycle


def _savefig(fig, filename, savefig):
    if savefig:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        fig.savefig(filename)


# ── plot_error_rates ──────────────────────────────────────────────────────────

def plot_error_rates(analysis,
                     savefig: bool = False,
                     filename: str = None,
                     include_threshold_estimate: bool = True):
    """Plot threshold curves and logical error rate vs physical error rate.

    Produces two figures:
      1. PanQEC's built-in threshold plots (all / X / Z sectors).
      2. Manual error-rate curves with pseudo-threshold reference lines.

    Parameters
    ----------
    analysis : panqec.analysis.Analysis
        Post-processed analysis object from simulate_code().
    savefig : bool
        Whether to save figures to disk.
    filename : str
        Path for the second figure. First figure gets '_thresholds' suffix.
    include_threshold_estimate : bool
        Whether to overlay the FSS threshold estimate on the plots.
    """
    if filename is None:
        filename = os.path.join('figures', 'error_rates.pdf')

    custom_cycle = _apply_style()

    # Figure 1: PanQEC built-in threshold plots
    fig1, ax1 = plt.subplots(ncols=3, figsize=(15, 5))
    plt.sca(ax1[0])
    analysis.plot_thresholds(include_threshold_estimate=include_threshold_estimate)
    plt.sca(ax1[1])
    analysis.plot_thresholds(sector='X', include_threshold_estimate=include_threshold_estimate)
    plt.sca(ax1[2])
    analysis.plot_thresholds(sector='Z', include_threshold_estimate=include_threshold_estimate)
    fig1.tight_layout()
    base, ext = os.path.splitext(filename)
    _savefig(fig1, f'{base}_thresholds{ext}', savefig)
    plt.show()

    # Figure 2: manual error-rate curves
    fig2, ax2 = plt.subplots(ncols=3, figsize=(15, 5))
    results = analysis.get_results()

    dict_arr = np.array([[*L_dict.values()][:-1] for L_dict in results['code_params']])
    com_list = []
    for d in dict_arr:
        lx, ly = d
        if (lx, ly) not in com_list:
            com_list.append((lx, ly))
    n_Ls = len(com_list)

    n_trials_pr = int(len(results['n_trials']) / n_Ls)
    code_names        = results['code'][::n_trials_pr]
    code_params       = results['code_params'][::n_trials_pr]
    error_models      = results['error_model'][::n_trials_pr]
    decoders          = results['decoder'][::n_trials_pr]
    biases            = results['bias'][::n_trials_pr]
    num_logical_qubits = results['k'][::n_trials_pr].values

    capsize, ms = 5, 5
    legend_lines, legend_labels = [], []

    n_vals = len(results['error_rate'].to_numpy())

    def _reshape(col):
        return results[col].to_numpy().reshape((n_Ls, n_vals // n_Ls)).T

    p_phys_grid        = _reshape('error_rate')
    p_est_grid         = _reshape('p_est')
    p_se_grid          = _reshape('p_se')
    p_est_X_grid       = _reshape('p_est_X')
    p_se_X_grid        = _reshape('p_se_X')
    p_est_Z_grid       = _reshape('p_est_Z')
    p_se_Z_grid        = _reshape('p_se_Z')

    for i, Ls in enumerate(code_params):
        Lx, Ly, Lz = Ls.values()
        k       = num_logical_qubits[i]
        p_phys  = p_phys_grid[:, i]
        kp      = 1 - (1 - p_phys) ** k

        line = ax2[0].errorbar(
            p_phys, p_est_grid[:, i], p_se_grid[:, i],
            label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$',
            capsize=capsize, marker='o', ms=ms,
        )
        linecolor = line[0].get_color()
        ax2[0].plot(p_phys, kp, color=linecolor, linestyle=(0, (3, 6)))
        ax2[0].plot(p_phys, kp, color='k', linestyle=(4.5, (3, 6)))

        ax2[1].errorbar(
            p_phys, p_est_X_grid[:, i], p_se_X_grid[:, i],
            label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$',
            capsize=capsize, marker='o', ms=ms,
        )
        ax2[2].errorbar(
            p_phys, p_est_Z_grid[:, i], p_se_Z_grid[:, i],
            label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$',
            capsize=capsize, marker='o', ms=ms,
        )
        legend_lines.append(line)
        legend_labels.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$')

    th_line1 = mlines.Line2D([], [], color='gray', linestyle=(0, (3, 6)))
    th_line2 = mlines.Line2D([], [], color='k', linestyle=(4.5, (3, 6)))
    legend_lines.append((th_line1, th_line2))
    legend_labels.append('pseudo-threshold')

    result_X = analysis.sector_thresholds['X']
    result_Z = analysis.sector_thresholds['Z']
    if include_threshold_estimate:
        ax2[0].axvline(analysis.thresholds.iloc[0]['p_th_fss'], color='red', linestyle='--')
        ax2[0].axvspan(
            analysis.thresholds.iloc[0]['p_th_fss_left'],
            analysis.thresholds.iloc[0]['p_th_fss_right'],
            alpha=0.5, color='pink',
        )
        ax2[1].axvline(result_X['p_th_fss'][0], color='red', linestyle='--')
        ax2[1].axvspan(result_X['p_th_fss_left'][0], result_X['p_th_fss_right'][0], alpha=0.5, color='pink')
        ax2[2].axvline(result_Z['p_th_fss'][0], color='red', linestyle='--')
        ax2[2].axvspan(result_Z['p_th_fss_left'][0], result_Z['p_th_fss_right'][0], alpha=0.5, color='pink')

    pth_str_1 = (r'$p_{\rm th}' + f' = ({100*analysis.thresholds.iloc[0]["p_th_fss"]:.2f}'
                 + r'\pm' + f'{100*analysis.thresholds.iloc[0]["p_th_fss_se"]:.2f})\\%$')
    pth_str_2 = (r'$p_{\rm th}' + f' = ({100*result_X["p_th_fss"][0]:.2f}'
                 + r'\pm' + f'{100*result_X["p_th_fss_se"][0]:.2f})\\%$')
    pth_str_3 = (r'$p_{\rm th}' + f' = ({100*result_Z["p_th_fss"][0]:.2f}'
                 + r'\pm' + f'{100*result_Z["p_th_fss_se"][0]:.2f})\\%$')

    dist = p_phys.max() - p_phys.min()
    ax2[0].set_xlim(p_phys.min() - 0.05 * dist, p_phys.max() + 0.05 * dist)
    ax2[0].set_ylim(ymax=1.1)

    code_name   = code_names.iloc[0]
    error_model = error_models.iloc[0]
    bias_label  = str(biases.iloc[0]).replace('inf', '\\infty')
    decoder     = decoders.iloc[0]
    fig2.suptitle(f'{error_model} {code_name}\n$\\eta_Z={bias_label}$, {decoder}\n')

    ax2[0].set_title('All errors')
    ax2[1].set_title('X errors')
    ax2[2].set_title('Z errors')

    for i, (pth_str, title) in enumerate(zip([pth_str_1, pth_str_2, pth_str_3],
                                              ['All errors', 'X errors', 'Z errors'])):
        ax2[i].set_xlabel('Physical error rate')
        ax2[i].set_ylabel('Logical error rate')

    ax2[0].legend(legend_lines, legend_labels, title=pth_str_1)
    ax2[1].legend(title=pth_str_2)
    ax2[2].legend(title=pth_str_3)

    fig2.tight_layout()
    _savefig(fig2, filename, savefig)
    plt.show()


# ── plot_compare_models ───────────────────────────────────────────────────────

def plot_compare_models(analysis_and_inputdata1, analysis_and_inputdata2,
                        relevant_error_params: list = ['r_x', 'r_y', 'r_z'],
                        relevant_decoder_params: list = ['max_bp_iter', 'lsd_order'],
                        collapse_1: bool = False,
                        collapse_2: bool = False,
                        savefig: bool = False,
                        filename: str = None):
    """Overlay logical error rate curves from two separate simulations.

    Parameters
    ----------
    analysis_and_inputdata1 : tuple[Analysis, dict]
        (analysis, input_data) from first simulate_code() call.
    analysis_and_inputdata2 : tuple[Analysis, dict]
        (analysis, input_data) from second simulate_code() call.
    collapse_1 / collapse_2 : bool
        If True, collapse all code sizes in that analysis into a single
        combined logical error rate curve (product over k logicals).
    savefig : bool
        Whether to save the figure.
    filename : str
        Output path. Auto-generated from simulation parameters if None.
    """
    analysis1, input_data1 = analysis_and_inputdata1
    analysis2, input_data2 = analysis_and_inputdata2

    custom_cycle = _apply_style()

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    def _extract(analysis, input_data):
        results = analysis.get_results()
        grids = input_data['ranges']['code']['parameters']
        n_Ls = len(grids)
        n_trials_pr = int(len(results['n_trials']) / n_Ls)
        n_vals = len(results['error_rate'].to_numpy())

        def _reshape(col):
            return results[col].to_numpy().reshape((n_Ls, n_vals // n_Ls)).T

        return results, grids, n_Ls, n_trials_pr, _reshape

    results1, grids_1, n_Ls1, n_pr1, reshape1 = _extract(analysis1, input_data1)
    results2, grids_2, n_Ls2, n_pr2, reshape2 = _extract(analysis2, input_data2)

    code_names1        = results1['code'][::n_pr1]
    code_params1       = results1['code_params'][::n_pr1]
    error_models1      = results1['error_model'][::n_pr1]
    decoders1          = results1['decoder'][::n_pr1]
    biases1            = results1['bias'][::n_pr1]
    num_logical1       = results1['k'][::n_pr1].values
    distance_1         = results1['d'][::n_pr1].values

    code_names2        = results2['code'][::n_pr2]
    code_params2       = results2['code_params'][::n_pr2]
    error_models2      = results2['error_model'][::n_pr2]
    decoders2          = results2['decoder'][::n_pr2]
    biases2            = results2['bias'][::n_pr2]
    num_logical2       = results2['k'][::n_pr2].values
    distance_2         = results2['d'][::n_pr2].values

    capsize, ms = 5, 5

    def _gauss_str(results, n_pr):
        params = results['decoder_params'][::n_pr].values[0]
        if 'gaussian' in params:
            return f'(Gaussian={params["gaussian"]}) '
        return ''

    gauss_str1 = _gauss_str(results1, n_pr1)
    gauss_str2 = _gauss_str(results2, n_pr2)

    p_phys1_grid  = reshape1('error_rate')
    p_est1_grid   = reshape1('p_est')
    p_se1_grid    = reshape1('p_se')
    p_phys2_grid  = reshape2('error_rate')
    p_est2_grid   = reshape2('p_est')
    p_se2_grid    = reshape2('p_se')

    analysis1_line = mlines.Line2D([], [], color='gray', linestyle='solid')
    legend_lines = [analysis1_line]
    legend_labels = ['Analysis 1']

    total_p_est1 = np.ones(p_est1_grid[:, 0].size)
    k_tot1 = 0
    for i, Ls in enumerate(code_params1):
        Lx, Ly, Lz = Ls.values()
        k = num_logical1[i]
        d = distance_1[i]
        k_tot1 += k
        p_phys1 = p_phys1_grid[:, i]
        p_est1  = p_est1_grid[:, i]
        p_se1   = p_se1_grid[:, i]
        total_p_est1 *= (1 - p_est1)
        kp = 1 - (1 - p_phys1) ** k

        if not collapse_1:
            line = ax.errorbar(
                p_phys1, p_est1, p_se1,
                label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$, $d\!: {d}$',
                capsize=capsize, marker='o', ms=ms,
            )
            linecolor = line[0].get_color()
            ax.plot(p_phys1, kp, color=linecolor, linestyle=(0, (3, 6)))
            ax.plot(p_phys1, kp, color='k', linestyle=(4.5, (3, 6)))
            legend_lines.append(line)
            legend_labels.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$, $d\!: {d}$')

    if collapse_1:
        kp = 1 - (1 - p_phys1) ** k_tot1
        line = ax.errorbar(
            p_phys1, 1 - total_p_est1, 0,
            label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot1}$, $d\!: {np.min(distance_1)}$',
            linestyle=(0, (2, 2)), capsize=capsize, marker='o', ms=ms,
        )
        linecolor = line[0].get_color()
        ax.plot(p_phys1, kp, color=linecolor, linestyle=(0, (3, 6)))
        ax.plot(p_phys1, kp, color='k', linestyle=(4.5, (3, 6)))
        legend_lines.append(line)
        legend_labels.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot1}$, $d\!: {np.min(distance_1)}$')

    plt.gca().set_prop_cycle(plt.cycler(color=custom_cycle))

    analysis2_line = mlines.Line2D([], [], color='gray', linestyle=(0, (2, 2)))
    legend_lines += [analysis2_line]
    legend_labels += ['Analysis 2']

    total_p_est2 = np.ones(p_est2_grid[:, 0].size)
    k_tot2 = 0
    for i, Ls in enumerate(code_params2):
        Lx, Ly, Lz = Ls.values()
        k = num_logical2[i]
        d = distance_2[i]
        k_tot2 += k
        p_phys2 = p_phys2_grid[:, i]
        p_est2  = p_est2_grid[:, i]
        p_se2   = p_se2_grid[:, i]
        total_p_est2 *= (1 - p_est2)
        kp = 1 - (1 - p_phys2) ** k

        if not collapse_2:
            line = ax.errorbar(
                p_phys2, p_est2, p_se2,
                label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$, $d\!: {d}$',
                linestyle=(0, (2, 2)), capsize=capsize, marker='o', ms=ms,
            )
            linecolor = line[0].get_color()
            ax.plot(p_phys2, kp, color=linecolor, linestyle=(0, (3, 6)))
            ax.plot(p_phys2, kp, color='k', linestyle=(4.5, (3, 6)))
            legend_lines.append(line)
            legend_labels.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k}$, $d\!: {d}$')

    if collapse_2:
        kp = 1 - (1 - p_phys2) ** k_tot2
        line = ax.errorbar(
            p_phys2, 1 - total_p_est2, 0,
            label=rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot2}$, $d\!: {np.min(distance_2)}$',
            linestyle=(0, (2, 2)), capsize=capsize, marker='o', ms=ms,
        )
        linecolor = line[0].get_color()
        ax.plot(p_phys2, kp, color=linecolor, linestyle=(0, (3, 6)))
        ax.plot(p_phys2, kp, color='k', linestyle=(4.5, (3, 6)))
        legend_lines.append(line)
        legend_labels.append(rf'$L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot2}$, $d\!: {np.min(distance_2)}$')

    th_line1 = mlines.Line2D([], [], color='gray', linestyle=(0, (3, 6)))
    th_line2 = mlines.Line2D([], [], color='k', linestyle=(4.5, (3, 6)))
    legend_lines.append((th_line1, th_line2))
    legend_labels.append('pseudo-threshold')

    dist = max(p_phys1.max() - p_phys1.min(), p_phys2.max() - p_phys2.min())
    ax.set_xlim(min(p_phys1.min(), p_phys2.min()) - 0.05 * dist,
                max(p_phys1.max(), p_phys2.max()) + 0.05 * dist)
    ax.set_ylim(ymax=1.1)

    code_name1  = code_names1.iloc[0]
    error_model1 = error_models1.iloc[0]
    bias_label1  = str(biases1.iloc[0]).replace('inf', '\\infty')
    decoder1     = decoders1.iloc[0]

    code_name2  = code_names2.iloc[0]
    error_model2 = error_models2.iloc[0]
    bias_label2  = str(biases2.iloc[0]).replace('inf', '\\infty')
    decoder2     = decoders2.iloc[0]

    title1 = (
        r"$\begin{array}{l}"
        r"\text{Analysis 1: }" + r"\text{" + f"{code_name1}" + r"}\\"
        r"\text{" + f"{error_model1}, $\\eta_Z={bias_label1}$" + r"}\\"
        r"\text{" + f"{decoder1}" + gauss_str1 + r"}"
        r"\end{array}$"
    )
    title2 = (
        r"$\begin{array}{l}"
        r"\text{Analysis 2: }" + r"\text{" + f"{code_name2}" + r"}\\"
        r"\text{" + f"{error_model2}, $\\eta_Z={bias_label2}$" + r"}\\"
        r"\text{" + f"{decoder2}" + gauss_str2 + r"}"
        r"\end{array}$"
    )
    ax.set_title(title1, loc='left')
    ax.set_title(title2, loc='right')
    ax.set_xlabel('Physical error rate')
    ax.set_ylabel('Logical error rate')
    ax.legend(legend_lines, legend_labels)

    fig.tight_layout()

    if filename is None:
        # Auto-generate filename from simulation parameters
        def _params_str(results, n_pr):
            ep = results['error_model_params'][0]
            dp = results['decoder_params'][0]
            ep_str = str({k: (v if v % 1 == 0 else f'{v:.2f}')
                          for k, v in ep.items() if k in ['r_x', 'r_y', 'r_z']})
            dp_str = str({k: v for k, v in dp.items() if k in ['gaussian']})
            return (ep_str + dp_str).replace("'", '').replace(':', ';').replace(' ', '')

        def _grids_str(grids):
            seen = []
            s = ''
            for g in grids:
                if g not in seen:
                    seen.append(g)
                    n = grids.count(g)
                    s += f'{n}{g}' if n != 1 else f'{g}'
            return s.replace("'", '').replace(':', ';').replace(' ', '')

        fn = os.path.join(
            'figures',
            f'compare_{code_name1}_{_params_str(results1, n_pr1)}_{_grids_str(grids_1)}'
            f'_{code_name2}_{_params_str(results2, n_pr2)}_{_grids_str(grids_2)}.pdf'
        )
        filename = fn

    _savefig(fig, filename, savefig)
    plt.show()


# ── plot_all ──────────────────────────────────────────────────────────────────

def plot_all(analysis_list, input_data_list,
             savefig: bool = False,
             filename: str = None):
    """Overlay collapsed logical error rate curves from multiple simulations.

    Each simulation is collapsed into a single combined curve (product over
    all code sizes / logical qubits), allowing direct comparison across
    different code families or parameter settings on one plot.

    Parameters
    ----------
    analysis_list : list[Analysis]
    input_data_list : list[dict]
    savefig : bool
    filename : str
    """
    if filename is None:
        filename = os.path.join('figures', 'all_codes.pdf')

    _apply_style()

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    legend_lines, legend_labels = [], []

    for analysis, input_data in zip(analysis_list, input_data_list):
        results = analysis.get_results()
        grids = input_data['ranges']['code']['parameters']
        n_Ls = len(grids)
        n_trials_pr = int(len(results['n_trials']) / n_Ls)
        n_vals = len(results['error_rate'].to_numpy())

        def _reshape(col):
            return results[col].to_numpy().reshape((n_Ls, n_vals // n_Ls)).T

        code_name1     = results['code'][::n_trials_pr].iloc[0]
        code_params1   = results['code_params'][::n_trials_pr]
        num_logical1   = results['k'][::n_trials_pr].values
        distance_1     = results['d'][::n_trials_pr].values

        p_phys1_grid = _reshape('error_rate')
        p_est1_grid  = _reshape('p_est')

        total_p_est1 = np.ones(p_est1_grid[:, 0].size)
        k_tot = 0
        for i, Ls in enumerate(code_params1):
            Lx, Ly, Lz = Ls.values()
            k = num_logical1[i]
            k_tot += k
            p_phys1 = p_phys1_grid[:, i]
            total_p_est1 *= (1 - p_est1_grid[:, i])

        kp = 1 - (1 - p_phys1) ** k_tot
        line = ax.errorbar(
            p_phys1, 1 - total_p_est1, 0,
            label=rf'{code_name1}: $L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot}$, $d\!: {np.min(distance_1)}$',
            capsize=5, marker='o', ms=5,
        )
        linecolor = line[0].get_color()
        ax.plot(p_phys1, kp, color=linecolor, linestyle=(0, (3, 6)))
        ax.plot(p_phys1, kp, color='k', linestyle=(4.5, (3, 6)))
        legend_lines.append(line)
        legend_labels.append(
            rf'{code_name1}: $L_x\!: {Lx}$, $L_y\!: {Ly}$, $k\!: {k_tot}$, $d\!: {np.min(distance_1)}$'
        )

    th_line1 = mlines.Line2D([], [], color='gray', linestyle=(0, (3, 6)))
    th_line2 = mlines.Line2D([], [], color='k', linestyle=(4.5, (3, 6)))
    legend_lines.append((th_line1, th_line2))
    legend_labels.append('pseudo-threshold')

    ax.set_xlabel('Physical error rate')
    ax.set_ylabel('Logical error rate')
    ax.legend(legend_lines, legend_labels)
    fig.tight_layout()
    _savefig(fig, filename, savefig)
    plt.show()
