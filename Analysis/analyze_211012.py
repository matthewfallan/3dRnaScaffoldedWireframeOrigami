from collections import Counter
import itertools
import math
import os
import random
import re
import sys

sys.path.append("/home/ma629/git")

from Bio import SeqIO
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta, wilcoxon, mannwhitneyu, pearsonr
import seaborn as sns
from statsmodels.formula.api import ols

from ariadne import ariadne, plots, daedalus, terms
from rouls.seq_utils import read_fasta
from rouls.struct_utils import predict_structure, read_ct_file_single
from rouls.dreem_utils import get_clusters_mu_filename, get_sample_and_run, read_clusters_mu, mu_histogram_paired, plot_data_structure_roc_curve, get_data_structure_agreement


plt.rcParams["font.family"] = "Arial"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

analysis_dir = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.dirname(analysis_dir)
em_dir = "EM_Clustering"

projects_dir = "/lab/solexa_rouskin/projects"
corona_dir = os.path.join(projects_dir, "Tammy", "Tammy_corona")

samples_exclude = ["rO44v2"]
dms_bases = ["A", "C"]
n_rre_clusters = 2

rre_ref_structs_name_to_dot = {
        "5-stem": ".((((((((.((((((((..(((((((((...))))))).))........((((((.....)))))).))))))))..(((((((....)))))))........((((..(((((((...)))))))..))))..(((((((((........))))))))).....)))).))))",
        "4-stem": ".((((((((.((((((((..(((((((((...))))))).))........((((((.....)))))).)))))))).((((((((......(((((...(((.....)))....)))))))))))))........(((((((((........))))))))).....)))).))))",
}

rre_ref_structs_dot_to_name = {dot: name for name, dot in rre_ref_structs_name_to_dot.items()}

rre_k_cluster_names = {1: ["5-stem"], 2: ["5-stem", "4-stem"]}

sample_clustering_runs = {("rO66v2", 2): 10}

amplicons = {
    "RRE": [(28, 202)],
    "23S": [(25, 463), (427, 871), (825, 1287), (1222, 1671), (1594, 1958)],
    "M13": [(31, 526), (525, 1033)],
    "EGFP": [(24, 473), (315, 765)],
    "rsc1218v1n660": [(29, 471), (436, 629)],
    "rsc1218v1n924": [(29, 471), (436, 892)],
}

rT66v2_stap10_target = list(range(259, 278 + 1)) + list(range(497, 520 + 1))


def read_sample_data():
    sample_file = os.path.join(analysis_dir, "samples.txt")
    sample_data = pd.read_csv(sample_file, sep="\t", index_col="Sample")
    return sample_data


def num_to_id(num):
    sample_id = f"210422Rou_D21-{num}"
    return sample_id


def get_fasta_full_name(fasta_file, project_dir=proj_dir):
    if not fasta_file.endswith(".fasta"):
        fasta_file = fasta_file + ".fasta"
    ref_genome_dir = os.path.join(project_dir, "Ref_Genome")
    fasta_file = os.path.join(ref_genome_dir, fasta_file)
    return fasta_file


def get_dot_file_full_name(proj_dir, em_dir, sample, ref, start, end, k, c, run=None, **kwargs):
    if run is None:
        run = "best"
    run_dir = get_sample_and_run(os.path.join(proj_dir, em_dir), k, sample, ref, start, end, run=run)
    prefix = run_dir.split(os.sep)[-3]
    dot_file = os.path.join(run_dir, f"{prefix}-K{k}_Cluster{c}_expUp_0_expDown_0_pk.dot")
    return dot_file


def read_dot_file(dot_file):
    with open(dot_file) as f:
        name, seq, structure = [f.readline().strip() for i in range(3)]
    return name, seq, structure


def get_sample_mus(sample_data, sample_name):
    # Get the IDs of the samples containing the origami and the RRE.
    origami_num = sample_data.loc[sample_name, "Origami"]
    origami_id = num_to_id(origami_num)
    rre_num = sample_data.loc[sample_name, "RRE"]
    rre_id = num_to_id(rre_num)
    # Read the sequence of the origami scaffold.
    scaf_name = sample_data.loc[sample_name, "Scaffold"]
    scaf_seq_file = get_fasta_full_name(scaf_name)
    _, scaf_seq = read_fasta(scaf_seq_file)
    # Read the sequence of the RRE.
    rre_seq_file = get_fasta_full_name("RRE")
    _, rre_seq = read_fasta(rre_seq_file)
    # Read mutation rates of RRE.
    ref = "RRE"
    em_dir = "EM_Clustering"
    start, end = amplicons[ref][0]
    rre_mus = dict()
    for k in range(1, n_rre_clusters + 1):
        run = sample_clustering_runs.get((sample_name, k), "best")
        structure_order = list()
        for c in range(1, k + 1):
            dot_file = get_dot_file_full_name(proj_dir, em_dir, rre_id, ref, start, end, k, c, run=run)
            name, seq, structure = read_dot_file(dot_file)
            structure_name = rre_ref_structs_dot_to_name[structure]
            structure_order.append(structure_name)
        rre_mus_file = get_clusters_mu_filename(em_clustering_dir=os.path.join(proj_dir, em_dir), k=k, sample=rre_id, ref=ref, start=start, end=end, run=run)
        rre_mus[k] = read_clusters_mu(rre_mus_file, flatten=False, seq=rre_seq, include_gu=False)
        rre_mus[k].columns = structure_order
    # Read mutation rates of origami.
    k = 1
    origami_mus = pd.Series(dtype=np.float64)
    for amplicon in amplicons[scaf_name]:
        start, end = amplicon
        mus_file = get_clusters_mu_filename(em_clustering_dir=os.path.join(proj_dir, em_dir), k=k, sample=origami_id, ref=scaf_name, start=start, end=end, run="best")
        amplicon_mus = read_clusters_mu(mus_file, flatten=True, seq=scaf_seq, include_gu=False)
        # Average signal in overlaps between amplicons.
        if len(origami_mus) > 0:
            overlap = sorted(set(origami_mus.index) & set(amplicon_mus.index))
            if len(overlap) > 0:
                overlap_prev = origami_mus.loc[overlap]
                overlap_amp = amplicon_mus.loc[overlap]
                corr_amp_prev = np.corrcoef(overlap_amp, overlap_prev)[0,1]
                # Double check that the correlation is high, i.e. above 0.9
                corr_min = 0.9
                assert corr_amp_prev >= corr_min
                # If the correlation is high, average the two, as is common practice for replicates.
                consensus = (overlap_prev + overlap_amp) / 2
                origami_mus.loc[overlap] = consensus
            # Add the new DMS signals.
            amplicon_new = amplicon_mus[sorted(set(amplicon_mus.index) - set(overlap))]
            origami_mus = pd.concat([origami_mus, amplicon_new])
        else:
            origami_mus = amplicon_mus
    return origami_mus, rre_mus


def get_beta_loglike(mus, params):
    """
    Log-likelihood function for beta distribution.
    """
    a, b = params
    loglike = np.sum(beta.logpdf(mus, a, b))
    return loglike


def fit_beta_dist_to_mus(mus):
    """
    For beta distribution, we have
    mean = a / (a + b)
    var = a*b / ( (a + b)^2 * (a + b + 1) )
    Solving for a and b:
    a = -mean * (mean**2 - mean + var) / var
    b = a * (1 - mean) / mean
    """
    n_mus = len(mus)
    mean = np.mean(mus)
    var = np.var(mus)
    a = -mean * (mean**2 - mean + var) / var
    b = a * (1 - mean) / mean
    params_init = np.array([a, b])
    bounds = [(0, None), (0, None)]
    def objective(params):
        return -get_beta_loglike(mus, params)
    result = minimize(objective, params_init, bounds=bounds)
    a, b = result.x
    n_bins = 11
    n_dist = 1001
    x = np.linspace(0, 1, n_dist)
    plt.hist(mus, bins=n_bins)
    plt.plot(x, beta.pdf(x, a, b) * n_mus / n_bins)
    plt.savefig("dist.png")
    plt.close()
    return a, b


def fit_beta_to_prob_labels(mus, labels):
    labels = np.asarray(labels)
    label_set = sorted(set(labels))
    a_params = dict()
    b_params = dict()
    for label in label_set:
        a_params[label], b_params[label] = fit_beta_dist_to_mus(
                mus.loc[labels == label])
    return a_params, b_params


def generate_beta_prob_function(a_params, b_params):
    label_set = list(a_params.keys())
    if any([v > 1 for v in Counter(label_set).values()]):
        raise ValueError("Duplicated labels")
    if sorted(label_set) != sorted(b_params.keys()):
        raise ValueError("a and b labels do not match")
    def get_prob_label(mus, eps=1E-6):
        if eps < 0.0 or eps >= 0.5:
            raise ValueError("eps must be in [0.0, 0.5)")
        if eps > 0.0:
            mus.loc[mus < eps] = eps
            mus.loc[mus > 1.0 - eps] = 1.0 - eps
        pdf = pd.DataFrame.from_dict({l: beta.pdf(mus, a_params[l], b_params[l])
                for l in label_set}, orient="columns")
        pdf.index = mus.index
        probs = pdf.divide(pdf.sum(axis=1), axis="index")
        return probs
    return get_prob_label


def generate_beta_prob_unpaired_function(get_prob_label):
    def beta_prob_unpaired(mus, eps=1E-6):
        p_unp = get_prob_label(mus, eps)["unpaired"]
        return p_unp
    return beta_prob_unpaired


def generate_beta_lodds_unpaired_function(get_prob_label):
    def beta_lodds_unpaired(mus, eps=1E-6):
        lodds = get_log_odds(get_prob_label(mus, eps)["unpaired"])
        return lodds
    return beta_lodds_unpaired


def fit_dist_to_rre_clusters(rre_mus, rre_standard):
    rre_dist_params = dict()
    rre_seq_file = get_fasta_full_name("RRE")
    _, rre_seq = read_fasta(rre_seq_file)
    start, end = amplicons["RRE"][0]
    rre_seq_amp = rre_seq[start: end + 1]
    for cluster_name in rre_k_cluster_names[n_rre_clusters]:
        cluster_mus = rre_mus[n_rre_clusters, cluster_name][rre_standard]
        dot_structure = rre_ref_structs_name_to_dot[cluster_name]
        paired_labels = np.array(["unpaired" if dot == "." else "paired"
            for dot, base in zip(dot_structure, rre_seq_amp) if base in "AC"])
        params = dict()
        params["a"], params["b"] = fit_beta_to_prob_labels(cluster_mus,
                paired_labels)
        for param, param_vals in params.items():
            if param not in rre_dist_params:
                rre_dist_params[param] = dict()
            for label, param_val in param_vals.items():
                if label not in rre_dist_params[param]:
                    rre_dist_params[param][label] = dict()
                rre_dist_params[param][label][cluster_name] = param_val
    # Create function of probability that a base is unpaired.
    rre_dist_params = {param: {label: np.mean(list(clusters.values()))
            for label, clusters in param_vals.items()}
            for param, param_vals in rre_dist_params.items()}
    get_lodds_unpaired = generate_beta_lodds_unpaired_function(
            generate_beta_prob_function(
            rre_dist_params["a"], rre_dist_params["b"]))
    return get_lodds_unpaired


def get_log_odds(x):
    log_odds = np.log10(x / (1.0 - x))
    return log_odds


def get_funcs_exp(params):
    funcs = {param: lambda x: math.pow(val, x) for param, val in params.items()}
    return funcs


def get_funcs_exp_sym(params, non_zero=True):
    funcs = {param: lambda x: 0.0 if non_zero and x == 0 else math.pow(val, abs(x)) for param, val in params.items()}
    return funcs


def run_regression_features_seq(base_info, origami_mus, model_results, all_origamis_data, origami):
    # Get the raw strings of the features.
    feature_strings = base_info["Feature"]
    # Split the features into three parts: strand, feature type, and direction.
    features_split = feature_strings.str.split("_", expand=True)
    features_split.columns = ["Strand", "Feature", "Direction"]
    # Reformat the feature strings into a more interpretable format.
    features = features_split.apply(lambda base: daedalus.format_feature_name(base["Feature"], str(base["Direction"])), axis=1)
    features.name = "Feature"
    features_levels = daedalus.get_formatted_features(base_info)
    # For the regression define the levels of the categorical variable Feature as the sorted features, with the value Middle first so that it is the default.
    default_loc = terms.MIDDLE
    try:
        default_idx = features_levels.index(default_loc)
        features_levels.pop(default_idx)
    except ValueError:
        pass
    features_levels = [default_loc] + features_levels
    # Combine the DMS signals, sequence, and features into one dataframe.
    origami_data = base_info["Base"].to_frame().join([features, origami_mus], how="inner")
    origami_data.columns = ["Sequence"] + list(origami_data.columns[1: -1]) + ["DMS_signal"]
    assert np.all(origami_data["Sequence"].isin(dms_bases))
    origami_data["Origami"] = origami
    #secondary_structure_fname = f"ss_{origami}.pdf"
    #colors = get_prob_unpaired(origami_mus)
    #plots.secondary_structure_signal(secondary_structure_fname, edges, g_up, g_dn, g_ax, base_info, colors)
    for formula_lhs, formula_label in formulas.items():
        print(origami, formula_label)
        zero_thresh = 1e-4
        zero_dms = origami_data["DMS_signal"] < zero_thresh
        print("DMS below thresh:\n", zero_dms.loc[zero_dms])
        origami_data["DMS_signal"] = origami_data["DMS_signal"] + zero_dms * zero_thresh
        formula_rhs = "C(Sequence, levels=dms_bases) + C(Feature, levels=features_levels) + C(Sequence, levels=dms_bases) * C(Feature, levels=features_levels)"
        """
        plt.hist(origami_mus, bins=128)
        plt.savefig("mus.png")
        plt.close()
        plt.hist(np.log10(origami_mus.loc[~zero_dms]), bins=128)
        plt.savefig("mus_log.png")
        plt.close()
        plt.hist(np.log10(origami_mus + 1), bins=128)
        plt.savefig("mus_log+1.png")
        plt.close()
        """
        results = ols(f"{formula_lhs} ~ {formula_rhs}", data=origami_data).fit()
        result_label = f"feature-sequence vs {formula_label}"
        model_results[result_label] = results
        """
        coeffs[name] = lm.params
        conf_ints[name] = lm.conf_int(conf_alpha)
        p_vals[name] = lm.pvalues
        r2s[name] = lm.rsquared
        """
        plt.scatter(results.fittedvalues, results.resid)
        plot_name = f"model_feature-seq_{origami}_{formula_label.replace(' ', '_')}.pdf"
        model_plots(origami_data["DMS_signal"], results, plot_name)
    all_origamis_data.append(origami_data)
    return features_levels


dms_col = "DMS_signal"
interaction_delim = "_int_"

def run_regression_dists_gc_dataframe(origami_data, model_results):
    factors_use = [col for col in origami_data.columns if col not in [dms_col]]
    """
    # Omit factors that are too correlated with another factor.
    coeff_det_max = 0.5  # max allowed R^2 b/w two independent variables
    coeff_det = origami_data.corr()**2
    factors_prim = {fac for fac in factors_all if interaction_delim not in fac}
    factors_int = {fac for fac in factors_all if interaction_delim in fac}
    factors_use = list()
    factors_omit = set()
    """
    # Omit all interaction effects.
    """
    for fac in factors_all:
        if interaction_delim in fac:
            factors_omit.add(fac)
        else:
            factors_use.append(fac)
    """
    """
    for fac in factors_all:
        if interaction_delim in fac:
            terms = fac.split(interaction_delim)
            is_end_or_xo = [any((x in term for x in ["End", "Crossover"])) for term in terms]
            if sum(is_end_or_xo) > 1:
                factors_omit.add(fac)
                continue
    """
    """
    for fac1 in factors_all:
        if fac1 in factors_omit:
            continue
        # factors that have not yet been omitted
        other_factors = factors_all[np.logical_not(np.logical_or(factors_all == fac1, np.isin(factors_all, factors_omit)))]
        colinear_factors = set(other_factors[coeff_det.loc[fac1, other_factors] >= coeff_det_max])
        colinear_factors_prim = colinear_factors & factors_prim
        colinear_factors_int = colinear_factors & factors_int
        if colinear_factors_prim:
            if interaction_delim in fac1:
                factors_omit.append(fac1)
            else:
                raise ValueError(f"{fac1} colinear with {colinear_factors_prim}")
        elif colinear_factors_int:
            if interaction_delim in fac1:
                factors_omit.add(fac1)
            else:
                factors_use.append(fac1)
                factors_omit.update(colinear_factors_int)
        else:
            factors_use.append(fac1)
    """
    origami_data_use = origami_data#.loc[:, factors_use + ["DMS_signal"]]
    formula_rhs = " + ".join(factors_use)
    for formula_lhs, formula_label in formulas.items():
        results = ols(f"{formula_lhs} ~ {formula_rhs}", data=origami_data_use).fit()
        result_label = f"distance-GC vs {formula_label}"
        model_results[result_label] = results
        plot_name = f"model_dist-gc_{origami}_{formula_label.replace(' ', '_')}"
        model_plots(origami_data_use["DMS_signal"], results, f"{plot_name}.pdf")
        with open(f"{plot_name}_summary.txt", "w") as f:
            f.write(str(results.summary()))
    return origami_data_use


def run_regression_dists_gc_params(base_info, origami_mus, params, model_results):
    dist_funcs = get_funcs_exp(params)
    gc_label = "GC"
    gc_func = get_funcs_exp_sym({gc_label: params[gc_label]}, non_zero=True)[gc_label]
    dist_matrix = daedalus.get_coeffs_matrix(base_info, dist_funcs)
    weighted_gc = daedalus.get_weighted_gc(base_info, gc_func)
    seg_length = base_info["SegmentLength"]
    sequence = base_info["Base"] == "C"
    origami_data = sequence.to_frame().join([weighted_gc, seg_length, dist_matrix, origami_mus], how="inner")
    assert origami_data.columns[-1] == "1"
    origami_data.columns = ["C"] + list(origami_data.columns[1: -1]) + [dms_col]
    col_min = 1e-7
    """
    for p1, p2 in itertools.combinations(origami_data.columns, 2):
        if not (("5" in p1 and "5" in p2) or ("3" in p1 and "3" in p2) or dms_col in [p1, p2]):
            interaction = origami_data[p1] * origami_data[p2]
            interaction_max = interaction.max()
            if interaction_max >= col_min:
                if ("5" in p1 and "3" in p2) or ("3" in p1 and "5" in p2):
                    interaction /= interaction_max
                origami_data[f"{p1}{interaction_delim}{p2}"] = interaction
    """
    zero_thresh = 1e-4
    zero_dms = origami_data["DMS_signal"] < zero_thresh
    origami_data["DMS_signal"] = origami_data["DMS_signal"] + zero_dms * zero_thresh
    return origami_data, run_regression_dists_gc_dataframe(origami_data, model_results)


def run_regression_dists_gc(base_info, sample_mus, model_results):
    gc_label = "GC"
    param_names = daedalus.get_formatted_features(base_info)
    x_init = np.array([0.5, 0.5])
    #x_init = np.array([1.0, 1.0])
    def x_to_params(x):
        params = {name: x[0] for name in param_names}
        params[gc_label] = x[1]
        return params
    """
    def objective(x):
        params = x_to_params(x)
        run_regression_dists_gc_params(base_info, sample_mus, params, model_results)
        rsquared = results["log DMS"].rsquared
        return -rsquared
    eps = 1e-6
    bounds = [(0 + eps, 1 - eps) for param in param_names]
    print("optimizing regression parameters")
    #x_opt = minimize(objective, x_init, bounds=bounds, tol=1e-6, method="Nelder-Mead")
    grid = np.linspace(1/np.e, 1/np.e, 1)
    r2s = np.zeros((10, 10))
    for x1, x2 in itertools.product(range(1), repeat=2):
        r2 = objective([grid[x1], grid[x2]])
        r2s[x1, x2] = r2
        print(x1, x2, r2)
    print(r2s)
    """
    params = x_to_params(x_init)
    return run_regression_dists_gc_params(base_info, sample_mus, params, model_results)


def calc_corrs_means_sliding(mus1, mus2, width):
    assert mus1.shape == mus2.shape
    assert np.all(mus1.index == mus2.index)
    idx_min = np.min(mus1.index)
    idx_max = np.max(mus1.index)
    windows = [(i, i + width - 1) for i in range(idx_min, idx_max - width + 2)]
    index = pd.MultiIndex.from_tuples(windows)
    index.names = ("first", "last")
    means1 = pd.Series([np.mean(mus1.loc[wi: wf]) for wi, wf in windows], index=index)
    means2 = pd.Series([np.mean(mus2.loc[wi: wf]) for wi, wf in windows], index=index)
    corrs = pd.Series([pearsonr(mus1.loc[wi: wf], mus2.loc[wi: wf])[0] for wi, wf in windows], index=index)
    return means1, means2, corrs


def plot_corr_signal_sliding(mus1, mus2, width, plot_file):
    aspect = 3.0
    means1, means2, corrs = calc_corrs_means_sliding(mus1, mus2, width)
    corrs.to_csv(f"{plot_file}_PCC.tsv", sep="\t")
    means1.to_csv(f"{plot_file}_DMS-1.tsv", sep="\t")
    means2.to_csv(f"{plot_file}_DMS-2.tsv", sep="\t")
    centers = np.asarray((corrs.index.get_level_values("first") + corrs.index.get_level_values("last")) // 2.0, dtype=int)
    fig, ax = plt.subplots()
    ax.plot(centers, corrs)
    ax.set_ylim((-1.0, 1.0))
    ax.set_aspect(len(centers) / 2.0 / aspect)
    ax.set_xlabel("Position")
    ax.set_ylabel("PCC")
    ax.set_title(plot_file)
    plt.savefig(f"{plot_file}_PCC.pdf")
    plt.close()
    for i, means in enumerate([means1, means2], start=1):
        fig, ax = plt.subplots()
        ax.plot(centers, means)
        ax.set_ylim((0.0, 0.1))
        ax.set_aspect(len(centers) / 0.1 / aspect)
        ax.set_xlabel("Position")
        ax.set_ylabel("DMS reactivity")
        ax.set_title(plot_file)
        plt.savefig(f"{plot_file}_DMS-{i}.pdf")
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(means, corrs)
        ax.set_xlim((0.0, 0.1))
        ax.set_ylim((-1.0, 1.0))
        ax.set_aspect(0.05)
        ax.set_xlabel("DMS reactivity")
        ax.set_ylabel("Correlation")
        ax.set_title(plot_file)
        plt.savefig(f"{plot_file}_DMS-PCC-{i}.pdf")
        plt.close()
    return centers, corrs


def model_plots(values, model, fname):
    fig, axs = plt.subplots(1, 2)
    min_coord = min(min(model.fittedvalues),
            min(model.fittedvalues + model.resid))
    max_coord = max(max(model.fittedvalues),
            max(model.fittedvalues + model.resid))
    axs[0].scatter(model.fittedvalues, model.fittedvalues + model.resid, s=0.5)
    axs[0].plot([min_coord, max_coord], [min_coord, max_coord], c="gray")
    axs[0].set_xlabel("Fitted")
    axs[0].set_ylabel("Actual")
    axs[0].set_xlim((min_coord, max_coord))
    axs[0].set_ylim((min_coord, max_coord))
    axs[0].set_aspect(1)
    axs[1].scatter(model.fittedvalues, model.resid, s=0.5)
    axs[1].plot([min_coord, max_coord], [min_coord, max_coord], c="gray")
    axs[1].set_xlabel("Fitted")
    axs[1].set_ylabel("Residual")
    axs[1].set_xlim((min_coord, max_coord))
    min_coord = min(model.resid)
    max_coord = max(model.resid)
    axs[1].set_ylim((min_coord, max_coord))
    axs[1].set_aspect(1)
    plt.tight_layout()
    plt.savefig(fname)
    if "dist-gc" in fname and "log_DMS" in fname:
        print("Model", fname)
        print("R^2 :", np.corrcoef(model.fittedvalues, model.fittedvalues + model.resid)[0, 1]**2)
        print("RMSD:", np.sqrt(np.mean(model.resid**2)))
    plt.close()


def winsor_norm(dataset, percentile):
    pct_val = np.percentile(dataset, percentile)
    if np.isclose(pct_val, 0.0):
        raise ValueError("Percentile is zero")
    if pct_val < 0.0:
        raise ValueError("Percentile is negative")
    winsorized = np.minimum(dataset / pct_val, 1.0)
    return winsorized


def validate_23S_mus(sample_mus):
    mus_23S = sample_mus["23S"]
    mus_23S.to_csv("23S_mus_normalized.tsv", sep="\t")
    mus_23S_simon = pd.read_excel("../Validation/Simon-et-al_23S.xlsx", index_col="Position")["Combined"]
    # Adjust index of our dataset to match that of Simon et al. (which uses the full 23S rRNA)
    offset = 29
    mus_23S = mus_23S.reindex(mus_23S_simon.index - offset, copy=True)
    mus_23S.index += offset
    # Drop any index with a missing value
    keep = np.logical_not(np.logical_or(np.isnan(mus_23S), np.isnan(mus_23S_simon)))
    mus_23S_keep = mus_23S.loc[keep]
    mus_23S_simon_keep = mus_23S_simon.loc[keep]
    # Apply 90% winsorization and normalization to mus_23S so it matches the processing in Simon et al.
    winsor_pct = 90.0
    mus_23S_keep = winsor_norm(mus_23S_keep, winsor_pct)
    # Compute correlation and scatterplot
    pearson_corr = pearsonr(mus_23S_keep, mus_23S_simon_keep)
    print("PCC:", pearson_corr)
    fig, ax = plt.subplots()
    ax.scatter(mus_23S_keep, mus_23S_simon_keep)
    ax.set_aspect(1.0)
    ax.set_xlabel("23S scaffold")
    ax.set_ylabel("23S in E. coli")
    plt.savefig("23S_mus_scatter.pdf")
    plt.close()
    na_val = -2.0
    diffs_23S = (mus_23S_keep.reindex(mus_23S_simon.index) - mus_23S_simon).fillna(na_val)
    diffs_23S.to_csv("23S_diffs.tsv", sep="\t", header=False)


if __name__ == "__main__":
    redo_dms_plots = 0
    redo_vs_scaf_plots = 0
    redo_ss_plots = 0
    # Remove existing plots to start with a fresh slate.
    if redo_dms_plots and redo_vs_scaf_plots and redo_ss_plots:
        os.system(f"rm {proj_dir}/Analysis/*pdf")
        os.system(f"rm {proj_dir}/Analysis/*png")
    # Load the sample IDs.
    sample_data = read_sample_data()
    for sample_name in samples_exclude:
        sample_data = sample_data.drop(sample_name, axis=0)
    # Load the DMS mutation rates.
    scaf_groups = dict()
    sample_mus = dict()
    rre_mus = dict()
    for sample in sample_data.index:
        scaffold = sample_data.loc[sample, "Scaffold"]
        if scaffold not in scaf_groups:
            scaf_groups[scaffold] = list()
        scaf_groups[scaffold].append(sample)
        sample_mus[sample], rre_mus[sample] = get_sample_mus(sample_data, sample)
    rre_mus = {(k, c): pd.DataFrame.from_dict({sample: mus[k][c] for sample, mus in rre_mus.items()}) for k in range(1, n_rre_clusters + 1) for c in rre_k_cluster_names[k]}
    # Load the scaffold sequences.
    scaf_seqs = dict()
    for seq_name in sample_data["Scaffold"]:
        if seq_name not in scaf_seqs:
            _, scaf_seqs[seq_name] = read_fasta(get_fasta_full_name(seq_name))
    scaf_seqs_df = {scaf: pd.Series(list(seq), index=range(1, len(seq) + 1)) for scaf, seq in scaf_seqs.items()}
    # Compute the DMS signal ratios wrt 23S RRE.
    rre_standard = "23S"
    rre_means = rre_mus[1, "5-stem"].mean(axis=0)
    sig_ratios = rre_means / rre_means[rre_standard]
    # Normalize the DMS signals on the origami using the ratios.
    sample_mus = {sample: mus / sig_ratios.loc[sample] for sample, mus in sample_mus.items()}
    validate_23S_mus(sample_mus)
    for k, c in rre_mus:
        # Compute the correlations over RRE controls.
        rre_corr = rre_mus[k, c].corr()
        rre_corr.to_csv("RRE_corr.txt", sep="\t")
        rre_mus[k, c] /= sig_ratios
        rre_mus[k, c].to_csv(f"rre_mus_{k}-{c}.tsv", sep="\t")
    # Fit a beta distribution to the paired and unpaired DMS signals for the RRE.
    get_lodds_unpaired = fit_dist_to_rre_clusters(rre_mus, rre_standard)
    # Collect all DMS signals into one dataframe.
    all_mus = pd.DataFrame([
            [
                sample_data.loc[sample, "Scaffold"],
                sample,
                idx,
                mu
            ]
            for sample, mus in sample_mus.items()
            for idx, mu in mus.items()
    ], columns=["Scaffold", "Sample", "Position", "DMS Signal"])
    if redo_dms_plots:
        # Plot the DMS signals for all origamis.
        sns.boxplot(data=all_mus, x="Sample", y="DMS Signal", hue="Scaffold", width=0.7, dodge=False, whis=2.0, linewidth=0.5, fliersize=0.5)
        plt.yscale("log")
        plt.ylim((0.0, np.max(all_mus["DMS Signal"]) * 1.05))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("dms_signals_box.pdf")
        plt.close()
    # Initialize data to collect from each origami.
    origamis = dict()  # name of scaffold: names of origamis with that scaffold
    scaffolds = dict()  # name of origami: name of scaffold
    delta_mus_medians = dict()  # name of origami: median quotient of scaffold to origami DMS among all bases
    delta_mus_p_values = dict()  # name of origami: p-value that its signal is different from scaffold (Wilcoxon signed rank test)
    mus_medians_delta = dict()  # name of origami: quotient of median scaffold signal and median origami signal 
    origamis_dir = os.path.join(proj_dir, "Origamis_singleXO")
    all_origamis_data = list()
    scaf_chain = "A"
    origami_analyze_features = ["rT55", "rT66v2", "rT77", "rO66v2", "rPB66v2"]
    base_info = dict()
    model_results = dict()
    origami_data = dict()
    origami_segments = dict()
    formulas = {"DMS_signal": "raw DMS", "np.log10(DMS_signal)": "log DMS", "get_lodds_unpaired(DMS_signal)": "log odds unpaired"}
    #formulas = {"np.log10(DMS_signal)": "log DMS"}
    for scaf, group in scaf_groups.items():
        group_df = pd.DataFrame.from_dict({x: sample_mus[x] for x in group})
        if redo_vs_scaf_plots:
            # Plot all the origamis in the group vs each other.
            print(group_df.corr()**2)
            sns.pairplot(group_df)
            plt.savefig(f"{scaf}_group.pdf")
            plt.close()
        group_df.to_csv(f"mus_{scaf}.tsv", sep="\t")
        # Predict structures for the scaffold.
        output_prefix = os.path.join(analysis_dir, f"{scaf}")
        seq = scaf_seqs[scaf]
        ct_file = f"{output_prefix}.ct"
        if not os.path.isfile(ct_file):
            predict_structure(scaf, seq, sample_mus[scaf], output_prefix,
                    program="Fold", normbases=0.05, overwrite=True, queue=False)
        # Plot a histogram of the DMS signals.
        _, pairs, unpaired, seq_ct = read_ct_file_single(ct_file, multiple=0)
        assert seq_ct == seq
        if redo_dms_plots:
            hist_file = f"{output_prefix}_hist.pdf"
            mu_histogram_paired(hist_file, sample_mus[scaf], unpaired,
                    bin_width=0.001)
        # Fit beta distributions to each structure.
        labels = [{True: "unpaired", False: "paired"}[is_unp]
                for i, is_unp in enumerate(unpaired, start=1)
                if i in sample_mus[scaf].index]
        get_beta_prob = generate_beta_prob_function(*fit_beta_to_prob_labels(
                sample_mus[scaf], labels))
        get_prob_unpaired = generate_beta_prob_unpaired_function(get_beta_prob)
        get_lodds_unpaired = generate_beta_lodds_unpaired_function(get_beta_prob)
        # Save a file of the mus for all of the origamis and the scaffold.
        pd.DataFrame.from_dict({x: sample_mus[x] for x in group})
        # Plot each origami vs its scaffold.
        origamis[scaf] = [x for x in group if x != scaf]
        for origami in origamis[scaf]:
            scaffolds[origami] = scaf
            print(origami)
            # Plot DMS signals of each origami vs its scaffold. 
            if redo_vs_scaf_plots:
                fig, ax = plt.subplots()
                ax.scatter(sample_mus[origami], sample_mus[scaf], s=0.5)
                ax.set_aspect("equal")
                plt.xlim((0, 0.5))
                plt.ylim((0, 0.5))
                plt.xlabel(origami)
                plt.ylabel(scaf)
                plt.savefig(f"{scaf}_vs_{origami}_scatter.pdf")
                plt.close()
                fig, ax = plt.subplots()
                origami_boxplot_data = pd.DataFrame({scaf: sample_mus[scaf], origami: sample_mus[origami]})
                data_min = 1E-4
                origami_boxplot_data[origami_boxplot_data < data_min] = data_min
                flierprops = dict(marker='o', markerfacecolor="gray", markersize=2,
                                  linestyle='none', markeredgecolor="gray")
                sns.boxplot(data=origami_boxplot_data, flierprops=flierprops, linewidth=1.0)
                ax.set_ylim((data_min/2.0, 1E0))
                ax.set_aspect(1.5)
                ax.set_yscale("log")
                ax.set_ylabel("DMS Reactivity")
                plt.savefig(f"{scaf}_vs_{origami}_box.pdf")
                plt.close()
            # Plot the correlation of the DMS signals in the origami vs. the scaffold as a sliding window.
            sliding_plot = f"{scaf}_vs_{origami}_sliding"
            sliding_width = 20
            scaf_corrs_centers, scaf_corrs = plot_corr_signal_sliding(sample_mus[scaf], sample_mus[origami], sliding_width, sliding_plot)
            # Compute the median change in DMS signal for the origami wrt the scaffold.
            delta_mus = sample_mus[scaf] / sample_mus[origami]
            delta_mus_medians[origami] = np.nanmedian(delta_mus)
            mus_medians_delta[origami] = np.median(sample_mus[scaf]) / np.median(sample_mus[origami])
            # Compute the significance of the difference using Wilcoxon signed-rank test.
            w_stat, p_value = wilcoxon(sample_mus[origami], sample_mus[scaf])
            delta_mus_p_values[origami] = p_value
            if origami not in origami_analyze_features:
                # Do not analyze the feature-DMS relationship.
                continue
            # Get the feature feature of each base in the origami.
            origami_dir = os.path.join(origamis_dir, sample_data.loc[origami, "Directory"])
            edges, g_up, g_dn, g_ax, base_info[origami], dssr_info = ariadne.analyze_design(origami_dir, compute_bond_lengths=False, clobber=True)
            # Remove staples, leaving only scaffold bases.
            base_info[origami] = base_info[origami].loc[base_info[origami].loc[:, "PDB chain"] == scaf_chain]
            # Get all the segments in the origami.
            segments = daedalus.get_segments(base_info[origami], g_up, g_dn)
            segments["Origami"] = origami
            origami_segments[origami] = segments
            #FIXME adjusting the sequence is necessary for origamis for which I don't have the DAEDALUS output with the right scaffold sequence -- if all were correct then this step would be unnecessary
            base_info[origami]["Base"] = scaf_seqs_df[scaf]
            model_results[origami] = dict()
            # Run linear regression using the feature labels and sequence.
            features_levels = run_regression_features_seq(base_info[origami], sample_mus[origami], model_results[origami], all_origamis_data, origami)
            # Run linear regression using the distance to each feature and the GC content.
            origami_data[origami] = run_regression_dists_gc(base_info[origami], sample_mus[origami], model_results[origami])[0]
            if redo_ss_plots:
                print("redoing plots")
                # Draw a secondary structure diagram.
                secondary_structure_fname = f"ss_{origami}.pdf"
                dms_min = 1e-4
                if origami == "rO66v1":
                    colors = pd.Series((scaf_corrs.values + 1.0) / 2.0, index=scaf_corrs_centers)
                else:
                    colors = sample_mus[origami]
                    colors.loc[colors < dms_min] = dms_min
                    colors = np.log10(colors)
                    colors = (colors - colors.min()) / (colors.max() - colors.min())
                if not os.path.isfile(secondary_structure_fname):
                    # This takes about one minute per plot.
                    plots.secondary_structure_signal(secondary_structure_fname, edges, g_up, g_dn, g_ax, base_info[origami], colors)
    plt.close()

    # Count the number of times each type of segment appears.
    origami_segments = pd.concat(list(origami_segments.values()), axis=0, ignore_index=True)
    f5_vals = ["scaf_scafDX_3", "scaf_stapDX_3", "scaf_stapSX_3", "scaf_vertex_3", "scaf_stapEND3_0", "scaf_stapSXEND3_0"]
    f3_vals = ["scaf_scafDX_5", "scaf_stapDX_5", "scaf_stapSX_5", "scaf_vertex_5", "scaf_stapEND5_0", "scaf_stapSXEND5_0"]
    segment_props = {prop: pd.DataFrame(index=f5_vals, columns=f3_vals, dtype=int) for prop in ["count", "avg_len", "avg_dms"]}
    segment_data = pd.DataFrame(index=origami_segments.index,
            columns=["length", "GC", "DMS_signal"], dtype=float)
    dms_min = 1e-4
    for base in "NAC":
        for f5, f3 in itertools.product(f5_vals, f3_vals):
            segs_fs = origami_segments[np.logical_and(origami_segments["Feature5"] == f5, origami_segments["Feature3"] == f3)]
            count = segs_fs.shape[0]
            if count > 0:
                lengths = segs_fs["Seg3"] - segs_fs["Seg5"] + 1
                dms_segs = list()
                for seg in segs_fs.index:
                    seg5 = segs_fs.loc[seg, "Seg5"]
                    seg3 = segs_fs.loc[seg, "Seg3"]
                    origami = None
                    dms_seg = list()
                    for position in range(seg5, seg3 + 1):
                        origami = segs_fs.loc[seg, "Origami"]
                        if position in origami_data[origami].index:
                            if base == "N" or (base == "C") == origami_data[origami].loc[position, "C"]:
                                dms = np.log10(max(origami_data[origami].loc[position, "DMS_signal"], dms_min))
                                dms_seg.append(dms)
                    seq = scaf_seqs[scaffolds[origami]][seg5 - 1: seg3]
                    gc_content = (seq.count("G") + seq.count("C")) / len(seq)
                    segment_data.loc[seg, "length"] = lengths.loc[seg]
                    segment_data.loc[seg, "GC"] = gc_content
                    if len(dms_seg) > 0:
                        segment_data.loc[seg, "DMS_signal"] = sum(dms_seg) / len(dms_seg)
                    else:
                        segment_data.loc[seg, "DMS_signal"] = np.nan
                    dms_segs.extend(dms_seg)
                avg_len = lengths.mean()
                avg_dms = sum(dms_segs) / len(dms_segs)
            else:
                avg_len = np.nan
                avg_dms = np.nan
            segment_props["count"].loc[f5, f3] = count
            segment_props["avg_len"].loc[f5, f3] = avg_len
            segment_props["avg_dms"].loc[f5, f3] = avg_dms

        for prop, data in segment_props.items():
            if prop == "avg_dms":
                vmin, vmax = -2.4, -1.2
            else:
                vmin, vmax = data.min().min(), data.max().max()
            sns.heatmap(data, cmap="plasma", square=True, vmin=vmin, vmax=vmax)
            plt.savefig(f"segment_props_{base}_{prop}.pdf")
            plt.close()
        seg_length_dms_means = pd.Series({int(round(length)): segment_data.loc[segment_data["length"] == length, "DMS_signal"].mean() for length in sorted(set(segment_data["length"])) if not np.isnan(length)})
    
        avg_len = segment_props["avg_len"].values.flatten("C")
        avg_dms = segment_props["avg_dms"].values.flatten("C")
        plt.close()
        markers5 = {"scaf_scafDX_3": "x", "scaf_stapDX_3": "+", "scaf_stapSX_3": "o", "scaf_vertex_3": "*", "scaf_stapEND3_0": "^", "scaf_stapSXEND3_0": "s", "scaf_scafEND5_0": "v"}
        colors3 = {"scaf_scafDX_5": "#5ab4e5", "scaf_stapDX_5": "#0a72ba", "scaf_stapSX_5": "#00a875", "scaf_vertex_5": "#f7941d", "scaf_stapEND5_0": "#f15a22", "scaf_stapSXEND5_0": "#da6fab", "scaf_scafEND3_0": "#aaaaaa"}
        jitter_std = 0.15
        jitter_max = 0.30
        for p1, p2 in itertools.combinations(segment_data.columns, 2):
            labels = set()
            fig, ax = plt.subplots()
            for seg in segment_data.index:
                x = segment_data.loc[seg, p1]
                y = segment_data.loc[seg, p2]
                x_jitter = np.random.randn() * jitter_std
                if x_jitter > 0:
                    x_jitter = min(x_jitter, jitter_max)
                else:
                    x_jitter = max(x_jitter, -jitter_max)
                if p2 == "DMS_signal":
                    y = np.power(10.0, y)
                    ax.set_yscale("log")
                label = None
                f5 = origami_segments.loc[seg, "Feature5"]
                f3 = origami_segments.loc[seg, "Feature3"]
                marker = markers5[f5]
                color = colors3[f3]
                if f5 not in labels:
                    label = f5
                if f3 not in labels:
                    label = f3
                if label is not None:
                    labels.add(label)
                ax.scatter(x + x_jitter, y, s=16.0, marker=marker, c=color, label=label)
            # add means
            if p2 == "DMS_signal":
                y = np.power(10.0, seg_length_dms_means)
            else:
                y = seg_length_dms_means
            ax.scatter(y.index, y, c="black", marker="D")
            print(seg_length_dms_means)
            ax.set_xlabel(p1)
            ax.set_ylabel(p2)
            if p1 == "GC":
                ax.set_xlim((0.0, 1.0))
            elif p1 == "length":
                ax.set_xlim((6, 24))
                ax.set_xticks(list(range(6, 24 + 1, 2)))
            if p2 == "DMS_signal":
                ax.set_ylim((1e-3, 1e0))
            plt.legend()
            plt.savefig(f"segment_data_{base}_{p1}_{p2}.pdf")
            plt.close()

    all_origamis = list(itertools.chain(*origamis.values()))
    origami_data_use = dict()
    origami_data_r2_use = dict()
    for base in "NAC":
        print("Analysis of all origamis,", base)
        origami = f"v2-{base}"
        origami_data[origami] = pd.concat([origami_data[ori] for ori in all_origamis if ori in origami_analyze_features], axis=0)
        if base in "AC":
            origami_data[origami] = origami_data[origami].loc[origami_data[origami]["C"] == (base == "C")].drop(columns="C")
        # Run linear regression using the distance-GC model for all v2 origamis at once.
        origami_data[origami].to_csv(f"origami_data_{origami}.csv")
        model_results[origami] = dict()
        origami_data_use[origami] = run_regression_dists_gc_dataframe(origami_data[origami], model_results[origami])
        print("all data:", origami_data[origami])
        print("use data:", origami_data_use[origami])
        origami_data_use[origami].to_csv(f"origami_data_{origami}_use.csv")
        origami_data_r2_use[origami] = origami_data_use[origami].corr()**2
        origami_data_r2_use[origami].to_csv(f"origami_data_{origami}_use_R2s.csv")
    """
    # Volcano plots
    volcano_xmin = -3.0
    volcano_xmax = 3.0
    for predictor in ["distance-GC"]:
        method = f"{predictor} vs log DMS"
        for origami, methods in model_results.items():
            params = methods[method].params
            ses = methods[method].bse
            log_pvals = -np.log10(methods[method].pvalues)
            is_intercept = params.index == "Intercept"
            params_noint = params[~is_intercept]
            log_pvals_noint = log_pvals[~is_intercept]
            ses_noint = ses[~is_intercept]
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            def get_volcano_color(param):
                if interaction_delim in param:
                    terms = param.split(interaction_delim)
                else:
                    terms = [param]
                r = 0.7 if any(["End" in t for t in terms]) else 0.3
                g = 0.7 if any([t in ["C", "C[T.True]", "WeightedGC"] for t in terms]) else 0.3
                b = 0.7 if any(["Crossover" in t for t in terms]) else 0.3
                return (r, g, b)
            colors = [get_volcano_color(param) for param in params_noint.index]
            p_cutoff = -np.log10(0.05)
            for x, p, xe, c in zip(params_noint, log_pvals_noint, ses_noint, colors):
                for ax in [ax1, ax2]:
                    ax.errorbar(x, p, xerr=xe, elinewidth=0.7, capthick=0.7, capsize=2.0, fmt="o", markersize=3.0, color=c, ecolor=[0.8, 0.8, 0.8])
            ax1.set_ylim((200.0, 225.0))
            ax2.set_ylim((0.0, 25.0))
            ax1.spines.bottom.set_visible(False)
            ax1.spines.right.set_visible(False)
            ax2.spines.right.set_visible(False)
            ax1.spines.top.set_visible(False)
            ax2.spines.top.set_visible(False)
            ax1.tick_params(labeltop=False)
            ax2.xaxis.tick_bottom()
            fig.subplots_adjust(hspace=0.0)
            ax2.set_xlim((-1.0, 1.0))
            ax2.set_xlabel("Effect size [log10(DMS)]")
            ax2.set_ylabel("-log10(P-value)")
            ax1.set_title(origami)
            ax2.set_aspect(0.040)
            ax1.set_aspect(0.008)
            fig.set_size_inches(10.0, 10.0)
            fname = f"{origami}_{predictor}_volcano"
            plt.savefig(fname + ".pdf")
            plt.close()
            with open(fname + ".tsv", 'w') as f:
                f.write("Parameter\tEst\tSE\tlogP\n")
                text = "\n".join([f"{i.strip()}\t{x}\t{e}\t{p}" for i, x, e, p in zip(params.index, params, ses, log_pvals)])
                f.write(text)
    """

    # Plot the signals over features for all non-skipped origamis.
    all_origamis_data = pd.concat(all_origamis_data, axis=0)
    all_origamis_data.to_csv("all_origamis_data.csv")
    fig, ax = plt.subplots()
    flierprops = dict(marker='o', markerfacecolor="gray", markersize=2,
            linestyle='none', markeredgecolor="gray")
    dms = np.log10(all_origamis_data["DMS_signal"])
    ylim = (-4, 0)
    sns.boxplot(data=pd.DataFrame({"Feature": all_origamis_data["Feature"], "DMS_signal": dms}),
                x="Feature", y="DMS_signal", order=features_levels, flierprops=flierprops)
    #ax.set_yscale("log")
    #ax.set_ylim((1E-4, 1E0))
    ax.set_ylim(ylim)
    ax.set_aspect(2)
    plt.xticks(rotation=90)
    fig.set_size_inches(8, 8)
    plt.savefig(f"dms_sig_vs_feature{'_group' * int(group)}.pdf")
    plt.close()
    all_origamis_data_means = pd.Series({
        feature: np.log10(all_origamis_data.loc[all_origamis_data["Feature"] == feature, "DMS_signal"]).mean()
        for feature in features_levels
    })
    all_origamis_data_medians = pd.Series({
        feature: np.log10(all_origamis_data.loc[all_origamis_data["Feature"] == feature, "DMS_signal"]).median()
        for feature in features_levels
    })
    all_origamis_data_stdev = pd.Series({
        feature: np.log10(all_origamis_data.loc[all_origamis_data["Feature"] == feature, "DMS_signal"]).std()
        for feature in features_levels
    })
    all_origamis_data_ns = pd.Series({
        feature: (all_origamis_data["Feature"] == feature).sum()
        for feature in features_levels
    })
    all_origamis_data_means.to_csv("dms_sig_vs_feature_mean.tsv", sep="\t")
    all_origamis_data_medians.to_csv("dms_sig_vs_feature_median.tsv", sep="\t")
    diff_featureless = pd.DataFrame(index=features_levels, columns=features_levels, dtype=float)
    diff_featureless.loc[:, :] = 0.0
    for feature1, feature2 in itertools.combinations(features_levels, 2):
        feature1_signals = all_origamis_data.loc[all_origamis_data["Feature"] == feature1, "DMS_signal"]
        feature2_signals = all_origamis_data.loc[all_origamis_data["Feature"] == feature2, "DMS_signal"]
        result = mannwhitneyu(feature1_signals, feature2_signals)
        u, p_value = result.statistic, result.pvalue
        diff_featureless.loc[feature1, feature2] = -np.log10(p_value)
        diff_featureless.loc[feature2, feature1] = -np.log10(np.abs(np.median(feature1_signals) / np.median(feature2_signals))) #/ (len(feature1_signals) * len(feature2_signals))
    sns.heatmap(diff_featureless, cbar=True, square=True)
    plt.savefig("diff_featureless.pdf")
    plt.close()
    diff_featureless.to_csv("diff_featureless.tsv", sep="\t")
    
    # For each feature, plot its effects vs the difference in DMS reactivity compared to no-feature.
    fig, ax = plt.subplots()
    feature_mean_diffs = all_origamis_data_means.drop("none") - all_origamis_data_means["none"]
    feature_mean_diffs_sem = (all_origamis_data_stdev / np.sqrt(all_origamis_data_ns - 1)).drop("none")
    index = feature_mean_diffs.index
    results = model_results["v2-N"]["distance-GC vs log DMS"]
    effects = results.params.loc[index]
    effects_sem = results.bse.loc[index]
    print("x ", feature_mean_diffs)
    print("xe", feature_mean_diffs_sem)
    print("y ", effects)
    print("ye", effects_sem)
    ax.errorbar(feature_mean_diffs, effects, xerr=feature_mean_diffs_sem, yerr=effects_sem, fmt="none", capsize=2.5, ecolor=(0.7, 0.7, 0.7), elinewidth=0.5)
    ax.scatter(feature_mean_diffs, effects)
    for i in index:
        x = feature_mean_diffs.loc[i]
        y = effects.loc[i]
        ax.text(x, y, i)
    cmin, cmax = -0.2, 1.0
    plt.plot([cmin, cmax], [cmin, cmax], linestyle="--", color="gray")
    ax.set_xlim((cmin, cmax))
    ax.set_ylim((cmin, cmax))
    ax.set_xlabel("mean diff from none")
    ax.set_ylabel("effect parameter")
    ax.set_aspect(1.0)
    plt.savefig("effect_vs_mean_diff.pdf")
    plt.close()
    
    # Also plot rT66v2 with and w/o staple 10.
    rT66v2_df = pd.concat([sample_mus["rT66v2"], sample_mus["rT66v2-10"]], axis=1)
    rT66v2_df.columns = ["rT66v2", "rT66v2-10"]
    rT66v2_df["Target"] = rT66v2_df.index.isin(rT66v2_stap10_target)
    plt.scatter(rT66v2_df["rT66v2"], rT66v2_df["rT66v2-10"], c=rT66v2_df["Target"])
    plt.savefig("rT66v2_scatter.pdf")
    plt.close()
    total_corr = np.corrcoef(rT66v2_df["rT66v2"], rT66v2_df["rT66v2-10"])[0, 1]
    target_corr = np.corrcoef(rT66v2_df.loc[rT66v2_df["Target"], "rT66v2"],
                              rT66v2_df.loc[rT66v2_df["Target"], "rT66v2-10"])[0, 1]
    other_corr = np.corrcoef(rT66v2_df.loc[~rT66v2_df["Target"], "rT66v2"],
                              rT66v2_df.loc[~rT66v2_df["Target"], "rT66v2-10"])[0, 1]
    target_stap = rT66v2_df.loc[rT66v2_df['Target'], 'rT66v2']
    target_no_stap = rT66v2_df.loc[rT66v2_df['Target'], 'rT66v2-10']
    w_target, p_target = wilcoxon(target_stap, target_no_stap)
    other_stap = rT66v2_df.loc[~rT66v2_df['Target'], 'rT66v2']
    other_no_stap = rT66v2_df.loc[~rT66v2_df['Target'], 'rT66v2-10']
    w_other, p_other = wilcoxon(other_stap, other_no_stap)
    with open("rT66v2_data.txt", "w") as f:
        f.write(f"""corr total: \t{total_corr}
corr target:\t{target_corr}
corr other: \t{other_corr}
med target + stap:\t{np.median(target_stap)}
med target - stap:\t{np.median(target_no_stap)}
med target diff:  \t{np.median(target_no_stap - target_stap)}
target p-value:   \t{p_target}
med other + stap: \t{np.median(other_stap)}
med other - stap: \t{np.median(other_no_stap)}
med other diff:  \t{np.median(other_no_stap - other_stap)}
other p-value:   \t{p_other}""")
    rT66v2_df = pd.melt(rT66v2_df, id_vars=["Target"], var_name="Origami", value_name="DMS signal")
    sns.violinplot(data=rT66v2_df, x="Target", y="DMS signal", hue="Origami")
    plt.savefig("rT66v2_violin.pdf")
    plt.close()
    r2s = pd.DataFrame.from_dict({origami: {method: res.rsquared
        for method, res in methods.items()}
        for origami, methods in model_results.items()}, orient="index")
    r2s["Origami"] = r2s.index
    r2s = pd.melt(r2s, id_vars=["Origami"], var_name="Method", value_name="R^2")
    sns.barplot(data=r2s, x="Origami", y="R^2", hue="Method")
    plt.xlabel("Origami")
    plt.ylabel("R^2")
    plt.savefig("dms_fits_r2s.pdf")
    plt.close()
    for ax, predictor in zip([axs], ["distance-GC"]):
        method = f"{predictor} vs log DMS"
        params = pd.DataFrame.from_dict({origami: methods[method].params
            for origami, methods in model_results.items()}, orient="index").T
        pvals = pd.DataFrame.from_dict({origami: methods[method].pvalues
            for origami, methods in model_results.items()}, orient="index").T
        pvals_min = pvals.min(axis=1)
        pvals["min"] = pvals_min
        params["min"] = pvals_min
        pvals.sort_values(by="min", axis=0, inplace=True)
        params.sort_values(by="min", axis=0, inplace=True)
        del pvals["min"]
        del params["min"]
        if predictor == "feature-sequence":
            param_pattern = re.compile("\[T\.([^\]]+)\]")
            param_codes = [param_pattern.findall(param) for param in params.index]
        else:
            param_pattern = re.compile("\[T\.([^\]]+)\]")
            param_codes = [[param_pattern.search(field).groups()[0] if param_pattern.search(field) else field for field in param.split(interaction_delim)] for param in params.index]
        new_idx = [" + ".join(code).replace("_", " ").replace("3", "3'").replace("5", "5'") for code in param_codes]
        params.index = new_idx
        pvals.index = new_idx
        keep_tol = 0.05
        keep_cols = np.logical_and(pvals.min(axis=1) < keep_tol, np.logical_not(params.index.isin(["Intercept", ""])))
        #drop_cols = np.logical_or(drop_cols, [" + " in col for col in params.columns])
        params = params.loc[keep_cols, :]
        pvals = pvals.loc[keep_cols, :]
        sns.heatmap(params, square=True, xticklabels=True, yticklabels=True, cmap="seismic", center=0, cbar=True, ax=ax)
        alphas = [1, 0.05, 0.01, 0.001, 0.0]
        significance = pd.DataFrame(index=pvals.index, columns=pvals.columns)
        for i in range(len(alphas) - 1):
            symbol = "*" * i
            significance[np.logical_and(pvals <= alphas[i], pvals > alphas[i + 1])] = symbol
        max_abs_params = np.max(np.max(np.abs(params)))
        for x, col in enumerate(significance.columns):
            for y, row in enumerate(significance.index):
                if abs(params.loc[row, col]) > max_abs_params * 0.3:
                    color = "white"
                else:
                    color = "black"
                ax.text(x + 0.5, y + 0.7, significance.loc[row, col], ha="center", va="center", c=color)
        plt.xticks(rotation=90)
    fig.set_size_inches(8, 8)
    plt.savefig(f"dms_fits_params.pdf")
    plt.close()
    # Plot the ROC curve of DMS reactivities versus the 23S secondary structure from PDB 4V9D
    ss_pdb_ct_file = "23S_CRW-J01695+pk.ct"
    seq_start = -28
    _, pairs_23S, paired_23S, seq_23S = read_ct_file_single(ss_pdb_ct_file, start_pos=seq_start)
    plot_file = "23S_CRW-J01695+pk_roc.pdf"
    plot_data_structure_roc_curve(paired_23S, sample_mus["23S"], plot_file)
    auroc = get_data_structure_agreement("AUROC", paired_23S, sample_mus["23S"])
    print("AUROC 23S:", auroc)

