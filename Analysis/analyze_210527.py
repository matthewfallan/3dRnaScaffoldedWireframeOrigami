from collections import Counter
import itertools
import os
import re
import sys

from Bio import SeqIO
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta, wilcoxon, mannwhitneyu
import seaborn as sns
from statsmodels.formula.api import ols

from ariadne import ariadne, plots
from rouls.seq_utils import read_fasta
from rouls.struct_utils import predict_structure, read_ct_file_single
from rouls.dreem_utils import get_clusters_mu_filename, get_sample_and_run, read_clusters_mu, mu_histogram_paired


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
                assert corr_amp_prev >= 0.9
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
    log_odds = np.log10(x / (1 - x))
    return log_odds


if __name__ == "__main__":
    # Remove existing plots to start with a fresh slate.
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
    for k, c in rre_mus:
        # Compute the correlations over RRE controls.
        rre_corr = rre_mus[k, c].corr()
        rre_corr.to_csv("RRE_corr.txt", sep="\t")
        rre_mus[k, c] /= sig_ratios
        rre_mus[k, c].to_csv(f"rre_mus_{k}-{c}.csv")
    # Fit a beta distribution to the paired and unpaired DMS signals for the RRE.
    get_lodds_unpaired = fit_dist_to_rre_clusters(rre_mus, rre_standard)
    pair_x = pd.Series(np.linspace(0, 1, 1001), index=list(range(1, 1002)))
    p_unpair_rre = get_lodds_unpaired(pair_x)
    plt.plot(pair_x, p_unpair_rre)
    plt.savefig("p_unpair_RRE.pdf")
    plt.close()
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
    # Plot the DMS signals for all origamis.
    sns.boxplot(data=all_mus, x="Sample", y="DMS Signal", hue="Scaffold", width=0.7, dodge=False, whis=2.0, linewidth=0.5, fliersize=0.5)
    plt.ylim((0.0, np.max(all_mus["DMS Signal"]) * 1.05))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("dms_signals_box.pdf")
    plt.close()
    origamis = dict()
    delta_mus_medians = dict()
    delta_mus_p_values = dict()
    mus_medians_delta = dict()
    # Obtain the geometric information for each origami.
    all_origamis = list(itertools.chain(*origamis.values()))
    origamis_dir = os.path.join(proj_dir, "Origamis")
    all_locations = dict()
    all_edges = dict()
    all_gs = dict()
    all_origamis_data = list()
    scaf_chain = "A"
    skip_origamis = ["rT55", "rT77", "rT66v2-10", "rO66v1", "rO44v1"]
    model_results = dict()
    formulas = {"DMS": "raw DMS", "np.log10(DMS)": "log DMS", "get_lodds_unpaired(DMS)": "log odds unpaired"}
    #formulas = {"np.log10(DMS)": "log DMS"}
    for scaf, group in scaf_groups.items():
        # Plot all the origamis in the group vs each other.
        group_df = pd.DataFrame.from_dict({x: sample_mus[x] for x in group})
        print(group_df.corr()**2)
        #sns.pairplot(group_df)
        plt.savefig(f"{scaf}_group.pdf")
        plt.close()
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
        p_unpair_scaf = get_lodds_unpaired(pair_x)
        plt.plot(pair_x, p_unpair_scaf)
        plt.savefig(f"p_unpair_{scaf}.pdf")
        plt.close()
        # Plot each origami vs its scaffold.
        origamis[scaf] = [x for x in group if x != scaf]
        for origami in origamis[scaf]:
            fig, ax = plt.subplots()
            ax.scatter(sample_mus[origami], sample_mus[scaf], s=0.5)
            ax.set_aspect("equal")
            plt.xlim((0, 0.5))
            plt.ylim((0, 0.5))
            plt.xlabel(origami)
            plt.ylabel(scaf)
            plt.savefig(f"{scaf}_vs_{origami}.pdf")
            plt.close()
            # Compute the median change in DMS signal for the origami wrt the scaffold.
            delta_mus = sample_mus[scaf] / sample_mus[origami]
            delta_mus_medians[origami] = np.nanmedian(delta_mus)
            mus_medians_delta[origami] = np.median(sample_mus[scaf]) / np.median(sample_mus[origami])
            # Compute the significance of the difference using Wilcoxon signed-rank test.
            w_stat, p_value = wilcoxon(sample_mus[origami], sample_mus[scaf])
            delta_mus_p_values[origami] = p_value
            if origami in skip_origamis:
                # Skip these for now because Ariadne is not yet able to process odd edge lengths.
                continue
            """
            #FIXME
            if scaf != "23S":
                continue
            """
            # Get the location feature of each base in the origami.
            origami_dir = os.path.join(origamis_dir, sample_data.loc[origami, "Directory"])
            edges, g_up, g_dn, g_ax, base_info, dssr_info = ariadne.analyze_design(origami_dir, compute_bond_lengths=False, clobber=True)
            location_strings = base_info.loc[base_info.loc[:, "PDB chain"] == scaf_chain, "location"]
            base_info["base"] = scaf_seqs_df[sample_data.loc[origami, "Scaffold"]].loc[1: len(base_info["base"])] #FIXME adjusting the sequence is necessary for origamis for which I don't have the DAEDALUS output with the right scaffold sequence -- if all were correct then this step would be unnecessary
            locations_split = location_strings.str.split("_", expand=True)
            locations_split.columns = ["Strand", "Location", "Direction"]
            assert np.all(locations_split["Strand"] == "Scaffold")
            locations = locations_split.apply(lambda base: base["Location"] if str(base["Direction"]) == "0" else f"{base['Location']} {base['Direction']}'", axis=1)
            locations.name = "Location"
            locations_levels = sorted(set(locations))
            default_loc = "Middle"
            default_idx = locations_levels.index(default_loc)
            locations_levels.pop(default_idx)
            locations_levels = [default_loc] + locations_levels
            # Record the structural features.
            all_locations[origami] = locations
            all_edges[origami] = edges
            all_gs[origami] = {"up": g_up, "dn": g_dn, "ax": g_ax}
            # Combine the DMS signals, sequence, and features into one dataframe.
            origami_seq = scaf_seqs_df[scaf]
            origami_seq.name = "Sequence"
            origami_mus = sample_mus[origami]
            origami_mus.name = "DMS"
            origami_data = origami_seq.to_frame().join([locations, origami_mus], how="inner")
            assert np.all(origami_data["Sequence"].isin(dms_bases))
            origami_data["Origami"] = origami
            all_origamis_data.append(origami_data)
            secondary_structure_fname = f"ss_{origami}.png"
            colors = get_prob_unpaired(origami_mus)
            #plots.secondary_structure_signal(secondary_structure_fname, edges, g_up, g_dn, g_ax, base_info, colors)
            model_results[origami] = dict()
            for formula_lhs, formula_label in formulas.items():
                if formula_label in ["log DMS"]:
                    zero_dms = np.isclose(origami_data["DMS"], 0.0)
                    origami_data = origami_data.loc[~zero_dms]
                formula_rhs = "C(Sequence, levels=dms_bases) + C(Location, levels=locations_levels) + C(Sequence, levels=dms_bases) * C(Location, levels=locations_levels)"
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
                model_results[origami][formula_label] = results
                """
                coeffs[name] = lm.params
                conf_ints[name] = lm.conf_int(conf_alpha)
                p_vals[name] = lm.pvalues
                r2s[name] = lm.rsquared
                """
                fitted = results.fittedvalues
                residuals = results.resid
                plt.scatter(fitted, residuals)
                plt.title(origami)
                plt.savefig(f"fr_{origami}_{formula_label.replace(' ', '_')}.png")
                plt.close()
    # Plot the signals over locations for all non-skipped origamis.
    all_origamis_data = pd.concat(all_origamis_data, axis=0)
    exclude_locations = ["ScaffoldEnd 5'", "ScaffoldEnd 3'"]
    locations_levels = [x for x in locations_levels if x not in exclude_locations]
    all_origamis_data = all_origamis_data.loc[all_origamis_data["Location"].isin(locations_levels)]
    fig, ax = plt.subplots()
    sns.boxplot(data=all_origamis_data, x="Location", y="DMS", order=locations_levels)
    ax.set_aspect(10)
    plt.xticks(rotation=90)
    fig.set_size_inches(8, 8)
    plt.savefig("dms_sig_vs_location.pdf")
    plt.close()
    location_featureless = all_origamis_data.loc[all_origamis_data["Location"] == "Middle", "DMS"]
    diff_featureless = dict()
    for location in locations_levels:
        location_signals = all_origamis_data.loc[all_origamis_data["Location"] == location, "DMS"]
        u, p_value = mannwhitneyu(location_signals, location_featureless)
        diff_featureless[location] = p_value
    with open("diff_featureless", "w") as f:
        f.write("\n".join([f"{location}\t{p_value}" for location, p_value in diff_featureless.items()]))
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
        f.write(f"""
corr total: \t{total_corr}
corr target:\t{target_corr}
corr other: \t{other_corr}
med target + stap:\t{np.median(target_stap)}
med target - stap:\t{np.median(target_no_stap)}
med target diff:  \t{np.median(target_no_stap - target_stap)}
target p-value:   \t{p_target}
med other + stap: \t{np.median(other_stap)}
med other - stap: \t{np.median(other_no_stap)}
med other diff:  \t{np.median(other_no_stap - other_stap)}
other p-value:   \t{p_other}
""")
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
    method = "log DMS"
    param_pattern = re.compile("\[T\.([^\]]+)\]")
    params = pd.DataFrame.from_dict({origami: methods[method].params
        for origami, methods in model_results.items()}, orient="index")
    param_codes = [param_pattern.findall(param) for param in params.columns]
    params.columns = [" + ".join(code) for code in param_codes]
    params.drop(labels=[col for col in params.columns if "ScaffoldEnd" in col or col == ""], axis=1, inplace=True)
    print("Params:")
    print(params.T)
    sns.heatmap(params.T, square=True, cmap="seismic", center=0, cbar=True)
    pvals = pd.DataFrame.from_dict({origami: methods[method].pvalues
        for origami, methods in model_results.items()}, orient="index")
    param_codes = [param_pattern.findall(param) for param in pvals.columns]
    pvals.columns = [" + ".join(code) for code in param_codes]
    pvals.drop(labels=[col for col in pvals.columns if "ScaffoldEnd" in col or col == ""], axis=1, inplace=True)
    print("P-values:")
    print(pvals.T)
    alphas = [1, 0.05, 0.01, 0.001, 0.0]
    significance = pd.DataFrame(index=pvals.index, columns=pvals.columns)
    for i in range(len(alphas) - 1):
        symbol = "*" * i
        significance[np.logical_and(pvals <= alphas[i], pvals > alphas[i + 1])] = symbol
    for y, col in enumerate(significance.columns):
        for x, row in enumerate(significance.index):
            plt.text(x + 0.5, y + 0.7, significance.loc[row, col], ha="center", va="center", c="white")
    """
    params["Origami"] = params.index
    params = pd.melt(params, id_vars=["Origami"], var_name="Parameter", value_name="Effect")
    fig, ax = plt.subplots()
    sns.barplot(data=params, x="Parameter", y="Effect", hue="Origami")
    plt.xlabel("Parameter")
    plt.ylabel("Effect on DMS signal")
    ax.set_aspect(4.0)
    """
    fig.set_size_inches(8, 8)
    plt.xticks(rotation=90)
    plt.savefig("dms_fits_params.pdf")
    plt.close()
    """
    conf_alpha = 0.05
    conf_ints = {origami: res.conf_int(conf_alpha)
        for origami, res in model_results["log DMS"].items()}
    conf_ints = pd.DataFrame.from_dict({origami: res.conf_int(conf_alpha)
        for origami, res in model_results["log DMS"].items()}, orient="index")
    param_codes = [param_pattern.findall(param) for param in conf_ints.columns]
    conf_ints.columns = [" + ".join(code) for code in param_codes]
    """

