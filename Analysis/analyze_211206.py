from collections import Counter, defaultdict
import itertools
import math
import os
import random
import re
import sys

sys.path.append("/home/ma629/git")

from Bio import SeqIO
from Bio.SeqUtils import MeltingTemp as mt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta, wilcoxon, mannwhitneyu, pearsonr, spearmanr, ttest_rel, ttest_ind
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

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


exclude_samples = ["rO44v2"]


def geo_mean(x, ignore_nan=True):
    if np.any(np.asarray(x) < 0.0):
        return np.nan
    if np.any(np.isclose(x, 0.0)):
        return 0.0
    if ignore_nan:
        mean = np.nanmean
    else:
        mean = np.mean
    return np.exp(mean(np.log(x)))
    

def read_sample_data():
    sample_file = os.path.join(proj_dir, "samples.txt")
    sample_data = pd.read_csv(sample_file, sep="\t", index_col="Sample")
    sample_data = sample_data.loc[~sample_data.index.isin(exclude_samples)]
    print(sample_data)
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


dms_col = "DMS_signal"


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
    scaf_chain = "A"
    origami_analyze_features = ["rT55", "rT66v2", "rT77", "rO66v2", "rPB66v2"]
    base_info = dict()
    origami_segments = dict()
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
                sns.boxplot(data=np.log10(origami_boxplot_data), flierprops=flierprops, linewidth=1.0)
                ax.set_ylim((data_min/2.0, 1E0))
                ax.set_aspect(1.5)
                #ax.set_yscale("log")
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
            segments["Length"] = segments["Seg3"] - segments["Seg5"] + 1
            assert np.all(segments["Length"] > 0)
            origami_segments[origami] = segments
            # Adjusting the sequence is necessary for origamis for which I don't have the DAEDALUS output with the right scaffold sequence -- if all were correct then this step would be unnecessary
            base_info[origami]["Base"] = scaf_seqs_df[scaf]
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
    
    f5_vals = ["scaf_stapDX_3", "scaf_stapSXEND3_0", "scaf_stapSX_3", "scaf_vertex_3", "scaf_scafDX_3", "scaf_stapEND3_0"]
    f3_vals = ["scaf_scafDX_5", "scaf_stapDX_5", "scaf_stapSX_5", "scaf_stapSXEND5_0", "scaf_stapEND5_0", "scaf_vertex_5"]

    base_col = "Base"
    dms_min = 1e-4
    # Compile a DataFrame for each base in each origami of the upstream and downstream features and the DMS signal.
    origami_segment_bases = dict()
    for origami, segments in origami_segments.items():
        print("origami", origami)
        segment_attrs = pd.DataFrame(index=segments.index, dtype=float)
        seq = scaf_seqs[scaffolds[origami]]
        for seg in segments.index:
            start = segments.loc[seg, "Seg5"]
            end = segments.loc[seg, "Seg3"]
            seq_seg = seq[start - 1: end]
            assert len(seq_seg) > 0
            dms_seg = list()
            data = segments.loc[seg].to_dict()
            positions = list(range(start, end + 1))
            for pos in positions:
                dms = sample_mus[origami].get(pos, np.nan)
                origami_segment_bases[origami, pos] = {**data, dms_col: dms, base_col: seq[pos - 1]}
                dms_seg.append(dms)
            gc_content = np.mean([x in "CG" for x in seq_seg])
            # Predict melting temperature of RNA/DNA duplex
            tm = mt.Tm_NN(seq_seg, nn_table=mt.R_DNA_NN1, Na=300, saltcorr=5)
            attrs = dict()
            for base in "MAC":
                dms_seg_base = [dms if base in ["M", seq_seg[pos]] else np.nan
                        for pos, dms in enumerate(dms_seg)]
                if len(dms_seg_base) > 0:
                    dms_mean = geo_mean(np.maximum(dms_seg_base, dms_min))
                    attrs[f"DMS5{base}"] = dms_seg_base[0]
                    attrs[f"DMS3{base}"] = dms_seg_base[-1]
                else:
                    dms_mean = np.nan
                    attrs[f"DMS5{base}"] = np.nan
                    attrs[f"DMS3{base}"] = np.nan
                if len(dms_seg_base) > 2:
                    dms_mean_interior = geo_mean(np.maximum(dms_seg_base[1: -1], dms_min))
                    attrs[f"DMS5a{base}"] = dms_seg_base[1]
                    attrs[f"DMS3a{base}"] = dms_seg_base[-2]
                else:
                    dms_mean_interior = np.nan
                    attrs[f"DMS5a{base}"] = np.nan
                    attrs[f"DMS3a{base}"] = np.nan
                attrs[f"DMSall{base}"] = dms_mean
                attrs[f"DMSint{base}"] = dms_mean_interior
            attrs["GC"] = gc_content
            attrs["Tm"] = tm
            for attr, value in attrs.items():
                segment_attrs.loc[seg, attr] = value
        origami_segments[origami] = pd.concat([origami_segments[origami], segment_attrs], axis=1)
    origami_segment_bases = pd.DataFrame.from_dict(origami_segment_bases, orient="index")
    origami_segment_bases.to_csv("origami_segment_bases.tsv", sep="\t")

    # Make a plot matrix of showing the mean DMS signal at each position in each type of segment.
    segments_dms = dict()
    segments_base = dict()
    for index in origami_segment_bases.index:
        origami, pos = index
        seg5 = origami_segment_bases.loc[index, "Seg5"]
        seg3 = origami_segment_bases.loc[index, "Seg3"]
        feat5 = origami_segment_bases.loc[index, "Feature5"]
        feat3 = origami_segment_bases.loc[index, "Feature3"]
        length = origami_segment_bases.loc[index, "Length"]
        dms = origami_segment_bases.loc[index, dms_col]
        base = origami_segment_bases.loc[index, base_col]
        seg = seg5, seg3
        feat = feat5, feat3
        if feat not in segments_dms:
            segments_dms[feat] = dict()
            segments_base[feat] = dict()
        if origami not in segments_dms[feat]:
            segments_dms[feat][origami] = dict()
            segments_base[feat][origami] = dict()
        if seg not in segments_dms[feat][origami]:
            segments_dms[feat][origami][seg] = dict()
            segments_base[feat][origami][seg] = dict()
        col = pos - seg5 + 1
        segments_dms[feat][origami][seg][col] = dms
        segments_base[feat][origami][seg][col] = base
    for feat in segments_dms:
        feat_segs_dms = segments_dms[feat]
        feat_segs_base = segments_base[feat]
        for origami in feat_segs_dms:
            origami_segs_dms = feat_segs_dms[origami]
            origami_segs_base = feat_segs_base[origami]
            seg_lengths = list(map(len, origami_segs_dms.values()))
            seg_lengths_uniq = sorted(set(seg_lengths))
            max_len = max(seg_lengths_uniq)
            if len(seg_lengths_uniq) == 0:
                raise ValueError("empty set")
            elif len(seg_lengths_uniq) == 1:
                origami_segs_dms_adjust = origami_segs_dms
                origami_segs_base_adjust = origami_segs_base
            else:
                # Different segments with the same bordering features have different lengths.
                # Adjust the numbering so that everything is numbered as if it's the longest segment.
                origami_segs_dms_adjust = dict()
                origami_segs_base_adjust = dict()
                for seg in origami_segs_dms:
                    seg_dms = origami_segs_dms[seg]
                    seg_base = origami_segs_base[seg]
                    seg_len = len(seg_dms)
                    def adjust_col(col):
                        if col <= seg_len / 2:
                            return col
                        else:
                            return col + max_len - seg_len
                    seg_dms_adjust = {adjust_col(col): dms for col, dms in seg_dms.items()}
                    seg_base_adjust = {adjust_col(col): base for col, base in seg_base.items()}
                    origami_segs_dms_adjust[seg] = seg_dms_adjust
                    origami_segs_base_adjust[seg] = seg_base_adjust
            segments_dms[feat][origami] = pd.DataFrame.from_dict(origami_segs_dms_adjust, orient="index")
            segments_base[feat][origami] = pd.DataFrame.from_dict(origami_segs_base_adjust, orient="index")
            assert segments_dms[feat][origami].shape == segments_base[feat][origami].shape
            fname = f"segments_dms_{'-'.join(feat)}_{origami}.tsv"
            segments_dms[feat][origami].to_csv(fname, sep="\t")
            fname = f"segments_base_{'-'.join(feat)}_{origami}.tsv"
            segments_base[feat][origami].to_csv(fname, sep="\t")
            
    fig, axs = plt.subplots(len(f5_vals), len(f3_vals), squeeze=False, sharex=True, sharey=True)
    for i5, feat5 in enumerate(f5_vals):
        for i3, feat3 in enumerate(f3_vals):
            feat = feat5, feat3
            if feat in segments_dms:
                ax = axs[i5, i3]
                ax.set_xlabel(feat3)
                ax.set_ylabel(feat5)
                for origami, data in segments_dms[feat].items():
                    mean = data.mean(axis=0, skipna=True)
                    ax.plot(data.columns, mean, label=origami, linewidth=0.1)
    plt.savefig("origami_segments_dms.pdf")
    plt.close()

    max_len = 22
    for base in "MAC":
        # DMS signal
        for log in [True, False]:
            fig, axs = plt.subplots(len(f5_vals), len(f3_vals), squeeze=False, sharex=True, sharey=True)
            for i5, feat5 in enumerate(f5_vals):
                for i3, feat3 in enumerate(f3_vals):
                    feat = feat5, feat3
                    ax = axs[i5, i3]
                    if feat in segments_dms:
                        dms = pd.concat(list(segments_dms[feat].values()), axis=0)
                        if base in "AC":
                            bases = pd.concat(list(segments_base[feat].values()), axis=0)
                            dms = dms.where(bases == base)
                        if log:
                            dms = np.log10(dms)
                            value_name = "logDMS"
                        else:
                            value_name = "DMS"
                        longform = dms.melt(var_name="position", value_name=value_name)
                        sns.boxplot(data=longform, x="position", y=value_name, ax=ax, linewidth=0.25, fliersize=0.50)
                    else:
                        ax.set_xticks(list(range(max_len)))
                    ax.set_xlabel(feat3)
                    ax.set_ylabel(feat5)
            fig.set_size_inches(20, 16)
            if log:
                fname = f"origami_segments_logdms_boxplot_{base}.pdf"
            else:
                fname = f"origami_segments_dms_boxplot_{base}.pdf"
            plt.savefig(fname)
            plt.close()
        # Percentage of A and C
        fig, axs = plt.subplots(len(f5_vals), len(f3_vals), squeeze=False, sharex=True, sharey=True)
        for i5, feat5 in enumerate(f5_vals):
            for i3, feat3 in enumerate(f3_vals):
                feat = feat5, feat3
                ax = axs[i5, i3]
                if feat in segments_dms:
                    base_ids = pd.concat(list(segments_base[feat].values()), axis=0)
                    if base in "AC":
                        bases = pd.concat(list(segments_base[feat].values()), axis=0)
                        base_ids = base_ids.where(bases == base)
                    frac_A = base_ids.isin(["A"]).sum(axis=0) / base_ids.isin(["A", "C"]).sum(axis=0)
                    value_name = "FracA"
                    ax.bar(x=frac_A.index, height=frac_A)
                else:
                    ax.set_xticks(list(range(max_len)))
                ax.set_xlabel(feat3)
                ax.set_ylabel(feat5)
        fig.set_size_inches(20, 16)
        fname = f"origami_segments_fracA_boxplot_{base}.pdf"
        plt.savefig(fname)
        plt.close()
    
    
    origami_segments = pd.concat(list(origami_segments.values()), axis=0, ignore_index=True)
    origami_segments.to_csv("origami_segments.tsv", sep="\t")
    feature_names = ["scafDX", "stapDX", "stapSX", "vertex", "stapEND", "stapSXEND"]
    markers5 = {"scaf_scafDX_3": "x", "scaf_stapDX_3": "+", "scaf_stapSX_3": "1", "scaf_vertex_3": "*", "scaf_stapEND3_0": "o", "scaf_stapSXEND3_0": "D", "scaf_scafEND5_0": "s"}
    colors3 = {"scaf_scafDX_5": "#5ab4e5", "scaf_stapDX_5": "#0a72ba", "scaf_stapSX_5": "#00a875", "scaf_vertex_5": "#f7941d", "scaf_stapEND5_0": "#f15a22", "scaf_stapSXEND5_0": "#da6fab", "scaf_scafEND3_0": "#aaaaaa"}
    def get_markers(f5, f3):
        marker = markers5[f5]
        filled_marker = marker in "*oDs"
        if filled_marker:
            color = [(1.0, 1.0, 1.0, 0.0)]
            edgecolor = colors3[f3]
        else:
            color = colors3[f3]
            edgecolor = None
        return marker, color, edgecolor

    jitter_std = 0.15
    jitter_max = 0.30
    for base in "MAC":
        # For each segment class, determine the ratio of the ends to the interior
        seg_end_ratios = dict()
        fig, ax = plt.subplots()
        for feat5, feat3 in itertools.product(f5_vals, f3_vals):
            selector = np.logical_and(origami_segments["Feature5"] == feat5,
                                      origami_segments["Feature3"] == feat3)
            segs_feat = origami_segments.loc[selector]
            n_segs = segs_feat.shape[0]
            if n_segs > 0:
                # Confidence interval for ratio of ends to interior - unpaired
                log5 = np.log10(segs_feat[f"DMS5{base}"]).dropna()
                log3 = np.log10(segs_feat[f"DMS3{base}"]).dropna()
                log_int = np.log10(segs_feat[f"DMSint{base}"]).dropna()
                cm5 = sms.CompareMeans(sms.DescrStatsW(log5),
                                       sms.DescrStatsW(log_int))
                int5 = cm5.tconfint_diff(usevar="unequal")
                cm3 = sms.CompareMeans(sms.DescrStatsW(log3),
                                       sms.DescrStatsW(log_int))
                int3 = cm3.tconfint_diff(usevar="unequal")
                print(feat5, feat3)
                print("log5", log5)
                print("log3", log3)
                print("int", log_int)
                print("interval 5:", int5)
                print("interval 3:", int3)
                # Confidence interval for ratio of ends to interior - paired
                feat5_logratios = np.log10(segs_feat[f"DMS5{base}"] / segs_feat[f"DMSint{base}"])
                feat3_logratios = np.log10(segs_feat[f"DMS3{base}"] / segs_feat[f"DMSint{base}"])
                feat5_meanlog = np.mean(feat5_logratios)
                feat3_meanlog = np.mean(feat3_logratios)
                feat5_stdlog = np.std(feat5_logratios)
                feat3_stdlog = np.std(feat3_logratios)
                feat5_semlog = feat5_stdlog / np.sqrt(np.sum(np.logical_not(np.isnan(feat5_logratios))))
                feat3_semlog = feat3_stdlog / np.sqrt(np.sum(np.logical_not(np.isnan(feat3_logratios))))
                seg_end_ratios[feat5, feat3] = {"meanlog5": feat5_meanlog, "meanlog3": feat3_meanlog,
                                                "semlog5": feat5_semlog, "semlog3": feat3_semlog}
                marker, color, edgecolor = get_markers(feat5, feat3)
                ax.scatter(feat5_logratios, feat3_logratios,
                        marker=marker, c=color, edgecolor=edgecolor, alpha=0.3)
                ax.errorbar(feat5_meanlog, feat3_meanlog,
                        xerr=feat5_semlog, yerr=feat3_semlog,
                        marker=marker, c=colors3[feat3], alpha=1.0, capsize=5.0)
        ax.set_xlim((-1.0, 2.0))
        ax.set_ylim((-1.0, 2.0))
        ax.set_aspect(1.0)
        plt.savefig(f"seg_end_ratios_{base}.pdf")
        plt.close()
        seg_end_ratios = pd.DataFrame.from_dict(seg_end_ratios, orient="index")
        seg_end_ratios.to_csv(f"seg_end_ratios_{base}.tsv", sep="\t")

        # For each feature, get all the segments involving the feature
        feat_logratios = dict()
        for feat in feature_names:
            feat5s = [f for f in markers5 if f.split("_")[1].strip("53") == feat]
            assert len(feat5s) == 1
            feat5 = feat5s[0]
            feat3s = [f for f in colors3 if f.split("_")[1].strip("53") == feat]
            assert len(feat3s) == 1
            feat3 = feat3s[0]
            interval = dict()
            for side, featside in {"5": feat5, "3": feat3}.items():
                selector = origami_segments[f"Feature{side}"] == featside
                segs_feat = origami_segments.loc[selector]
                n_segs = segs_feat.shape[0]
                if n_segs > 0:
                    # Confidence interval for ratio of ends to interior - unpaired
                    log_side = np.log10(segs_feat[f"DMS{side}{base}"]).dropna()
                    log_int = np.log10(segs_feat[f"DMSint{base}"]).dropna()
                    cm = sms.CompareMeans(sms.DescrStatsW(log_side),
                                          sms.DescrStatsW(log_int))
                    interval[side] = cm.tconfint_diff(usevar="unequal")
                    # Confidence interval for ratio of ends to interior - paired
                    feat_logratios[feat, side] = np.log10(segs_feat[f"DMS{side}{base}"] / segs_feat[f"DMSint{base}"]).dropna()
        feat_logratios_df = pd.DataFrame.from_dict(dict(enumerate([{"Feature": f"{feat} at {side}'", "LogRatio": logratio}
                for (feat, side), logratios in feat_logratios.items() for logratio in logratios])), orient="index")
        feat_logratios_df.to_csv(f"feature_logratios_{base}.tsv", sep="\t")
        sns.swarmplot(data=feat_logratios_df, x="Feature", y="LogRatio")
        plt.xticks(rotation=90)
        plt.savefig(f"feature_logratios_{base}.pdf")
        plt.close()
        feature_diffs = defaultdict(dict)
        feature_pvals = defaultdict(dict)
        for f1, f2 in itertools.combinations(feat_logratios, 2):
            print(f1, f2)
            lr1 = feat_logratios[f1]
            lr2 = feat_logratios[f2]
            result = ttest_ind(lr1, lr2, equal_var=False)
            diff_mean = np.mean(lr2) - np.mean(lr1)
            pval = result.pvalue
            feature_diffs[f1][f2] = diff_mean
            feature_diffs[f2][f1] = diff_mean
            feature_pvals[f1][f2] = pval
            feature_pvals[f2][f1] = pval
        feature_diffs = pd.DataFrame.from_dict(feature_diffs)
        feature_pvals = pd.DataFrame.from_dict(feature_pvals)
        sns.heatmap(feature_diffs, square=True, center=0.0, cmap="coolwarm")
        plt.savefig(f"feature_diffs_{base}.pdf")
        plt.close()
        sns.heatmap(np.log10(feature_pvals), square=True)
        plt.savefig(f"feature_logpvals_{base}.pdf")
        plt.close()

        # Linear regression model of DMS vs. Tm 
        results = smf.ols(f"np.log(DMSint{base}) ~ Tm", data=origami_segments).fit()
        print(results.summary())
        # Plots of two variables at a time
        for p1, p2 in itertools.combinations(
            ["Length", "GC", "Tm", f"DMSint{base}"], 2):
            np.random.seed(0)
            labels = set()
            fig, ax = plt.subplots()
            xs = list()
            ys = list()
            for seg in origami_segments.index:
                x = origami_segments.loc[seg, p1]
                y = origami_segments.loc[seg, p2]
                xs.append(x)
                ys.append(y)
                x_jitter = np.random.randn() * jitter_std
                if x_jitter > 0:
                    x_jitter = min(x_jitter, jitter_max)
                else:
                    x_jitter = max(x_jitter, -jitter_max)
                if "DMS" in p1:
                    ax.set_xscale("log")
                if "DMS" in p2:
                    ax.set_yscale("log")
                label = None
                f5 = origami_segments.loc[seg, "Feature5"]
                f3 = origami_segments.loc[seg, "Feature3"]
                marker, color, edgecolor = get_markers(f5, f3)
                ax.scatter(x + x_jitter, y, s=16.0, marker=marker, c=color, edgecolors=edgecolor)
            # add means
            lengths = sorted(set(origami_segments["Length"]))
            if p1 == "Length" and "DMS" in p2:
                seg_length_dms_means = pd.Series({length: np.log10(origami_segments.loc[
                    origami_segments["Length"] == length, p2]).mean()
                    for length in lengths})
                y = np.power(10.0, seg_length_dms_means)
                ax.scatter(y.index, y, c="black", marker="_")
            ax.set_xlabel(p1)
            ax.set_ylabel(p2)
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            use = [not (np.isnan(x) or np.isnan(y)) for x, y in zip(xs, ys)]
            r = pearsonr(xs[use], ys[use])
            rho = spearmanr(xs[use], ys[use])
            print(p1, "vs", p2)
            print("\tPCC:", r)
            print("\tSRC:", rho)
            if p1 == "GC":
                ax.set_xlim((0.0, 1.0))
            elif p1 == "length":
                ax.set_xlim((6, 24))
                ax.set_xticks(list(range(6, 24 + 1, 2)))
            if p2 == "GC":
                ax.set_ylim((0.0, 1.0))
            elif p2 == "DMS_signal":
                ax.set_ylim((1e-3, 1e0))
            #plt.legend()
            plt.savefig(f"segment_data_{p1}_{p2}.pdf")
            plt.close()
    
        '''
        if base == "M":
            data = origami_segment_bases
        else:
            data = origami_segment_bases.loc[origami_segment_bases["Base"] == base]
        data = pd.concat([data[col] for col in data.columns if col != dms_col] + [np.log10(data[dms_col])], axis=1)
        ylim = (-4, 0)
        flierprops = dict(marker='o', markerfacecolor="gray", markersize=2,
                linestyle='none', markeredgecolor="gray")
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x="Feature5", y=dms_col, hue="Feature3", order=markers5, hue_order=colors3, flierprops=flierprops)
        plt.xticks(rotation=90)
        plt.savefig(f"dms_by_both_ends_{base}.pdf")
        plt.close()
        origami_markers = {"rT55": "<", "rT66v2": "^", "rT77": ">", "rO66v2": "D", "rPB66v2": "*"}
        print("point plot")
        for origami, marker in origami_markers.items():
            sns.pointplot(data=data.loc[data["Origami"] == origami], x="Feature5", y=dms_col, hue="Feature3", order=markers5, hue_order=colors3, join=False, dodge=1.0, ci=None, markers=marker, seed=0)
        plt.xticks(rotation=90)
        plt.savefig(f"dms_by_both_ends_origami_{base}.pdf")
        plt.close()
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x="Feature5", y=dms_col, hue="Origami", order=markers5, flierprops=flierprops)
        plt.xticks(rotation=90)
        plt.savefig(f"dms_by_origami_{base}.pdf")
        plt.close()
        '''

    exit()
    # Plot the signals over features for all non-skipped origamis.
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
    results = model_results["v2-M"]["distance-GC vs log DMS"]
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

    # Plot the ROC curve of DMS reactivities versus the 23S secondary structure from PDB 4V9D
    ss_pdb_ct_file = "23S_CRW-J01695+pk.ct"
    seq_start = -28
    _, pairs_23S, paired_23S, seq_23S = read_ct_file_single(ss_pdb_ct_file, start_pos=seq_start)
    plot_file = "23S_CRW-J01695+pk_roc.pdf"
    plot_data_structure_roc_curve(paired_23S, sample_mus["23S"], plot_file)
    auroc = get_data_structure_agreement("AUROC", paired_23S, sample_mus["23S"])
    print("AUROC 23S:", auroc)
    
