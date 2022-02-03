from collections import Counter

from Bio import SeqIO
from Bio.SeqUtils import MeltingTemp as mt
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.formula.api import ols
import os
import sys

import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
import patsy
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import matthews_corrcoef

sys.path.append("/home/mfallan/mfallan_git/ariadne")
import ariadne
import plots  # from ariadne


def read_pop_avg(pop_avg_file, scale):
    pop_avg_path = "BitVector_Plots/" + pop_avg_file + "_popavg_reacts.txt"
    data = pd.read_csv(pop_avg_path, sep="\t", index_col="Position")["Mismatches + Deletions"]
    return data.index.to_numpy(), data.to_numpy() * scale

def read_dot_file(dot_file):
    with open(dot_file) as f:
        energy = f.readline()
        seq = f.readline()
        struct = f.readline().strip()
    return struct

def read_ct_file(ct_file):
    idxs = list()
    g_dn = dict()
    g_up = dict()
    g_ax = dict()
    with open(ct_file) as f:
        header = f.readline()
        for line in f:
            info = line.split()
            try:
                idx, base, up, dn, ax, nat = line.split()
            except ValueError:
                break
            idx = int(idx)
            up = int(up)
            dn = int(dn)
            ax = int(ax)
            assert idx not in g_up
            assert idx != 0
            idxs.append(idx)
            g_up[idx] = up
            g_dn[idx] = dn
            g_ax[idx] = ax
            assert g_up.get(dn) in {None, idx}
            assert g_dn.get(up) in {None, idx}
            assert g_ax.get(ax) in {None, idx}
    return idxs, g_up, g_dn, g_ax

def get_pairing_status(g_up, g_dn, g_ax, idx):
    if g_ax[idx] == 0:
        return "unpaired"
    if g_up.get(g_ax.get(g_up.get(g_ax[idx]))) == idx and g_dn.get(g_ax.get(g_dn.get(g_ax[idx]))) == idx:
        return "paired"
    else:
        return "marginal"

def get_pairing_statuses(idxs, g_up, g_dn, g_ax):
    statuses = {idx: get_pairing_status(g_up, g_dn, g_ax, idx) for idx in idxs}
    return statuses

def read_cluster_mu_struct(cluster_prefix, k, scale):
    cluster_k_path = f"EM_Clustering/{cluster_prefix}_InfoThresh-0.2_SigThresh-0_IncTG-NO_DMSThresh-0.5/K_{k}"
    best = [d for d in os.listdir(cluster_k_path) if os.path.isdir(os.path.join(cluster_k_path, d)) and d.endswith("best")]
    assert len(best) == 1
    cluster_mu_file = os.path.join(cluster_k_path, best[0], "Clusters_Mu.txt")
    cluster_mu = pd.read_csv(cluster_mu_file, sep="\t", skiprows=2, index_col="Position")
    structs = dict()
    for cluster in range(1, k + 1):
        """
        cluster_dot_file = os.path.join(cluster_k_path, best[0], f"{cluster_prefix}_InfoThresh-0.2_SigThresh-0_IncTG-NO_DMSThresh-0.5-K{k}_Cluster{cluster}_expUp_0_expDown_0.dot")
        with open(cluster_dot_file) as f:
            energy = f.readline()
            seq = f.readline()
            structs[f"Cluster_{cluster}"] = f.readline().strip()
        """
        cluster_ct_file = os.path.join(cluster_k_path, best[0], f"{cluster_prefix}_InfoThresh-0.2_SigThresh-0_IncTG-NO_DMSThresh-0.5-K{k}_Cluster{cluster}_expUp_0_expDown_0.ct")
        idxs, g_up, g_dn, g_ax = read_ct_file(cluster_ct_file)
        structs[f"Cluster_{cluster}"] = {"i": idxs, "up": g_up, "dn": g_dn, "ax": g_ax}
    return cluster_mu, structs

def read_sequence(fasta_file):
    fasta_path = "Ref_Genome/" + fasta_file
    records = list(SeqIO.parse(fasta_path, "fasta"))
    assert len(records) == 1
    seq = str(records[0].seq).replace("T", "U")
    return seq

def only_bases(seq, indexes, data, bases):
    pos = [seq[i - 1] in bases for i in indexes]
    return indexes[pos], data[pos]

def only_AC(seq, indexes, data):
    return only_bases(seq, indexes, data, "AC")

def to_Series(idx, vals):
    return pd.Series(vals, index=idx)

def correlate(seq, indexes, sig1, sig2, fname, title, xlabel, ylabel, exclude=None, equal_axes=True, mean_y_line=False):
    print(title)
    if exclude:
        include = np.logical_not(exclude)
        indexes = indexes[include]
        sig1 = sig1[include]
        sig2 = sig2[include]
    idx_A, sig1_A = only_bases(seq, indexes, sig1, "A")
    idx_A, sig2_A = only_bases(seq, indexes, sig2, "A")
    idx_C, sig1_C = only_bases(seq, indexes, sig1, "C")
    idx_C, sig2_C = only_bases(seq, indexes, sig2, "C")
    sig1_AC = np.hstack([sig1_A, sig1_C])
    sig2_AC = np.hstack([sig2_A, sig2_C])
    n, = sig1_AC.shape
    r2_A = np.corrcoef(sig1_A, sig2_A)[0, 1]**2
    r2_C = np.corrcoef(sig1_C, sig2_C)[0, 1]**2
    r2_AC = np.corrcoef(sig1_AC, sig2_AC)[0, 1]**2
    print(f"n: {n}, R^2 = {round(r2_AC, 5)}\n\tA: {round(r2_A, 5)}\n\tC: {round(r2_C, 5)}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if equal_axes:
        fig.set_size_inches(5, 5)
        ax.set_aspect("equal")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    max_value = max((np.max(sig1_AC), np.max(sig2_AC))) * 1.05
    legend_items = list()
    if equal_axes:
        plt.plot([0, max_value], [0, max_value], color="green", alpha=0.3)
        legend_items.append("y = x")
    if mean_y_line:
        y_means_A = [np.mean([y for x2, y in zip(sig1_A, sig2_A) if np.isclose(x1, x2)]) for x1 in sorted(sig1_A)]
        plt.plot(sorted(sig1_A), y_means_A, color="pink")
        legend_items.append("mean A")
        y_means_C = [np.mean([y for x2, y in zip(sig1_C, sig2_C) if np.isclose(x1, x2)]) for x1 in sorted(sig1_C)]
        plt.plot(sorted(sig1_C), y_means_C, color="lightblue")
        legend_items.append("mean C")
    plt.scatter(sig1_A, sig2_A, color="red", alpha=0.3, marker=".")
    legend_items.append("A")
    plt.scatter(sig1_C, sig2_C, color="blue", alpha=0.3, marker=".")
    legend_items.append("C")
    plt.scatter(sig1_C, sig2_C, color="blue", alpha=0.3, marker=".")
    plt.legend(legend_items)
    if equal_axes:
        plt.xlim((0, max_value))
        plt.ylim((0, max_value))
    plt.savefig(fname, dpi=300)
    plt.close()
    return r2_AC

def difference(seq, indexes, sig1, sig2, exclude=None, include=None, bases="AC"):
    sig1_idx = to_Series(*only_bases(seq, indexes, sig1, bases))
    sig2_idx = to_Series(*only_bases(seq, indexes, sig2, bases))
    if exclude:
        sig1_idx = sig1_idx.loc[[idx for idx in sig1_idx.index if idx not in exclude]]
        sig2_idx = sig2_idx.loc[[idx for idx in sig2_idx.index if idx not in exclude]]
    if include:
        sig1_idx = sig1_idx.loc[[idx for idx in sig1_idx.index if idx in include]]
        sig2_idx = sig2_idx.loc[[idx for idx in sig2_idx.index if idx in include]]
    mean1 = np.mean(sig1_idx)
    mean2 = np.mean(sig2_idx)
    med1 = np.median(sig1_idx)
    med2 = np.median(sig2_idx)
    u, p = mannwhitneyu(sig1_idx, sig2_idx)
    print("Mean 1:", mean1)
    print("Mean 2:", mean2)
    print("Median 1:", med1)
    print("Median 2:", med2)
    print("P-value:", p)
    return mean1, mean2, med1, med2, u, p

def distribution(seq, indexes, sig, fname, title, xlabel, ylabel, exclude=None):
    print(title)
    if exclude:
        include = np.logical_not(exclude)
        indexes = indexes[include]
        sig = sig[include]
    idx_A, sig_A = only_bases(seq, indexes, sig, "A")
    idx_C, sig_C = only_bases(seq, indexes, sig, "C")
    sig_AC = np.hstack([sig_A, sig_C])
    n_AC, = sig_AC.shape
    n_A, = sig_A.shape
    n_C, = sig_C.shape
    median_AC = np.median(sig_AC)
    mean_AC = np.mean(sig_AC)
    median_A = np.median(sig_A)
    mean_A = np.mean(sig_A)
    median_C = np.median(sig_C)
    mean_C = np.mean(sig_C)
    print(f"n = {n_AC}, median = {round(median_AC, 5)}, mean = {round(mean_AC, 5)}")
    print(f"\tn A = {n_A}, median A = {round(median_A, 5)}, mean A = {round(mean_A, 5)}")
    print(f"\tn C = {n_C}, median C = {round(median_C, 5)}, mean C = {round(mean_C, 5)}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist([sig_A, sig_C], bins=np.arange(0.0, 0.25, 0.005), stacked=True, color=["red", "blue"])
    plt.legend(["A", "C"])
    plt.savefig(fname, dpi=300)
    plt.close()
    # Cumulative distribution
    pdf_x = bins=np.arange(0.0, 0.25, 0.001)
    pdf_y, _ = np.histogram(sig_AC, bins=pdf_x)
    cdf_y = np.cumsum(pdf_y)
    plt.plot(pdf_x[:-1], cdf_y)
    plt.title(f"Cumulative distribution: {title}")
    plt.xlabel("Mutation rate")
    plt.ylabel("Frequency")
    plt.savefig(f"cum_{fname}")
    plt.close()
    for thresh in [0.0025, 0.005, 0.010]:
        print(f"    fraction <= {thresh}: {sum(sig_AC <= thresh) / len(sig_AC)}")
    return median_AC, mean_AC

def dist_paired_vs_unpaired(seq, indexes, sigs, structs):
    assert len(indexes) == len(sig) == len(struct["i"])
    seq_idx_to_structs_idx = [{x: y for x, y in zip(indexes, struct["i"])} for struct in structs]
    for i in range(len(seq_idx_to_structs_idx) - 1):
        assert seq_idx_to_structs_idx[i] == seq_idx_to_structs_idx[i + 1]
    seq_idx_to_struct_idx = seq_idx_to_structs_idx[0]
    sigs_AC = [to_Series(*only_AC(seq, indexes, sig)) for sig in sigs]
    indexes_AC = [list(sig.index) for sig in sigs_AC]
    status_opts = ["paired", "marginal", "unpaired"]
    pairings = [get_pairing_statuses(struct["i"], struct["up"], struct["dn"], struct["ax"]) for struct in structs]
    statuses_AC = dict()
    for i, pairing in enumerate(pairings, start=1):
        with open(f"pairing_status_{i}.txt", 'w') as f:
            f.write("\n".join([str(status_opts.index(pairing[seq_idx_to_struct_idx[idx]]) / 2) for idx in indexes]))
    for idx in indexes_AC:
        status = pairing[seq_idx_to_struct_idx[idx]]
        if status not in statuses_AC:
            statuses_AC[status] = list()
        statuses_AC[status].append(idx)
    sig_status = {status: [sig_AC.loc[idx] for idx in indexes] for status, indexes in statuses_AC.items()}
    sig_status_bool = {status: [idx in status_indexes for idx in sig_AC.index] for status, status_indexes in statuses_AC.items()}
    # Find the threshold that distinguishes paired and unpaired bases with greatest MCC
    opt_thresh = None
    mcc_max = None
    mccs = list()
    threshs = list()
    ranked_sig = sorted(sig_AC)
    for i in range(len(ranked_sig) - 1):
        thresh_lo = ranked_sig[i]
        thresh_hi = ranked_sig[i + 1]
        thresh = (thresh_lo + thresh_hi) / 2
        mcc = matthews_corrcoef(sig_status_bool["unpaired"], sig_AC > thresh)
        if mcc_max is None or mcc > mcc_max:
            mcc_max = mcc
            opt_thresh = thresh
        mccs.append(mcc)
        threshs.append(thresh)
    plt.plot(threshs, mccs)
    plt.savefig("mcc.png")
    plt.close()
    print("\n".join([f"{t}\t{m}" for t, m in zip(threshs, mccs) if 0.03 < t < 0.07]))
    print("Opt thresh:", opt_thresh)
    print("MCC:", mcc_max)
    # Find the threshold that distinguishes paired and unpaired bases with greatest MCC
    a_p, loc_p, scale_p = stats.gamma.fit(sig_paired, floc=0)
    a_up, loc_up, scale_up = stats.gamma.fit(sig_unpaired, floc=0)
    dist_paired = stats.gamma(a=a_p, loc=loc_p, scale=scale_p)
    dist_unpaired = stats.gamma(a=a_up, loc=loc_up, scale=scale_up)
    x = np.linspace(0, max(sig_AC), 200)
    hist_bins = np.linspace(0, max(sig_AC), 20)
    for status, sigs in sig_status.items():
        s, loc, scale = stats.lognorm.fit(sigs, floc=0)
        dist = stats.lognorm(s=s, loc=loc, scale=scale)
        plt.plot(x, dist.pdf(x), label=f"{status} fit")
        plt.hist(sigs, label=status, alpha=0.5, bins=hist_bins)
    plt.xlabel("mutation rate")
    plt.ylabel("frequency")
    plt.legend()
    plt.savefig("paired_vs_unpaired.png")
    plt.close()

def dist_compare(seq, indexes, sig_ref, name_ref, sigs, name_sigs, fname, title, xlabel, ylabel, exclude=None):
    print(title)
    if exclude:
        include = np.logical_not(exclude)
        indexes = indexes[include]
        sig = sig[include]
    idx_ref_AC, sig_ref_AC = only_bases(seq, indexes, sig_ref, "AC")
    idxs_AC, sigs_AC = dict(), dict()
    for name, sig in zip(name_sigs, sigs):
        idx_AC, sig_AC = only_bases(seq, indexes, sig, "AC")
        idxs_AC[name] = idx_AC
        sigs_AC[name] = sig_AC
    idxs_AC = pd.DataFrame.from_dict(idxs_AC)
    sigs_AC = pd.DataFrame.from_dict(sigs_AC)
    sigs_diff = pd.DataFrame.from_dict({name: sig - sig_ref_AC for name, sig in sigs_AC.items()})
    sigs_diff = sigs_diff.melt(var_name="sample", value_name="mutation rate")
    sigs_AC["scaffold"] = sig_ref_AC
    sigs_rate = sigs_AC.melt(var_name="sample", value_name="mutation rate")
    sigs_all = pd.concat([sigs_rate, sigs_diff], axis=0)
    label = ["rate"] * len(sigs_rate) + ["difference"] * len(sigs_diff)
    sigs_all.loc[:, "label"] = label
    #fig, ax = plt.subplots()
    #ax.set_aspect(10)
    #sns.stripplot(data=data, x="sample", y="mutation rate", hue="base", hue_order=["A", "C"], palette=["red", "blue"], size=2)
    sns.set_style("darkgrid")
    #sns.stripplot(data=sigs_all, x="sample", y="mutation rate", hue="label", size=1.5, dodge=True)
    #sns.violinplot(data=sigs_all, x="sample", y="mutation rate", hue="label")
    sns.boxplot(data=sigs_rate, x="sample", y="mutation rate")
    sns.stripplot(data=sigs_rate, x="sample", y="mutation rate")
    #sns.stripplot(data=sigs_all, x="sample", y="mutation rate", palette=[(119/255, 101/255, 69/255)], size=1.5)
    #sns.boxplot(data=sigs_all, x="sample", y="mutation rate", palette=[(232/255, 197/255, 134/255, 1)], dodge=True)
    plt.title(title)
    plt.savefig(fname, dpi=300)
    plt.close()

def combine_regions_idx_sig(fnames, offsets, scale):
    if offsets is None:
        offsets = [0 for f in fnames]
    indexes = list()
    signals = list()
    assert len(fnames) == len(offsets)
    for fname, offset in zip(fnames, offsets):
        idx, sig = read_pop_avg(fname, scale)
        indexes.append(idx + offset)
        signals.append(sig)
    idx_cat = np.hstack(indexes)
    sig_cat = np.hstack(signals)
    return idx_cat, sig_cat

def compute_scale(seq_file, ref_popavg_file, query_popavg_file, fraction=0.5):
    """ Compute the ratio of a query signal to a reference signal. """
    seq = read_sequence(seq_file)
    # get population average signals
    idx_ref, sig_ref = read_pop_avg(ref_popavg_file, 1)
    idx_query, sig_query = read_pop_avg(query_popavg_file, 1)
    assert np.all(idx_query == idx_ref)
    # find the As and Cs
    idx_ref_ac, sig_ref_ac = only_bases(seq, idx_ref, sig_ref, "AC")
    idx_query_ac, sig_query_ac = only_bases(seq, idx_ref, sig_query, "AC")
    # find the most reactive bases in the query and reference
    n_top = int(round(len(idx_ref_ac) * fraction))
    ref_median = np.median(sorted(sig_ref_ac)[-n_top:])
    query_median = np.median(sorted(sig_query_ac)[-n_top:])
    scale = ref_median / query_median
    return scale


seq_23S = read_sequence("23S_scaffold.fasta")
seq_reg1 = read_sequence("23S_scaffold_reg1.fasta")
seq_reg3 = read_sequence("23S_scaffold_reg3.fasta")
seq_RRE = read_sequence("HIV_RRE_232nt.fasta")
offset_reg1 = 4
offset_reg3 = 803
offsets = [offset_reg1, offset_reg3]


# Distributions of lengths by position within origami (i.e. in middle or end of edge, at crossover or terminus position)
design_dir = "DAEDALUS_Designs"
design_to_name = {("rPB66", "v1"): "LibFig_rPB66", ("rPB66", "v2"): "23s_rPB66_v2", ("rOct66", "v1"): "LibFig_rOct66", ("rOct66", "v2"): "23s_rO66_v2"}
name_to_design = {v: k for k, v in design_to_name.items()}

"""
rPB66_v2_pdb = os.path.join(design_dir, "23s_rPB66_v2", "18_pentagonal_bipyramid_(J13)_66_scaf_23s_rPB66_v2_singleXOVs_2020-05-14.pdb")
rPB66_v2_cndo = os.path.join(design_dir, "23s_rPB66_v2", "18_pentagonal_bipyramid_(J13)_66_scaf_23s_rPB66_v2_singleXOVs_2020-05-14.cndo")
with open("rPB66_v2_color_script.txt", 'w') as f:
    f.write(dtool.color_locations(rPB66_v2_cndo, rPB66_v2_pdb))
"""


# Compute scale ratios
scales = {
        "23S_C": compute_scale("HIV_RRE_232nt.fasta", "23S_C_HIV_RRE_232nt_28_202", "23S_C_HIV_RRE_232nt_28_202"),
        "23S_H": compute_scale("HIV_RRE_232nt.fasta", "23S_C_HIV_RRE_232nt_28_202", "23S_H_HIV_RRE_232nt_28_202"),
        "rPB66_v1": compute_scale("HIV_RRE_232nt.fasta", "23S_C_HIV_RRE_232nt_28_202", "rPB66_v1_HIV_RRE_232nt_28_202"),
        "rPB66_v2_C": compute_scale("HIV_RRE_232nt.fasta", "23S_C_HIV_RRE_232nt_28_202", "rPB66_v2_C_HIV_RRE_232nt_28_202"),
        "rPB66_v2_H": compute_scale("HIV_RRE_232nt.fasta", "23S_C_HIV_RRE_232nt_28_202", "rPB66_v2_H_HIV_RRE_232nt_28_202"),
        "rPB66_v2_4": compute_scale("HIV_RRE_232nt.fasta", "23S_C_HIV_RRE_232nt_28_202", "rPB66_v2_4_HIV_RRE_232nt_28_202"),
        "rOct66_v1": compute_scale("HIV_RRE_232nt.fasta", "23S_C_HIV_RRE_232nt_28_202", "rOct66_v1_HIV_RRE_232nt_28_202"),
        "rOct66_v2": compute_scale("HIV_RRE_232nt.fasta", "23S_C_HIV_RRE_232nt_28_202", "rOct66_v2_HIV_RRE_232nt_28_202"),
        }

# Correlations between rPB66 v2 signal in cacodylate and HEPES
rPB66_v2_C_fnames = ["rPB66_v2_C_23S_scaffold_reg1_21_468", "rPB66_v2_C_23S_scaffold_reg3_22_484"]
idx_rPB66_v2_C, sig_rPB66_v2_C = combine_regions_idx_sig(rPB66_v2_C_fnames, offsets, scales["rPB66_v2_C"])
rPB66_v2_H_fnames = ["rPB66_v2_H_23S_scaffold_reg1_21_468", "rPB66_v2_H_23S_scaffold_reg3_22_484"]
idx_rPB66_v2_H, sig_rPB66_v2_H = combine_regions_idx_sig(rPB66_v2_H_fnames, offsets, scales["rPB66_v2_H"])
assert np.all(idx_rPB66_v2_C == idx_rPB66_v2_H)
correlate(seq_23S, idx_rPB66_v2_C, sig_rPB66_v2_C, sig_rPB66_v2_H, "rPB66_v2_C_vs_H.png", "Correlation of rPB66 signal in cacodylate and HEPES", "cacodylate", "HEPES")

# Correlations between RRE signal in cacodylate and HEPES
rPB66_v2_C_RRE_fnames = ["rPB66_v2_C_HIV_RRE_232nt_28_202"]
idx_rPB66_v2_C_RRE, sig_rPB66_v2_C_RRE = combine_regions_idx_sig(rPB66_v2_C_RRE_fnames, None, scales["rPB66_v2_C"])
rPB66_v2_H_RRE_fnames = ["rPB66_v2_H_HIV_RRE_232nt_28_202"]
idx_rPB66_v2_H_RRE, sig_rPB66_v2_H_RRE = combine_regions_idx_sig(rPB66_v2_H_RRE_fnames, None, scales["rPB66_v2_H"])
assert np.all(idx_rPB66_v2_C_RRE == idx_rPB66_v2_H_RRE)
correlate(seq_RRE, idx_rPB66_v2_C_RRE, sig_rPB66_v2_C_RRE, sig_rPB66_v2_H_RRE, "RRE_C_vs_H.png", "Correlation of RRE signal in cacodylate and HEPES", "cacodylate", "HEPES")

# Correlations between rPB66 with and without staple 4
rPB66_v2_4_fnames = ["rPB66_v2_4_23S_scaffold_reg1_21_468", "rPB66_v2_4_23S_scaffold_reg3_22_484"]
idx_rPB66_v2_4, sig_rPB66_v2_4 = combine_regions_idx_sig(rPB66_v2_4_fnames, offsets, scales["rPB66_v2_4"])
rPB66_v2_4_RRE_fnames = ["rPB66_v2_4_HIV_RRE_232nt_28_202"]
idx_rPB66_v2_4_RRE, sig_rPB66_v2_4_RRE = combine_regions_idx_sig(rPB66_v2_4_RRE_fnames, None, scales["rPB66_v2_4"])
assert np.all(idx_rPB66_v2_4 == idx_rPB66_v2_C)
correlate(seq_23S, idx_rPB66_v2_C, sig_rPB66_v2_C, sig_rPB66_v2_4, "rPB66_v2_C_vs_4.png", "Correlation of rPB66 signal with and without staple 4", "with", "without")
correlate(seq_RRE, idx_rPB66_v2_C_RRE, sig_rPB66_v2_C_RRE, sig_rPB66_v2_4_RRE, "RRE_C_vs_4.png", "Correlation of RRE signal with and without staple 4", "with", "without")

# correlation excluding the bases covered by staple 4
stap4_first, stap4_last = 1215, 1236
stap4_indexes = list(range(stap4_first, stap4_last + 1))
pos_excl_stap4 = [idx in stap4_indexes for idx in idx_rPB66_v2_4]
correlate(seq_23S, idx_rPB66_v2_C, sig_rPB66_v2_C, sig_rPB66_v2_4, "rPB66_v2_C_vs_4_excl.png", "Correlation of rPB66 signal with and without staple 4", "with", "without", exclude=pos_excl_stap4)
print("Staple 4")
difference(seq_23S, idx_rPB66_v2_C, sig_rPB66_v2_C, sig_rPB66_v2_4, include=stap4_indexes)
print("Bases besides staple 4")
difference(seq_23S, idx_rPB66_v2_C, sig_rPB66_v2_C, sig_rPB66_v2_4, exclude=stap4_indexes)

# Distribution of mutation rates in paired and unpaired in RRE
sig_RRE_23S, structs = read_cluster_mu_struct("23S_C_HIV_RRE_232nt_28_202", k=2, scale=1)
for cluster in ["Cluster_1", "Cluster_2"]:
    dist_paired_vs_unpaired(seq_RRE, sig_RRE_23S.index, sig_RRE_23S[cluster], structs[cluster])
    input()


# Distributions of rPB66 and rOct66 v1 and v2
idx_rPB66_v1, sig_rPB66_v1 = combine_regions_idx_sig(["rPB66_v1_23S_scaffold_reg1_21_468", "rPB66_v1_23S_scaffold_reg3_22_484"], offsets, scales["rPB66_v1"])
idx_rOct66_v1, sig_rOct66_v1 = combine_regions_idx_sig(["rOct66_v1_23S_scaffold_reg1_21_468", "rOct66_v1_23S_scaffold_reg3_22_484"], offsets, scales["rOct66_v1"])
idx_rOct66_v2, sig_rOct66_v2 = combine_regions_idx_sig(["rOct66_v2_23S_scaffold_reg1_21_468", "rOct66_v2_23S_scaffold_reg3_22_484"], offsets, scales["rOct66_v2"])
idx_23S, sig_23S = combine_regions_idx_sig(["23S_C_23S_scaffold_reg1_21_468", "23S_C_23S_scaffold_reg3_22_484"], offsets, scales["23S_C"])
distribution(seq_23S, idx_rPB66_v1, sig_rPB66_v1, "rPB66_v1_hist.png", "Distribution of signal in rPB66 v1", "signal", "frequency")
distribution(seq_23S, idx_rPB66_v2_C, sig_rPB66_v2_C, "rPB66_v2_C_hist.png", "Distribution of signal in rPB66 v2", "signal", "frequency")
distribution(seq_23S, idx_rOct66_v1, sig_rOct66_v1, "rOct66_v1_hist.png", "Distribution of signal in rOct66 v1", "signal", "frequency")
distribution(seq_23S, idx_rOct66_v2, sig_rOct66_v2, "rOct66_v2_hist.png", "Distribution of signal in rOct66 v2", "signal", "frequency")
distribution(seq_23S, idx_23S, sig_23S, "23S_hist.png", "Distribution of signal in 23S scaffold", "signal", "frequency")

# Significance of differences between origamis
print("rPB66 v1 vs v2")
difference(seq_23S, idx_23S, sig_rPB66_v1, sig_rPB66_v2_C)
print("rPB66 v1 vs scaffold")
difference(seq_23S, idx_23S, sig_rPB66_v1, sig_23S)
print("rPB66 v1 vs scaffold")
difference(seq_23S, idx_23S, sig_rPB66_v2_C, sig_23S)
print("v1 rO66 vs rPB66")
difference(seq_23S, idx_23S, sig_rOct66_v1, sig_rPB66_v1)
print("v2 rO66 vs rPB66")
difference(seq_23S, idx_23S, sig_rOct66_v2, sig_rPB66_v2_C)
print("rO66 v1 vs scaffold")
difference(seq_23S, idx_23S, sig_rOct66_v1, sig_23S)
print("rO66 v2 vs scaffold")
difference(seq_23S, idx_23S, sig_rOct66_v2, sig_23S)
print("rO66 v1 vs v2")
difference(seq_23S, idx_23S, sig_rOct66_v1, sig_rOct66_v2)


# Comparisons of distributions on scaffold and origami
dist_compare(seq_23S, idx_23S, sig_23S, "scaffold", [sig_rPB66_v1, sig_rPB66_v2_C, sig_rOct66_v1, sig_rOct66_v2], ["rPB66 α", "rPB66 β", "rOct66 α", "rOct66 β"], "rPB66_v2_C_vs_scaf.png", "DMS signal in rPB66 v2 and scaffold", "condition", "signal")


signals = {
    ("rPB66", "v1"): pd.Series(sig_rPB66_v1, idx_23S),
    ("rPB66", "v2"): pd.Series(sig_rPB66_v2_C, idx_23S),
    ("rOct66", "v1"): pd.Series(sig_rOct66_v1, idx_23S),
    ("rOct66", "v2"): pd.Series(sig_rOct66_v2, idx_23S),
    ("23S", ""): pd.Series(sig_23S, idx_23S),
}

# Correlations between versions 1 and 2 of each origami
correlate(seq_23S, idx_23S, sig_rPB66_v1, sig_rPB66_v2_C, "rPB66_v1_vs_v2.png", "Correlation of signal on rPB66 v1 and v2", "rPB66 v1", "rPB66 v2")
correlate(seq_23S, idx_23S, sig_rOct66_v1, sig_rOct66_v2, "rOct66_v1_vs_v2.png", "Correlation of signal on rOct66 v1 and v2", "rOct66 v1", "rOct66 v2")


# Distributions of rPB66 and rOct66 signal by location of base
loc_dist_labels = list()
loc_dist_values = list()

def signal_by_loc(origami, version, topology, indexes, signal, labels, values):
    for loc, bases in topology.items():
        loc_bases = [idx in bases for idx in indexes]
        if any(loc_bases):
            loc_sig = signal[loc_bases]
            labels.append(f"{origami} {version} {loc}")
            values.append(loc_sig)
            #distribution(seq_23S, loc_bases, loc_sig, f"{origami}_{version}_{loc}_hist.png", f"Distribution of signal in {origami} {version} {loc}", "signal", "frequency")

seq_23S_series = pd.Series(dict(enumerate(seq_23S, start=1)))
features_all = dict()
edges_all = dict()
g_all = dict()
for origami, name in design_to_name.items():
    edges, g_up, g_dn, g_ax, base_info, dssr_info = ariadne.analyze_design(os.path.join(design_dir, name), clobber=True)
    features_all[origami] = base_info.iloc[list(range(len(seq_23S_series))), :]
    edges_all[origami] = edges
    g_all[origami] = {"up": g_up, "dn": g_dn, "ax": g_ax}

coeffs = dict()
conf_alpha = 0.95
conf_ints = dict()
p_vals = dict()
r2s = dict()
print("FEATURE RELATIONSHIPS")
for origami, features_origami in features_all.items():
    print(origami)
    name = design_to_name[origami]
    features_origami.index = seq_23S_series.index
    features_origami["base"] = seq_23S_series
    #features_origami["seq"] = seq_23S_series
    features_origami["signal"] = signals[origami]
    features_origami["siglog"] = np.log10(signals[origami])
    features_origami.to_csv(f"{name}_features-all.csv")
    # retain only the bases with signal for further analysis
    sig_bases = ["A", "C"]
    features_origami_sig = features_origami.loc[
            np.logical_and(np.logical_not(np.isnan(features_origami["signal"])), np.isin(features_origami["base"], sig_bases)),
            :]
    sns.boxplot(data=features_origami_sig, x=f"siglog", y="location", hue="base", dodge=True)
    plt.title(f"Distribution of signal by location in {name}")
    plt.xlabel("log10(signal)")
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    fig.savefig(f"{name}_sig_vs_loc_hist.png", dpi=300)
    plt.close()
    # secondary structure plot
    secondary_structure_fname = f"secondary_structure_{name}.pdf"
    plots.secondary_structure_signal(secondary_structure_fname, edges_all[origami], g_all[origami]["up"], g_all[origami]["dn"], g_all[origami]["ax"], features_origami, features_origami_sig["signal"])
    plt.close()
    # linear regression
    default_loc = "Scaffold_Middle_0"
    locations = sorted(set(features_origami_sig["location"]))
    default_idx = locations.index(default_loc)
    locations.pop(default_idx)
    locations = [default_loc] + locations
    formula_lhs = "siglog"
    formula_rhs = "C(base, levels=sig_bases) + C(location, levels=locations) + C(Segment5Feature) + C(Segment3Feature) + SegmentLength + SegmentGC"
    #formula_rhs = "C(base, levels=sig_bases) + C(Segment5Feature) + C(Segment3Feature) + SegmentLength + SegmentGC"
    #formula_rhs = "C(base, levels=sig_bases)"
    lm = ols(f"{formula_lhs} ~ {formula_rhs}", data=features_origami_sig).fit()
    print(lm.summary())
    coeffs[name] = lm.params
    conf_ints[name] = lm.conf_int(conf_alpha)
    p_vals[name] = lm.pvalues
    r2s[name] = lm.rsquared
    fitted = lm.fittedvalues
    residuals = lm.resid
    plt.scatter(fitted, residuals)
    plt.title("Residuals vs fitted")
    plt.xlabel("fitted")
    plt.ylabel("residuals")
    plt.savefig(f"{name}_resid_vs_fitted.png", dpi=300)
    plt.close()
    """
    # kmer signal vs melting temp
    ks = [5, 10, 15, 20]
    for k in ks:
        k_tms_ref = dict()
        k_tms = list()
        k_sig_means = list()
        k_mean_sig = dict()
        for k_start in range(1, len(seq_23S) - k + 2):
            k_end = k_start + k - 1
            k_seq = seq_23S[k_start - 1: k_end]
            k_sig = [features_origami_sig["siglog"][i] if i in features_origami_sig["siglog"] else np.nan for i in range(k_start, k_end + 1)]
            k_sig_mean = np.nanmean(k_sig)
            k_sig_means.append(k_sig_mean)
            if k_seq not in k_tms_ref:
                k_tm = mt.Tm_NN(k_seq, nn_table=mt.R_DNA_NN1, saltcorr=6)
                k_tms_ref[k_seq] = k_tm
            k_tms.append(k_tms_ref[k_seq])
            if k_seq not in k_mean_sig:
                k_mean_sig[k_seq] = list()
            k_mean_sig[k_seq].append(k_sig_mean)
        plt.scatter(k_tms, k_sig_means)
        r = np.corrcoef(k_tms, k_sig_means)[1, 0]
        plt.title(f"Tm (predicted) vs mutation rate for {k}-mers in {name}")
        plt.xlabel("Tm (predicted) (C)")
        plt.ylabel("log10(mutation rate)")
        plt.savefig(f"{name}_DMS_vs_Tm_{k}-mer_{r}.png", dpi=300)
        plt.close()
        k_mean_sig = [(k_seq, np.nanmean(sigs)) for k_seq, sigs in k_mean_sig.items()]
        order = np.argsort([sig for seq, sig in k_mean_sig])
        k_mean_sig_ordered = [k_mean_sig[i] for i in order]
        with open(f"{name}_{k}-mer_sig.txt", 'w') as f:
            f.write("\n".join([f"{seq}\t{sig}" for seq, sig in k_mean_sig_ordered]))
    """

coeffs = pd.DataFrame.from_dict(coeffs)
#conf_range = pd.DataFrame.from_dict(conf_ints)
#conf_radius = conf_range[1] - conf_range[0]
coeffs = pd.melt(coeffs, var_name="origami", value_name="value", ignore_index=False)
coeffs.reset_index(inplace=True)
coeffs.rename(columns={"index": "location"}, inplace=True)
r2s = pd.Series(r2s)
print(coeffs)
#print(conf_range)
#print(pvalues)
print(r2s)
print(coeffs.columns)
fig = plt.figure()
ax = fig.add_subplot(111)
sns.barplot(data=coeffs, x="location", y="value", hue="origami")
plt.title("Multilinear regression coefficients")
plt.xlabel("coefficient")
plt.xticks(rotation=90)
plt.ylabel("value")
ax.set_aspect(len(coeffs) / 30)
ax.get_legend().remove()
plt.tight_layout()
plt.savefig("coeffs.png", dpi=300)

