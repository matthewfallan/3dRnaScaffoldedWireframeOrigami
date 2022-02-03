import itertools
import os
import sys

from Bio import SeqIO
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta
import seaborn as sns
from statsmodels.formula.api import ols

sys.path.append("/home/mfallan/mfallan_git/ariadne")
import ariadne
import plots  # from ariadne


plt.rcParams["font.family"] = "Arial"

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
    "rsc1218v1-660": [(29, 471), (436, 629)],
    "rsc1218v1-924": [(29, 471), (436, 892)],
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


def read_fasta(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    n_rec = len(records)
    if n_rec != 1:
        raise ValueError(f"Found {n_rec} records in {fasta_file} (expected 1)")
    seq = str(records[0].seq).replace("U", "T")
    return seq


def get_subset_positions(sequence, subset, start=1):
    subset_positions = [pos for pos, base in enumerate(sequence, start=start) if base in subset]
    return subset_positions


def get_ac_positions(sequence, start=1):
    ac_positions = get_subset_positions(sequence, "AC", start=start)
    return ac_positions


def read_cluster_mus(cluster_mus_file, positions=None, flatten=False, skiprows=2, seq=None, only_ac=True):
    cluster_mus = pd.read_csv(cluster_mus_file, sep="\t", skiprows=skiprows, index_col="Position")
    if only_ac:
        assert seq is not None
        ac_pos = get_ac_positions(seq)
        if positions is None:
            positions = ac_pos
        else:
            positions = [pos for pos in positions if pos in ac_pos]
    if positions is not None:
        retain = [pos for pos in positions if pos in cluster_mus.index]
        cluster_mus = cluster_mus.loc[retain, :]
    cluster_mus.columns = [int(col.split("_")[1]) for col in cluster_mus.columns]
    if flatten and len(cluster_mus.columns) == 1:
        return cluster_mus.squeeze()
    else:
        return cluster_mus


def read_pop_avg(pop_avg_file, positions=None, flatten=False):
    pop_avg = pd.read_csv(pop_avg_file, sep="\t", index_col="Position")
    if positions is not None:
        retain = [pos for pos in positions if pos in pop_avg.index]
        pop_avg = pop_avg.loc[retain, :]
    if flatten:
        return pop_avg["Mismatches + Deletions"]
    else:
        return pop_avg


def get_identifier(em_dir, sample, ref, start, end, k, **kwargs):
    info = kwargs.get("info", "0.05")
    sig = kwargs.get("sig", "0.0")
    tg = kwargs.get("tg", "NO")
    dms = kwargs.get("dms", "0.5")
    prefix = f"{sample}_{ref}_{start}_{end}"
    matches = [d for d in os.listdir(em_dir) if d.startswith(prefix)]
    if len(matches) == 0:
        raise ValueError(f"Identifier {prefix} not found in {em_dir}")
    if len(matches) > 1:
        prefix = f"{sample}_{ref}_{start}_{end}_InfoThresh-{info}_SigThresh-{sig}_IncTG-{tg}_DMSThresh-{dms}"
        matches = [d for d in os.listdir(em_dir) if d.startswith(prefix)]
        if len(matches) > 1:
            raise ValueError("Ambiguous identifier: please specify info, sig, tg, and dms.")
        elif len(matches) == 0:
            raise ValueError(f"Identifier {prefix} not found in {em_dir}")
    identifier = matches[0]
    return identifier


def get_run_dir(project, em_dir, sample, ref, start, end, k, join_to_projects_dir=True, run=None, **kwargs):
    if join_to_projects_dir:
        project_em_dir = os.path.join(projects_dir, project, em_dir)
    else:
        project_em_dir = os.path.join(project, em_dir)
    identifier = get_identifier(project_em_dir, sample, ref, start, end, k, **kwargs)
    k_dir = os.path.join(project_em_dir, identifier, f"K_{k}")
    if not os.path.isdir(k_dir):
        raise ValueError(f"Clustering directory does not exist: {k_dir}")
    if run is None:
        best_run_dirs = [d for d in os.listdir(k_dir) if d.endswith("best")]
        if len(best_run_dirs) == 0:
            raise ValueError(f"Could not find best run in directory: {k_dir}")
        if len(best_run_dirs) > 1:
            raise ValueError(f"Found multiple best runs in directory: {k_dir}")
        run_dir = os.path.join(k_dir, best_run_dirs[0])
    else:
        run_dir = os.path.join(k_dir, f"run_{run}")
        if not os.path.isdir(run_dir):
            raise ValueError(f"Directory for run does not exist: {run_dir}")
    return run_dir


def get_mu_file_full_name(project, em_dir, sample, ref, start, end, k, run=None, **kwargs):
    run_dir = get_run_dir(project, em_dir, sample, ref, start, end, k, run=run, **kwargs)
    mu_file = os.path.join(run_dir, "Clusters_Mu.txt")
    return mu_file


def get_dot_file_full_name(project, em_dir, sample, ref, start, end, k, c, run=None, **kwargs):
    run_dir = get_run_dir(project, em_dir, sample, ref, start, end, k, run=run, **kwargs)
    prefix = run_dir.split(os.sep)[-3]
    dot_file = os.path.join(run_dir, f"{prefix}-K{k}_Cluster{c}_expUp_0_expDown_0_pk.dot")
    return dot_file


def read_dot_file(dot_file):
    with open(dot_file) as f:
        name, seq, structure = [f.readline().strip() for i in range(3)]
    return name, seq, structure


def hamming_dist(str1, str2):
    assert len(str1) == len(str2)
    dist = sum([x != y for x, y in zip(str1, str2)])
    return dist


def get_sample_mus(sample_data, sample_name):
    # Get the IDs of the samples containing the origami and the RRE.
    origami_num = sample_data.loc[sample_name, "Origami"]
    origami_id = num_to_id(origami_num)
    rre_num = sample_data.loc[sample_name, "RRE"]
    rre_id = num_to_id(rre_num)
    # Read the sequence of the origami scaffold.
    scaf_name = sample_data.loc[sample_name, "Scaffold"]
    scaf_seq_name = sample_data.loc[sample_name, "Sequence"]
    scaf_seq_file = get_fasta_full_name(scaf_seq_name)
    scaf_seq = read_fasta(scaf_seq_file)
    # Read the sequence of the RRE.
    rre_seq_file = get_fasta_full_name("RRE")
    rre_seq = read_fasta(rre_seq_file)
    # Read mutation rates of RRE.
    ref = "RRE"
    em_dir = "EM_Clustering"
    start, end = amplicons[ref][0]
    rre_mus = dict()
    for k in range(1, n_rre_clusters + 1):
        run = sample_clustering_runs.get((sample_name, k))
        structure_order = list()
        for c in range(1, k + 1):
            dot_file = get_dot_file_full_name(proj_dir, em_dir, rre_id, ref, start, end, k, c, run=run)
            name, seq, structure = read_dot_file(dot_file)
            structure_name = rre_ref_structs_dot_to_name[structure]
            structure_order.append(structure_name)
        rre_mus_file = get_mu_file_full_name(proj_dir, em_dir, rre_id, ref, start, end, k, run=run)
        rre_mus[k] = read_cluster_mus(rre_mus_file, flatten=False, seq=rre_seq, only_ac=True)
        rre_mus[k].columns = structure_order
    # Read mutation rates of origami.
    k = 1
    origami_mus = pd.Series(dtype=np.float64)
    for amplicon in amplicons[scaf_name]:
        start, end = amplicon
        mus_file = get_mu_file_full_name(proj_dir, "EM_Clustering", origami_id, scaf_seq_name, start, end, k)
        amplicon_mus = read_cluster_mus(mus_file, flatten=True, seq=scaf_seq, only_ac=True)
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


def disassemble_params(params):
    k = (len(params) + 1) // 3
    assert len(params) + 1 == k * 3
    a_params = params[0: k]
    b_params = params[k: 2 * k]
    prop_given = params[2 * k:]
    prop_all = np.hstack([prop_given, [1 - np.sum(prop_given)]])
    return a_params, b_params, prop_all


def beta_loglike(mus, params):
    a_params, b_params, prop_all = disassemble_params(params)
    mus = np.linspace(0, 1, 10)
    mu_filter = np.logical_and(0 < mus, mus < 1)
    likelihoods = np.array([beta.pdf(mus[mu_filter], a, b) for a, b in zip(a_params, b_params)])
    loglike = np.sum(np.log(np.dot(likelihoods.T, prop_all)))
    print(loglike)
    return loglike


def fit_betas_to_mus(mus, k, eps=0.01):
    assert eps > 0
    n_mus = len(mus)
    mean = np.mean(mus)
    var = np.var(mus)
    """
    For beta distribution, we have
    mean = a / (a + b)
    var = a*b / ( (a + b)^2 * (a + b + 1) )
    Solving for a and b:
    """
    a = -mean * (mean**2 - mean + var) / var
    b = a * (1 - mean) / mean
    delta = eps * (k - 1) / 2
    a_params = np.linspace(a - delta, a + delta, k)
    b_params = np.linspace(b - delta, b + delta, k)
    proportions = np.ones(k - 1) / k
    params_init = np.hstack([a_params, b_params, proportions])
    bounds = [(1, None)] * (2 * k) + [(0, 1)] * (k - 1)
    def objective(params):
        return -beta_loglike(mus, params)
    result = minimize(objective, params_init, bounds=bounds)
    a_params, b_params, prop = disassemble_params(result.x)
    n_bins = 101
    n_dist = 1001
    x = np.linspace(0, 1, n_dist)
    plt.hist(mus, bins=n_bins)
    for a, b, p in zip(a_params, b_params, prop):
        plt.plot(x, beta.pdf(x, a, b) * n_mus / n_bins * p)
    plt.savefig("dist.png")
    plt.close()


if __name__ == "__main__":
    # Remove existing plots to start with a fresh slate.
    os.system(f"rm {proj_dir}/Analysis/*pdf")
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
    for seq_name in sample_data["Sequence"]:
        if seq_name not in scaf_seqs:
            scaf_seqs[seq_name] = read_fasta(get_fasta_full_name(seq_name))
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
    for scaf, group in scaf_groups.items():
        # Plot all the origamis in the group vs each other.
        group_df = pd.DataFrame.from_dict({x: sample_mus[x] for x in group})
        print(group_df.corr()**2)
        #sns.pairplot(group_df)
        plt.savefig(f"{scaf}_group.pdf")
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
    # Also plot rT66v2 with and wo staple 10.
    rT66v2_df = pd.concat([sample_mus["rT66v2"], sample_mus["rT66v2-10"]], axis=1)
    rT66v2_df["Target"] = rT66v2_df.index.isin(rT66v2_stap10_target)
    plt.scatter(sample_mus["rT66v2"], sample_mus["rT66v2-10"], c=rT66v2_df["Target"])
    plt.savefig("rT66v2.pdf")
    plt.close()
    # Obtain the geometric information for each origami.
    all_origamis = list(itertools.chain(*origamis.values()))
    origamis_dir = os.path.join(proj_dir, "Origamis")
    all_locations = dict()
    all_edges = dict()
    all_gs = dict()
    scaf_chain = "A"
    skip_origamis = ["rT55", "rT77"]
    for origami in all_origamis:
    #for origami in origamis["EGFP"]:
        print(origami)
        if origami in skip_origamis:
            # Skip these for now because Ariadne is not yet able to process odd edge lengths.
            continue
        # Get the location feature of each base in the origami.
        origami_dir = os.path.join(origamis_dir, sample_data.loc[origami, "Directory"])
        edges, g_up, g_dn, g_ax, base_info, dssr_info = ariadne.analyze_design(origami_dir, compute_bond_lengths=False, clobber=True)
        location_strings = base_info.loc[base_info.loc[:, "PDB chain"] == scaf_chain, "location"]
        base_info["base"] = scaf_seqs_df[sample_data.loc[origami, "Sequence"]].loc[1: len(base_info["base"])] #FIXME adjusting the sequence is necessary for origamis for which I don't have the DAEDALUS output with the right scaffold sequence -- if all were correct then this step would be unnecessary
        locations_split = location_strings.str.split("_", expand=True)
        locations_split.columns = ["Strand", "Location", "Direction"]
        assert np.all(locations_split["Strand"] == "Scaffold")
        locations = locations_split.apply(lambda base: base["Location"] if str(base["Direction"]) == "0" else f"{base['Location']} {base['Direction']}'", axis=1)
        locations.name = "Location"
        # Record the structural features.
        all_locations[origami] = locations
        all_edges[origami] = edges
        all_gs[origami] = {"up": g_up, "dn": g_dn, "ax": g_ax}
        # Combine the DMS signals, sequence, and features into one dataframe.
        seq_name = sample_data.loc[origami, "Sequence"]
        origami_seq = scaf_seqs_df[seq_name]
        origami_seq.name = "Sequence"
        origami_mus = sample_mus[origami]
        origami_mus.name = "DMS"
        origami_data = origami_seq.to_frame().join([locations, origami_mus], how="inner")
        assert np.all(origami_data["Sequence"].isin(dms_bases))
        # 
        
        zero_dms = np.isclose(origami_data["DMS"], 0.0)
        origami_data = origami_data.loc[~zero_dms]
        formula_lhs = "np.log10(DMS)"
        formula_rhs = "C(Sequence, levels=dms_bases) + C(Location)"
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
        lm = ols(f"{formula_lhs} ~ {formula_rhs}", data=origami_data).fit()
        print(lm.summary())
        """
        coeffs[name] = lm.params
        conf_ints[name] = lm.conf_int(conf_alpha)
        p_vals[name] = lm.pvalues
        r2s[name] = lm.rsquared
        """
        fitted = lm.fittedvalues
        residuals = lm.resid
        plt.scatter(fitted, residuals)
        plt.title(origami)
        plt.savefig("fr.png")
        plt.close()
        secondary_structure_fname = f"{origami}_ss.png"
        color_max = 0.1
        colors = np.minimum(origami_mus / color_max, 1.0)
        plots.secondary_structure_signal(secondary_structure_fname, edges, g_up, g_dn, g_ax, base_info, colors)
        #origami_data = pd.DataFrame.from_dict({"Origami": origami, "Sequence": scaf_seqs[seq_name], "Location": locations,
        #fit_betas_to_mus(origami_mus, 1)
