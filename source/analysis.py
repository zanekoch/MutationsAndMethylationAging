import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
#from sklearnex import patch_sklearn
#patch_sklearn()
from scipy import stats
import sys
import os 
import seaborn as sns
from matplotlib.ticker import PercentFormatter
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# CONSTANTS
VALID_MUTATIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G", "G>C","G>A", "A>T", "A>G" , "A>C", "G>T", "C>-"]
JUST_CT = True
DATA_SET = "TCGA"

import utils

def plot_mutations_distributions(all_mut_df, illumina_cpg_locs_df, all_methyl_df_t):
    """
    For each dataset plot distribtion of mutation types, also for just mutations in illumina measured CpG sites
    """
    sns.set_context('paper')
    # CpG vs non CpG mutation frequency
    # get all mutations in measured CpG sites
    fig, axes = plt.subplots(figsize=(4,6),  dpi =100)
    cpg_mut_num = all_mut_df.loc[all_mut_df['is_cpg'] == True].shape[0]
    all_mut_num = all_mut_df.shape[0]
    NUM_CPG = 28299634
    BP_NUM = 3137144693
    expected_cpg = NUM_CPG/BP_NUM
    expected_non_cpg = 1 - expected_cpg
    cpg_mut_freq = pd.DataFrame({'Mutation class': ['CpG', 'Expected\nCpG', 'non-CpG', 'Expected\nnon-CpG'], 'mut_freq': [cpg_mut_num/all_mut_num,expected_cpg, 1-cpg_mut_num/all_mut_num, expected_non_cpg]})
    print(cpg_mut_freq)
    p = sns.barplot(x=cpg_mut_freq['Mutation class'], y=cpg_mut_freq['mut_freq'], ax=axes, palette=['white', 'black','white', 'black' ], edgecolor='black', errorbar=None)
    # remove x label 
    axes.set_xlabel("")
    axes.set_ylabel("")
    axes.set_ylim([0,1])
    # angle x ticks
    for tick in axes.get_xticklabels():
        tick.set_rotation(45)




def compare_mf_mutated_sample_vs_avg(mutation_in_measured_cpg_df, all_methyl_df_t):
    """
    Plot MF at sites of mutation event vs same site with no mutation. Write pvals of testing difference of distribution between mutated and not 
    """
    # output plots of avg vs not as seaborn kde's
    sns.set_context('paper')
    
    non_mutated_methyl_df_t = all_methyl_df_t.loc[:, ~all_methyl_df_t.columns.isin(mutation_in_measured_cpg_df['#id'])]
    
    # limit to only big DNA_VAF mutations
    print("Number of mutations in measured CpG sites: {}".format(len(mutation_in_measured_cpg_df)))
    # select the 1000 rows with largest DNA_VAF
    mutation_in_measured_cpg_df = mutation_in_measured_cpg_df.sort_values(by=['DNA_VAF'], ascending=False).iloc[:500]
    #mutation_in_measured_cpg_df = mutation_in_measured_cpg_df.loc[mutation_in_measured_cpg_df['DNA_VAF'] >.35]
    print("selected 1000 mtuations events with largest DNA_VAF, minimum DNA_VAF: {}".format(mutation_in_measured_cpg_df['DNA_VAF'].min()))
    # plot distribution of MF at sites of mutation event vs same site with no mutation
    to_plot_df = pd.DataFrame(pd.concat(
        [mutation_in_measured_cpg_df['avg_methyl_frac'],
         mutation_in_measured_cpg_df['methyl_fraction'],
         non_mutated_methyl_df_t.mean(axis=0)
         ], axis=0)).reset_index(drop=True)
    
    to_plot_df.columns = ['Methylation Fraction']
    to_plot_df['Type'] = ['Non mutated CpGs'] * len(mutation_in_measured_cpg_df['avg_methyl_frac']) +  ['Mutated CpGs'] * len(mutation_in_measured_cpg_df['methyl_fraction']) + ['Site of no CpG mutation'] * len(non_mutated_methyl_df_t.mean(axis=0))
    # all together
    fig, axes = plt.subplots(dpi=100, figsize=(8,5))
    p = sns.kdeplot(
        data=to_plot_df, x='Methylation Fraction', hue='Type', fill=True,
        common_norm=False, clip=[0,1], palette = ['steelblue', 'maroon', 'grey'], 
        ax=axes, legend=False
        )
    #plt.savefig("/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure1/figure1E_kde_methyl_distr.svg",format='svg', dpi = 300)
    #plt.savefig("/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/supplemental/icgc_figure1E_kde_methyl_distr.svg",format='svg', dpi = 300)

    # just Non mutated CpGs vs Site of no CpG mutation
    """fig, axes = plt.subplots(dpi=100, figsize=(8,5))
    p = sns.kdeplot(
        data=to_plot_df[to_plot_df['Type'].isin(['Non mutated CpGs', 'Site of no CpG mutation'])], x='Methylation Fraction', hue='Type', fill=True,
        common_norm=False, clip=[0,1], palette = ['steelblue', 'grey'], 
        ax=axes, legend=False
        )
    #plt.savefig("/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure1/figure1E_kde_methyl_distr_only_blueGrey.svg",format='svg', dpi = 300)
    #plt.savefig("/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/supplemental/icgc_figure1E_kde_methyl_distr_only_blueGrey.svg",format='svg', dpi = 300)
    # just Site of no CpG mutation
    fig, axes = plt.subplots(dpi=100, figsize=(8,5))
    p = sns.kdeplot(
        data=to_plot_df[to_plot_df['Type'].isin(['Site of no CpG mutation'])], x='Methylation Fraction', hue='Type', fill=True,
        common_norm=False, clip=[0,1], palette = ['grey'], 
        ax=axes, legend=False
        )"""
    #plt.savefig("/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure1/figure1E_kde_methyl_distr_only_grey.svg",format='svg', dpi = 300)
    #plt.savefig("/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/supplemental/icgc_figure1E_kde_methyl_distr_only_grey.svg",format='svg', dpi = 300)
    
    return to_plot_df

def compare_mf_site_of_mutation_vs_not(mutation_in_measured_cpg_df, all_methyl_df_t, out_dir):
    # test for difference between average methylation fraction at non-mutated CpG sites and at mutated CpG sites in non-mutated samples
    non_mutated_methyl_df_t = all_methyl_df_t[all_methyl_df_t.columns[~all_methyl_df_t.columns.isin(mutation_in_measured_cpg_df['#id'])]]
    with open(os.path.join(out_dir, "methylation_fraction_results.txt"), "a+") as f:
        statistic, p_val = stats.ranksums(non_mutated_methyl_df_t.mean().to_numpy(), mutation_in_measured_cpg_df['avg_methyl_frac'].to_numpy(), alternative='less')
        f.write("Wilcoxon rank sum p-value testing if the dsitr. of average methylation fraction at non-mutated CpG sites is lesser than at mutated CpG sites in non-mutated samples {} and statistic {}\n".format(p_val, statistic))
        f.write("mean average methylation fraction at non-mutated {} mutated CpG sites in non-mutated samples {}".format(non_mutated_methyl_df_t.mean().mean(),mutation_in_measured_cpg_df['avg_methyl_frac'].mean() ))
        # fisher
        non_mut_less = len(non_mutated_methyl_df_t.mean()[non_mutated_methyl_df_t.mean()<=.5])
        mut_loc_less = len(mutation_in_measured_cpg_df['avg_methyl_frac'] <= .5)
        non_mut_greater = len(non_mutated_methyl_df_t.mean()[non_mutated_methyl_df_t.mean()>.5])
        mut_loc_greater = len(mutation_in_measured_cpg_df['avg_methyl_frac']>.5)
        contingency_table = [[non_mut_less, mut_loc_less],[non_mut_greater, mut_loc_greater]]
        oddsr, p = stats.fisher_exact(table=contingency_table, alternative='less')
        f.write("Fisher p-value for dsitr. of average methylation fraction at non-mutated CpG sites has greater proportion <.5 than at mutated CpG sites in non-mutated samples: {}".format(p))

    # histogram of average methylation fraction at sites with C>T mutation vs without
    fig, axes = plt.subplots(facecolor="white", dpi=200)
    weights = np.ones_like(mutation_in_measured_cpg_df['avg_methyl_frac']) / len(mutation_in_measured_cpg_df['avg_methyl_frac'])
    mutation_in_measured_cpg_df['avg_methyl_frac'].plot.hist(weights=weights,bins=12, ax = axes,alpha=.7, color = 'goldenrod')
    weights = np.ones_like(non_mutated_methyl_df_t.loc['mean']) / len(non_mutated_methyl_df_t.loc['mean'])
    non_mutated_methyl_df_t.loc['mean'].plot.hist(weights = weights,bins=12, ax = axes, alpha=.7, color='dimgray')
    axes.legend(["Sites of C>T mutation events\n(including mutated samples)", "Sites of no C>T mutation\n event"])
    axes.set_ylabel("Probability")
    axes.set_xlabel("Mean methylation fraction")
    fig.savefig(os.path.join(out_dir, 'non_mut_vs_mut_site_mf.png'))

def get_same_age_tissue_means(mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t, age_bin_size = 10):
    """
    Get the average MF at the mutated sites in the same age bin range and tissue type as the sample where the mutation occured
    """
    # for each mutation in mut_in_measured_cpg_w_methyl_age_df, get the average MF at the same CpG site in the same age bin range and tissue type as the sample where the mutation occured
    # get the average MF at the same CpG site in the same age bin range and tissue type as the sample where the mutation occured

    mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac'] = mut_in_measured_cpg_w_methyl_age_df.apply(
        lambda mut_row: all_methyl_age_df_t[
            (np.abs(all_methyl_age_df_t['age_at_index'] - mut_row['age_at_index']) <= age_bin_size/2) 
            & (all_methyl_age_df_t['dataset'] == mut_row['dataset'])
            ][mut_row['#id']].mean(), axis=1)
    """ #Prepare all_methyl_age_df_t by filtering out rows with age difference larger than age_bin_size/2
    merged_df = all_methyl_age_df_t.copy()
    merged_df['age_diff'] = np.abs(merged_df['age_at_index'].values[:, None] - mut_in_measured_cpg_w_methyl_age_df['age_at_index'].values)
    merged_df = merged_df[merged_df['age_diff'] <= age_bin_size/2]

    # Merge the two DataFrames on the 'dataset' column
    merged_df = mut_in_measured_cpg_w_methyl_age_df.merge(merged_df, on='dataset', suffixes=('', '_y'))

    # Group by the original index and calculate the mean value for each group
    grouped_df = merged_df.groupby(merged_df.index)

    # Map the mean values to the original DataFrame
    mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac'] = grouped_df.apply(lambda group: group['#id_y'].mean())"""

    return mut_in_measured_cpg_w_methyl_age_df

def methylation_fraction_comparison(
    all_mut_df, illumina_cpg_locs_df, all_methyl_df_t,
    out_dir, all_meta_df, age_bin_size = 10
    ):
    """
    Measure the effect a mutation has on MF at that site
    @ returns: pandas dataframe of all mutations in illumina measured CpG sites, their methylation fraction in mutated sample, and average methylation at that site across other samples (within 5 years of age)
    """
    # get just C>T mutations
    if JUST_CT:
        mutations_df = all_mut_df[all_mut_df['mutation'] == 'C>T']
    else:
        mutations_df = all_mut_df
        
    #for each CpG with methylation data get genomic location of its C
    mut_in_measured_cpg_df = utils.join_df_with_illum_cpg(mutations_df, illumina_cpg_locs_df, all_methyl_df_t)
    methyl_fractions = utils.get_methyl_fractions(mut_in_measured_cpg_df, all_methyl_df_t)
    mut_in_measured_cpg_df['methyl_fraction'] = methyl_fractions

    # add ages and datasets to both dfs
    mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(mut_in_measured_cpg_df, all_meta_df, all_methyl_df_t)

    # get rid of samples that do not have methylayion data
    mut_in_measured_cpg_w_methyl_age_df = mut_in_measured_cpg_w_methyl_age_df[mut_in_measured_cpg_w_methyl_age_df['methyl_fraction'] != -1]
    # drop samples with nan age value
    mut_in_measured_cpg_w_methyl_age_df = mut_in_measured_cpg_w_methyl_age_df.dropna(subset=['age_at_index'])
    
    # get means 
    """mut_in_measured_cpg_w_methyl_age_df = get_same_age_tissue_means(
        mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t, age_bin_size = age_bin_size
        )"""
    # old way
    mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac'] = all_methyl_df_t[mut_in_measured_cpg_w_methyl_age_df['#id']].mean().values

    # get difference between mean and mutated sample
    mut_in_measured_cpg_w_methyl_age_df['difference'] = mut_in_measured_cpg_w_methyl_age_df['methyl_fraction'] - mut_in_measured_cpg_w_methyl_age_df['avg_methyl_frac']
    # test for a difference
    to_plot_df = compare_mf_mutated_sample_vs_avg(mut_in_measured_cpg_w_methyl_age_df, out_dir, all_methyl_df_t)
    """compare_mf_site_of_mutation_vs_not(mut_in_measured_cpg_w_methyl_age_df, all_methyl_df_t, out_dir)"""
    return mut_in_measured_cpg_w_methyl_age_df, to_plot_df

def main(illumina_cpg_locs_df, out_dir, all_mut_df, all_methyl_df_t, all_meta_df, dataset_names_list, age_bin_size):
    # make output directories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "bootstrap"), exist_ok=True)

    # do mutation analysis 
    #mut_in_measured_cpg_df = plot_mutations(all_mut_df, all_meta_df, dataset_names_list, out_dir, illumina_cpg_locs_df, all_methyl_df_t)
    # subset to only C>T mutations
    # TODO: remove this return if possible
    
    mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t = methylation_fraction_comparison(all_mut_df, illumina_cpg_locs_df, all_methyl_df_t, out_dir, all_meta_df, age_bin_size = age_bin_size)

    #return mut_in_measured_cpg_w_methyl_age_df, all_methyl_age_df_t
    
    
