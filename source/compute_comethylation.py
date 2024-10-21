import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import utils
import seaborn as sns
from statsmodels.stats.weightstats import ztest as ztest
from rich.progress import track
import os
import dask
import pyarrow as pa
import sys
from tqdm import tqdm
from collections import defaultdict
import random
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy.stats import spearmanr, pearsonr

plt.style.use("seaborn-deep")
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


CHROM_LENGTHS = {
    '1': 248956422,'2': 242193529,'3': 198295559, '4': 190214555,
    '5': 181538259,'6': 170805979,'7': 159345973,'8': 145138636,
    '9': 138394717,'10': 133797422,'11': 135086622,'12': 133275309,
    '13': 114364328,'14': 107043718,'15': 101991189,'16': 90338345,
    '17': 83257441,'18': 80373285,'19': 58617616,'20': 64444167,
    '21': 46709983,'22': 50818468,
}

class analyzeComethylation:
    """
    For analysis of a completed mutationScan run
    """
    def __init__(self):
        pass
    
    def get_mean_metrics_by_dist(
        self, 
        all_metrics_df, 
        dist_starts = [0, 10000, 50000, 100000, 500000, 1000000], 
        window_size = 10000, 
        use_window_size = False,
        negative_values_seperate = False
        ):
        """
        @ all_metrics_df: pd.DataFrame of the metrics for each mutation event (Actual mutation events and  background events) from a mutationScan run
        @ absolute_distances: list of distances to get the mean metrics within
        @ use_window_size: bool, wether to only calc marginal (so when doing dist 1000, only calc dmf for 500-1000 dist sites)
        @ return: pd.DataFrame of the mean metrics for each mutation event within each distance
        """
        if negative_values_seperate:
            neg_dist_starts = [x * -1 for x in dist_starts[1:]]
            # combine negative and positive distances on after the other
            dist_starts = sorted(neg_dist_starts + dist_starts)
        
        all_metrics_df['abs_delta_mf_median'] = all_metrics_df['delta_mf_median'].abs()
        # subset to only metrics we need
        all_metrics_df = all_metrics_df[['mut_event', 'delta_mf_median', 'abs_delta_mf_median', 'measured_site_dist', 'is_background', 'index_event', 'mutated_sample', 'measured_site']]  #'sample', 
        # for each distance, get the mean and median of the delta_mf_median and abs_delta_mf_median of sites within that distance
        mean_metric_by_dist_dfs = []
        for i, dist in enumerate(dist_starts):
            if use_window_size:
                if negative_values_seperate:
                    if dist < 0: # choose sites between -10000 and -11000 for dist -10000 and window 1000, for example
                        subset_df = all_metrics_df.loc[
                            (all_metrics_df['measured_site_dist'] < dist) 
                            & (all_metrics_df['measured_site_dist'] >= dist - window_size)
                            ]
                        print(f"num {dist} dist sites", subset_df.shape[0])
                        
                    elif dist == 0: # choose site on either side of 0
                        subset_df = all_metrics_df.loc[
                            all_metrics_df['measured_site_dist'].abs() <= window_size/2
                            ]
                        print("num 0 dist sites", subset_df.shape[0])
                    else:
                        subset_df = all_metrics_df.loc[
                            (all_metrics_df['measured_site_dist'] > dist - window_size)
                            & (all_metrics_df['measured_site_dist'] <= dist)
                            ]
                        print(f"num {dist} dist sites", subset_df.shape[0])
                else:
                    # subset to sites at least absolute_distances[i-1] and at most dist away
                    subset_df = all_metrics_df.loc[
                        (all_metrics_df['measured_site_dist'].abs() > dist)
                        & (all_metrics_df['measured_site_dist'].abs() <= dist + window_size)
                        ]
            else:
                # subset to only sites within the distance
                subset_df = all_metrics_df.loc[all_metrics_df['measured_site_dist'].abs() <= dist]
                
            # group by mutation event
            grouped_subset_df = subset_df.groupby(['mut_event'])
            # get the mean and median of the delta_mf_median and abs_delta_mf_median
            mean_by_sample = grouped_subset_df[['delta_mf_median', 'abs_delta_mf_median']].mean()
            mean_by_sample.columns = ['mean_dmf', 'mean_abs_dmf']
            median_by_sample = grouped_subset_df[['delta_mf_median','abs_delta_mf_median']].median()
            median_by_sample.columns = ['median_dmf', 'median_abs_dmf']
            ## count by sample
            count_by_sample = grouped_subset_df['measured_site'].nunique()
            count_by_sample.name = 'num_measured_sites'
            # get weighted means
            # linear
            """weighted_mean_by_sample = grouped_subset_df.apply(
                lambda x: np.average(x['delta_mf_median'], weights=(1/x['measured_site_dist'].abs()))
                ).to_frame().rename(columns={0: 'weighted_mean_dmf'})"""
            """abs_weighted_mean_by_sample = grouped_subset_df.apply(
                lambda x: np.average(x['abs_delta_mf_median'], weights=(1/x['measured_site_dist'].abs()))
                ).to_frame().rename(columns={0: 'weighted_mean_abs_dmf'})"""
            # log
            """log_weighted_mean_by_sample = grouped_subset_df.apply(
                lambda x: np.average(x['delta_mf_median'], weights=(1/np.log10(x['measured_site_dist'].abs())))
                ).to_frame().rename(columns={0: 'log_weighted_mean_dmf'})"""
            """
            log_abs_weighted_mean_by_sample = grouped_subset_df.apply(
                lambda x: np.average(x['abs_delta_mf_median'], weights=(1/np.log(x['measured_site_dist'].abs())))
                ).to_frame().rename(columns={0: 'log_weighted_mean_dmf'})
            # gaussian
            gaussian_weighted_mean_by_sample = grouped_subset_df.apply(
                lambda x: np.average(x['delta_mf_median'], weights=(stats.norm.pdf(x['measured_site_dist'].abs()))))"""
            
            # merge mean and median dfs
            mean_metrics_df = mean_by_sample.merge(median_by_sample, left_index=True, right_index=True)
            mean_metrics_df['distance'] = dist
            mean_metrics_df = mean_metrics_df.merge(count_by_sample, left_index=True, right_index=True)
            # merge weighted mean and median
            """mean_metrics_df = mean_metrics_df.merge(weighted_mean_by_sample, left_index=True, right_index=True)
            mean_metrics_df = mean_metrics_df.merge(log_weighted_mean_by_sample, left_index=True, right_index=True)"""
            """mean_metrics_df = mean_metrics_df.merge(abs_weighted_mean_by_sample, left_index=True, right_index=True)
            mean_metrics_df = mean_metrics_df.merge(log_weighted_mean_by_sample, left_index=True, right_index=True)
            mean_metrics_df = mean_metrics_df.merge(log_abs_weighted_mean_by_sample, left_index=True, right_index=True)"""
            mean_metric_by_dist_dfs.append(mean_metrics_df)
            if i % 10 == 0:
                print(f'Finished {i} of {len(dist_starts)} distances')
        # combine all the mean metrics dfs
        mean_metrics_by_dist_df = pd.concat(mean_metric_by_dist_dfs, axis=0)

        mut_event_to_background_map = all_metrics_df.drop_duplicates(subset=['mut_event'])
        mut_event_to_background_map.set_index(['mut_event'], inplace=True)
        mean_metrics_by_dist_df = mean_metrics_by_dist_df.merge(
            mut_event_to_background_map[['is_background', 'index_event', 'mutated_sample']], 
            left_index=True, right_index=True, how = 'left'
            )
        mean_metrics_by_dist_df = mean_metrics_by_dist_df.reset_index()
        return mean_metrics_by_dist_df
    
    def plot_delta_mf_kdeplot(
        self, 
        mean_metrics_df, 
        metric,
        axes,
        consortium,
        out_fn,
        line_label:str = "",
        line_color:str = "black"
        ):
        """
        
        """
        # increase font size
        #sns.set_theme(style='white', font_scale=1.3, rc={'xtick.bottom': True, 'ytick.left': True})
        sns.set_context('paper')
        #fig, axes = plt.subplots(figsize=(8, 3.5), dpi = 100)
        mut = mean_metrics_df.loc[mean_metrics_df.mutated_sample == True]
        mut = mut.rename(
            columns={'is_background': 'Locus'}
            )
        sns.kdeplot(
            data=mut, x=metric, hue='Locus',
            common_norm=False, palette=['maroon', 'steelblue'],
            fill=True, ax = axes[0], clip = (-1, 1), common_grid=False, legend = False,
            gridsize=1000, bw_adjust=1.5, #cut = 1,
            )
        # create 30 bin edges from -.75 to .75
        bins = np.linspace(-.75, .75, 28)
        """sns.histplot(
            data=mut, x=metric, hue='Locus',
            common_norm=False, palette=['maroon', 'steelblue'],
            fill=True, ax = axes[0],  legend = False, bins=bins, stat='probability',
            element= 'step',
            #kde = True, kde_kws={'clip': (-1, 1), 'gridsize': 1000}
            #gridsize=1000
            )"""
        axes[0].set_xlim(-.45, .45)
        
        # get counts of observations in each bin, for each locus
        counts = mut.groupby(['Locus', pd.cut(mut[metric], bins)]).size().unstack(fill_value=0).T
        counts['mut_prop'] = counts[False] / counts[False].sum()
        counts['bg_prop'] = counts[True] / counts[True].sum()
        counts['ratio'] = counts['mut_prop'] / counts['bg_prop']

        if consortium == 'ICGC':
            counts.loc[counts[False] < 4, 'ratio'] = np.nan
            # set y ticks tp be 0, 1, 2,3,4
            axes[1].set_yticks([0, 2, 4, 6, 8])
            # set y tick labels to be 0, 1, 2, 3, 4
            axes[1].set_yticklabels([0, 2, 4, 6, 8])
            #axes[1].set_ylim(0,5)
        elif consortium == 'TCGA':
            counts.loc[counts[False] < 15, 'ratio'] = np.nan
        else:
            raise ValueError('consortium must be TCGA or ICGC')
        # plot ratio as a lineplot with the same x axis and dots at each observation
        sns.lineplot(
            data=counts, x=bins[1:] - np.abs((bins[0] - bins[1]))/2, y='ratio', 
            ax = axes[1], color=line_color,
            marker='o', markersize=6, markeredgewidth=.3, markeredgecolor='black'
            )
        # add legend label to lineplot label
        if line_label != "":
            axes[1].legend([line_label])

        # plot dashed line as y=1
        axes[1].axhline(y=1, color='black', linestyle='--')
        #axes[1].set_ylim(0, 6)

        # set xlim
        #axes.set_yscale('log')
        # write delta in geek sybol
        axes[1].set_xlabel(r'Median $\Delta$MF across locus')
        axes[1].set_ylabel('Fold enrichment')
        axes[1].set_ylim(0, 8)
        
        # change legend labels
        #axes.legend(['Random', 'Mutated'], loc='upper right', title='Locus')
        # save as an svg
        if out_fn != "":
            plt.savefig(fname=out_fn, format='svg', dpi = 300)
        return counts
        
    
    def add_mutation_info_to_mean_metrics_df(
        self,
        mean_metrics_df: pd.DataFrame,
        distance: int, # distance for mean metrics df
        consortium: str
        ):
        """
        Read in the mutaiton info df which has all columns and merge it with the mean metrics df
        
        @ mean_metrics_df: pd.DataFrame of the mean metrics for each mutation event within each distance
        @ distance: distance for mean metrics df
        @ return: pd.DataFrame of the mean metrics for each mutated sample (RC of FG) mutation event within the specified distance with additional columns such as gc_percetange, mutation type, etc.
        """
        VALID_MUTATIONS = [
            "C>A", "C>G", "C>T", "T>A", "T>C", "T>G", "G>C",
            "G>A", "A>T", "A>G" , "A>C", "G>T", "C>-"
            ]
        if consortium == 'TCGA':
            # read in mutation data
            all_mut_all_col_df = pd.read_csv(
                '/cellar/users/zkoch/methylation_and_mutation/data/final_tcga_data/PANCAN_mut.tsv.gz',
                sep = '\t'
                )
            all_mut_all_col_df['sample'] = all_mut_all_col_df['sample'].str[:-3]
        elif consortium == 'ICGC':
            all_mut_all_col_df = pd.read_csv(
                '/cellar/users/zkoch/methylation_and_mutation/data/final_icgc_data/icgc_mut_df.csv.gz',
                sep = '\t'
                )
        elif consortium == 'TCGA-normal':
            all_mut_all_col_df = pd.read_csv(
                '/cellar/users/zkoch/methylation_and_mutation/data/final_tcga_data/TCGA_normal_mut.tsv.gz',
                sep = '\t'
                )
        else:
            raise ValueError('consortium must be TCGA or ICGC')
        all_mut_all_col_df.rename({'sample': 'case_submitter_id'}, axis = 1, inplace = True)
        all_mut_all_col_df["mut_type"] = all_mut_all_col_df["reference"] + '>' + all_mut_all_col_df["alt"]
        all_mut_all_col_df = all_mut_all_col_df.loc[all_mut_all_col_df["mut_type"].isin(VALID_MUTATIONS)]
        all_mut_all_col_df['mut_event'] = all_mut_all_col_df['case_submitter_id'].astype(str) + '_' + all_mut_all_col_df['chr'].astype(str) + ':' + all_mut_all_col_df['start'].astype(str)  
        
        
        # load reference and CGI
        """from pyfaidx import Fasta
        reference_genome = Fasta(
            '/cellar/users/zkoch/methylation_and_mutation/data/genome_annotations/hg19.fa'
            )
        cpg_islands = pd.read_csv(
            '/cellar/users/zkoch/methylation_and_mutation/data/genome_annotations/CpG_islands_hg19.bed.gz',
            sep = '\t', header = None
            )
        cpg_islands.columns = ['chr', 'start', 'end', 'name']
        cpg_islands['chr'] = cpg_islands['chr'].str.replace('chr', '')"""
        
        # merge
        if distance != -1:
            mean_metrics_w_mut = mean_metrics_df.loc[
                mean_metrics_df['mutated_sample']
                #& (dist_mean_metrics_df['is_background'] == False)
                & (mean_metrics_df['distance'] == distance)
                ].merge(all_mut_all_col_df, left_on = 'mut_event', right_on = 'mut_event', how = 'left')
        else:
            mean_metrics_w_mut = mean_metrics_df.loc[
                mean_metrics_df['mutated_sample']
                ].merge(all_mut_all_col_df, left_on = 'mut_event', right_on = 'mut_event', how = 'left')
        # map to combined mutation types
        mean_metrics_w_mut['mut_type_combined'] = mean_metrics_w_mut['mut_type'].map({
            'C>A': 'C>A', 'G>T': 'C>A', 
            'C>G': 'C>G', 'G>C': 'C>G', 
            'C>T': 'C>T', 'G>A': 'C>T', 
            'T>A': 'T>A', 'A>T': 'T>A',
            'T>C': 'T>C', 'A>G': 'T>C',
            'T>G': 'T>G', 'A>C': 'T>G'
            })
        # remove deletions
        # mean_metrics_w_mut = mean_metrics_w_mut.loc[mean_metrics_w_mut['alt'] != '-']
        # add chr and start for BG muts, and add alt as Z
        mean_metrics_w_mut['start'] = mean_metrics_w_mut[
            'mut_event'].str.split('_').str[1].str.split(':').str[1].astype(int)
        mean_metrics_w_mut['chr'] = mean_metrics_w_mut[
            'mut_event'].str.split('_').str[1].str.split(':').str[0]
        mean_metrics_w_mut.loc[mean_metrics_w_mut['is_background'], 'alt'] = 'Z'
        
        # get sequences, was_cpg, becomes_cpg, is_cgi, and gc percent
        # start - 1 is the mutated position
        """mean_metrics_w_mut['seq'] = mean_metrics_w_mut.apply(
            lambda x: reference_genome['chr'+x['chr']][x['start']-2: x['start'] + 1].seq,
            axis = 1)
        # was or becomes cpg, either C or G can be removed (was_cpg) or added (becomes_cpg)
        mean_metrics_w_mut['was_cpg'] = mean_metrics_w_mut['seq'].str.upper().str.contains('CG')
        mean_metrics_w_mut['new_seq'] = mean_metrics_w_mut.apply(
            lambda row: row['seq'][0] + row['alt'] + row['seq'][2],
            axis = 1)
        mean_metrics_w_mut['becomes_cpg'] = mean_metrics_w_mut['new_seq'].str.upper().str.contains('CG')
        # is cgi
        is_cgi = ((mean_metrics_w_mut['chr'].values[:, np.newaxis] == cpg_islands['chr'].values) &
                (mean_metrics_w_mut['start'].values[:, np.newaxis] >= cpg_islands['start'].values) &
                (mean_metrics_w_mut['start'].values[:, np.newaxis] <= cpg_islands['end'].values)).any(axis=1)
        mean_metrics_w_mut['is_cgi'] = is_cgi
        # also get sequence in 200bp window around each mutation 
        mean_metrics_w_mut['seq_10kb'] = mean_metrics_w_mut.apply(
            lambda x: reference_genome['chr'+x['chr']][x['start']-10001: x['start'] + 10000].seq,
            axis = 1)
        # get gc content in 200bp window
        mean_metrics_w_mut['cpg_count_10kb'] = (
            mean_metrics_w_mut['seq_10kb'].str.upper().str.count('GC') \
            + mean_metrics_w_mut['seq_10kb'].str.upper().str.count('CG')
            )"""
        return mean_metrics_w_mut
        
    
    def plot_delta_mf_violin(
        self, 
        mean_metrics_df, 
        metric = 'median_dmf',
        axes = None
        ):
        PROPS = {
            'boxprops':{'facecolor':'none', 'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},
            'capprops':{'color':'black'},
            'zorder': 2
        }
        # increase font size
        #sns.set_theme(style='white', font_scale=1.3, rc={'xtick.bottom': True, 'ytick.left': True})
        sns.set_context('paper')
        if axes is None:
            fig, axes = plt.subplots(figsize=(7, 4), dpi = 100)
            
        mut = mean_metrics_df.loc[mean_metrics_df.mutated_sample == True]
        mut = mut.rename(
            columns={'is_background': 'Locus'}
            ).replace(
                {'Locus': {True: 'Random', False: 'Mutated'}}
                )
        mut['dummy'] = 0
        sns.violinplot(
            data=mut, y=metric, x = 'Locus', #split=True,
            common_norm=False, palette=['maroon', 'steelblue'],
            scale_hue=True, scale = 'area', 
            ax = axes, cut = 0, gridsize=1000,
            #inner = None, linewidth=0, zorder = 0,
            order = ['Mutated', 'Random']
            )
        sns.boxplot(
            data=mut, y=metric,  x = 'Locus', 
            showfliers=False, ax = axes, boxprops={"zorder": 2, 'facecolor':'none', 'edgecolor':'black', }, medianprops = {'color':'black'},
            capprops = {'color':'black'}, whiskerprops = {'color':'black'}, zorder = 2, linewidth=1.2, 
            order = ['Mutated', 'Random']
            )
        # set xlim
        axes.set_ylim(-.17, .17)
        # write delta in geek sybol
        axes.set_ylabel(r'Median $\Delta$MF across locus')
        axes.set_xlabel('Locus')
        
        # change legend labels and colors 
        axes.legend([], [], frameon=False)
        #axes.legend(['Random', 'Mutated'], loc='lower center', title='Locus')

    def plot_delta_mf_boxplots(
        self,
        mean_metrics_df, 
        metric = 'mean_dmf'
        ):
        #sns.set_theme(style='white', font_scale=1.3, rc={'xtick.bottom': True, 'ytick.left': True})
        sns.set_context('paper')

        fig, axes = plt.subplots(figsize=(8, 4), dpi=100)
        # Randomized control or actual mutation
        to_plot = mean_metrics_df[[metric, 'is_background', 'mutated_sample' ]].replace(
                                    {'is_background': {True: 'RC-',
                                                        False: ''}})
        # BG or ''
        to_plot = to_plot.replace({'mutated_sample': {True: 'Mutated sample', False: 'BG sample'}})
        to_plot['combined'] = to_plot['is_background'] + to_plot['mutated_sample']
        pallette = [ 'maroon', 'steelblue', 'white', 'white']
        sns.boxplot(
            data=to_plot, y=metric, x='combined',
            ax = axes, showfliers=False, orient='v',
            palette=pallette, 
            order = ['Mutated sample', 'RC-Mutated sample', 'BG sample', 'RC-BG sample']
            )
        # remove ylabel 
        axes.tick_params(axis='x', labelrotation=0)
        axes.set_xlabel('')
        # delta in geek symbol
        axes.set_ylabel(r'Mean $\Delta$MF across locus')
        
        
        
    def plot_distance_of_effect_lineplot(
        self,
        mean_metrics_df: pd.DataFrame,
        all_metrics_df: pd.DataFrame,
        num_top_muts: int = 1000,
        smoothing_window_size_dist: int = 10000,
        smoothing_window_size_corr: int = 300,
        corr_dist: str = 'measured_site_dist',
        dist: int = 100000,
        plot_bg: bool = True,
        out_fn: str = None,
        corr_vs_dist: bool = False,
        illumina_cpg_locs_df: pd.DataFrame = None,
        rank_order_dist = False,
        plot_biggest_bg = False
        ):
        """
        @ mean_metrics_df: DataFrame with mean metrics for each mutation event
        @ all_metrics_df: DataFrame with all metrics for each mutation, without matched samples
        """
        # select only a certain distance of median rows 
        filtered_df = mean_metrics_df.query(
            'distance == @dist & mutated_sample == True & is_background == False'
            )
        # sort them in order of median_dmf
        sorted_df = filtered_df.sort_values(by='median_dmf', ascending=False)
        # either get all or the top & bottom X rows
        if num_top_muts == -1:
            pos_indices = sorted_df.loc[sorted_df['median_dmf'] > 0].index
            neg_indices = sorted_df.loc[sorted_df['median_dmf'] < 0].index
        else:
            neg_indices = sorted_df.index[-num_top_muts:]
            pos_indices = sorted_df.index[:num_top_muts]
        # get the 'mut_event' identifiers for these top events
        ################# CHANGE indices if want positive #############################
        biggest_neg_effect = sorted_df.loc[neg_indices, 'mut_event'].values
        biggest_pos_effect = sorted_df.loc[pos_indices, 'mut_event'].values
        
           
        if plot_biggest_bg:
            # do this same thing for bg rows
            filtered_bg_df = mean_metrics_df.query(
                'distance == @dist & mutated_sample == True & is_background == True'
                )
            sorted_bg_df = filtered_bg_df.sort_values(by='median_dmf', ascending=False)
            if num_top_muts == -1:
                pos_bg_indices = sorted_bg_df.loc[sorted_bg_df['median_dmf'] > 0].index
                neg_bg_indices = sorted_bg_df.loc[sorted_bg_df['median_dmf'] < 0].index
            else:
                neg_bg_indices = sorted_bg_df.index[-num_top_muts*100:]
                pos_bg_indices = sorted_bg_df.index[:num_top_muts*100] 
            # get the the mut event identifiers for the top/bottom bg events
            biggest_bg_neg_effect = sorted_bg_df.loc[neg_bg_indices, 'mut_event'].values
            biggest_bg_pos_effect = sorted_bg_df.loc[pos_bg_indices, 'mut_event'].values
            # now in all_metrics (the site by site df), select the cpgs that correspond to these events
            biggest_neg_to_plot = all_metrics_df.loc[
                all_metrics_df['mutated_sample'] == True
                ].query('mut_event in @biggest_neg_effect or mut_event in @biggest_bg_neg_effect')
            biggest_pos_to_plot = all_metrics_df.loc[
                all_metrics_df['mutated_sample'] == True
                ].query('mut_event in @biggest_pos_effect or mut_event in @biggest_bg_pos_effect')
        else:
            # in all_metrics, fg events with largest 25% of values and their corresponding bg events
            biggest_neg_to_plot = all_metrics_df.loc[
                all_metrics_df['mutated_sample'] == True
                ].query('mut_event in @biggest_neg_effect or index_event in @biggest_neg_effect')
            biggest_pos_to_plot = all_metrics_df.loc[
                all_metrics_df['mutated_sample'] == True
                ].query('mut_event in @biggest_pos_effect or index_event in @biggest_pos_effect')
            
        print("Number of unique mutation events in biggest_neg_to_plot")
        print(biggest_neg_to_plot.groupby("is_background")['mut_event'].nunique())
        
        def custom_agg1(column):
            return column.quantile(0.4)
        def custom_agg2(column):
            return column.median()
        def custom_agg3(column):
            return column.quantile(0.6)
        
        if corr_vs_dist:
            def get_genomic_dist_from_corr(corr_metrics_df, illumina_cpg_locs_df):
                mut_corr_all_metrics_df = corr_metrics_df.copy()
                #mut_corr_all_metrics_df = corr_all_metrics_df.query("is_background == False").copy()
                mut_corr_all_metrics_df.loc[:, 'mut_start'] = mut_corr_all_metrics_df['mut_event'].apply(
                    lambda x: int(x.split(':')[-1])
                    )
                mut_corr_all_metrics_df = illumina_cpg_locs_df.merge(
                    mut_corr_all_metrics_df, left_on='#id', right_on='measured_site', how = 'right'
                    )
                mut_corr_all_metrics_df.rename(columns={'start': 'measured_start'}, inplace=True)
                mut_corr_all_metrics_df['genomic_dist'] = mut_corr_all_metrics_df['mut_start'].astype(int) - mut_corr_all_metrics_df['measured_start'].astype(int)
                mut_corr_all_metrics_df['abs_genomic_dist'] = mut_corr_all_metrics_df['genomic_dist'].abs()
                return mut_corr_all_metrics_df
            # convert the corr distance to genomic distance and get rank order distance
            biggest_neg_to_plot_fg = get_genomic_dist_from_corr(
                biggest_neg_to_plot.query("is_background == False"), illumina_cpg_locs_df
                ).drop_duplicates(['measured_site', 'mut_event'])
            biggest_neg_to_plot_fg = biggest_neg_to_plot_fg.groupby('mut_event').apply(
                lambda x: x.sort_values('abs_genomic_dist', ascending=True).reset_index(drop=True)
                ).reset_index(names=['mut_event2', 'rank_order_dist'])
            biggest_neg_to_plot_fg.rename(columns={corr_dist: 'correlation_distance'}, inplace=True)
            # same for pos
            biggest_pos_to_plot_fg = get_genomic_dist_from_corr(
                biggest_pos_to_plot.query("is_background == False"), illumina_cpg_locs_df
                ).drop_duplicates(['measured_site', 'mut_event'])
            biggest_pos_to_plot_fg = biggest_pos_to_plot_fg.groupby('mut_event').apply(
                lambda x: x.sort_values('abs_genomic_dist', ascending=True).reset_index(drop=True)
                ).reset_index(names=['mut_event2', 'rank_order_dist'])
            biggest_pos_to_plot_fg.rename(columns={corr_dist: 'correlation_distance'}, inplace=True)
            # sort the DataFrames by rank_order_corr and drop duplicates
            biggest_neg_to_plot_sorted_corr = biggest_neg_to_plot_fg.query(
                "is_background == False"
                ).sort_values(by='correlation_distance', ascending=False).drop_duplicates()
            biggest_pos_to_plot_sorted_corr = biggest_pos_to_plot_fg.query(
                "is_background == False"
                ).sort_values(by='correlation_distance', ascending=False).drop_duplicates()
            # within each rolling window, calculate the median, 40th and 60th percentile
            biggest_neg_to_plot_smoothed_corr = (
                biggest_neg_to_plot_sorted_corr.set_index('correlation_distance')
                .rolling(smoothing_window_size_corr, center=True, min_periods=0, closed = 'left')
                ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                #.agg(['mean',  'std', 'count']).reset_index()
                ).iloc[::num_top_muts, :]
            # keep only every 1000th row
            """biggest_pos_to_plot_smoothed_corr = (
                biggest_pos_to_plot_sorted_corr.set_index('correlation_distance')
                .rolling(smoothing_window_size_corr, center=True, min_periods=0, closed = 'left')
                ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                #.agg(['mean',  'std', 'count']).reset_index()
                ).iloc[::num_top_muts, :]"""
            # then do the same for distance measure
            dist_measure = 'abs_genomic_dist' if not rank_order_dist else 'rank_order_dist'
            biggest_neg_to_plot_sorted_dist = biggest_neg_to_plot_fg.query(
                "is_background == False"
                ).sort_values(by=dist_measure, ascending=False).drop_duplicates()
            """biggest_pos_to_plot_sorted_dist = biggest_pos_to_plot_fg.query(
                "is_background == False"
                ).sort_values(by=dist_measure, ascending=False).drop_duplicates()"""
            biggest_neg_to_plot_smoothed_dist = (
                biggest_neg_to_plot_sorted_dist.set_index(dist_measure)
                .rolling(smoothing_window_size_dist, center=True, min_periods=0, closed = 'left')
                ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                #.agg(['mean',  'std', 'count']).reset_index()
                ).iloc[::num_top_muts, :]
            """biggest_pos_to_plot_smoothed_dist = (
                biggest_pos_to_plot_sorted_dist.set_index(dist_measure)
                .rolling(smoothing_window_size_dist, center=True, min_periods=0, closed = 'left')
                ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                #.agg(['mean',  'std', 'count']).reset_index()
                ).iloc[::num_top_muts, :]"""
            fig, axes = plt.subplots(figsize=(10, 6), dpi = 100)
            axes_top = axes.twiny()
            #fig2, axes2 = plt.subplots(figsize=(8, 6), dpi = 100, sharex='col', sharey='row')
            # select 4 colors from a color map of maroon
            sns.set_context("paper")
            """axes = axes.flatten()
            # show y and x tick labels on all subplots
            for i in range(len(axes)):
                axes[i].xaxis.set_tick_params(labelbottom=True)
                axes[i].yaxis.set_tick_params(labelleft=True)"""
            
            # positive corr
            """axes.plot(
                biggest_pos_to_plot_smoothed_corr['correlation_distance'],
                biggest_pos_to_plot_smoothed_corr['median'],
                color=mut_colors[0], label = 'Methylation-coherence Top 25%',
                linewidth = 1.5
                )
            
            axes.fill_between(
                biggest_pos_to_plot_smoothed_corr['correlation_distance'],
                biggest_pos_to_plot_smoothed_corr['5th'],
                biggest_pos_to_plot_smoothed_corr['95th'],
                color=mut_colors[0], alpha=0.2, rasterized=True
                )"""
            
            
            # negative corr
            axes.plot(
                biggest_neg_to_plot_smoothed_corr['correlation_distance'],
                biggest_neg_to_plot_smoothed_corr['median'],
                color='maroon', label = 'Methylation-coherence Bottom 25%',
                linewidth = 1.5
                )
            """axes.fill_between(
                biggest_neg_to_plot_smoothed_corr['correlation_distance'],
                biggest_neg_to_plot_smoothed_corr['5th'],
                biggest_neg_to_plot_smoothed_corr['95th'],
                color=mut_colors[2], alpha=0.2, rasterized=True
                )"""
            axes.set_xlabel('Methylation-coherence distance (rank order)')
            axes.set_ylabel('$\Delta$MF')
            #axes[2].set_xlabel('Correlation distance (rank order)')
            #axes[2].set_ylabel('$\Delta$MF')            
            # positive dist  
            """axes_top.plot(
                biggest_pos_to_plot_smoothed_dist[dist_measure] / 1000,
                biggest_pos_to_plot_smoothed_dist['median'],
                color=mut_colors[0], label = 'Genomic distance Top 25%',
                linewidth = 1.5, linestyle = 'dashed'
                )
            axes_top.fill_between(
                biggest_pos_to_plot_smoothed_dist[dist_measure] / 1000,
                biggest_pos_to_plot_smoothed_dist['5th'],
                biggest_pos_to_plot_smoothed_dist['95th'],
                color=mut_colors[0], alpha=0.2, rasterized=True
                )"""
            #axes[1].set_ylabel('')
            # negative dist
            axes_top.plot(
                biggest_neg_to_plot_smoothed_dist[dist_measure] / 1000,
                biggest_neg_to_plot_smoothed_dist['median'],
                color='maroon', label = 'Genomic distance Bottom 25%',
                linewidth = 1.5, linestyle = 'dashed'
                )
            axes_top.set_xlabel('Genomic distance (kb)')
            
            """axes_top.fill_between(
                biggest_neg_to_plot_smoothed_dist[dist_measure] / 1000,
                biggest_neg_to_plot_smoothed_dist['5th'],
                biggest_neg_to_plot_smoothed_dist['95th'],
                color=mut_colors[2], alpha=0.2, rasterized=True
                )"""
            # show legend
            axes.legend(loc='upper right', title='Mutation event')
            axes_top.legend(loc='upper left', title='Mutation event')
            #axes[3].set_xlabel('Genomic distance (rank order)')
            #axes[3].set_ylabel('')
            if plot_bg:
                biggest_neg_bg_to_plot = get_genomic_dist_from_corr(
                    biggest_neg_to_plot.query("is_background == True").head(len(biggest_neg_to_plot_sorted_dist * 5)), illumina_cpg_locs_df
                    ).drop_duplicates(['measured_site', 'mut_event'])
                biggest_neg_bg_to_plot = biggest_neg_bg_to_plot.groupby('mut_event').apply(
                    lambda x: x.sort_values('abs_genomic_dist', ascending=True).reset_index(drop=True)
                    ).reset_index(names=['mut_event2', 'rank_order_dist'])
                biggest_neg_bg_to_plot.rename(columns={corr_dist: 'correlation_distance'}, inplace=True)
                # pos
                """biggest_pos_bg_to_plot = get_genomic_dist_from_corr(
                    biggest_pos_to_plot.query("is_background == True").head(len(biggest_neg_to_plot_sorted_dist)), illumina_cpg_locs_df
                    ).drop_duplicates(['measured_site', 'mut_event'])
                biggest_pos_bg_to_plot = biggest_pos_bg_to_plot.groupby('mut_event').apply(
                    lambda x: x.sort_values('abs_genomic_dist', ascending=True).reset_index(drop=True)
                    ).reset_index(names=['mut_event2', 'rank_order_dist'])
                biggest_pos_bg_to_plot.rename(columns={corr_dist: 'correlation_distance'}, inplace=True)"""
                # correlation
                biggest_neg_bg_to_plot_sorted_corr = biggest_neg_bg_to_plot.query(
                    "is_background == True"
                    ).sort_values(by='correlation_distance', ascending=False).drop_duplicates()
                """biggest_pos_bg_to_plot_sorted_corr = biggest_pos_bg_to_plot.query(
                    "is_background == True"
                    ).sort_values(by='correlation_distance', ascending=False).drop_duplicates()"""
                bg_biggest_neg_to_plot_smoothed_corr = (
                    biggest_neg_bg_to_plot_sorted_corr.set_index('correlation_distance')
                    .rolling(smoothing_window_size_corr, center=True, min_periods=0, closed = 'left')
                    ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                    .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                    ).iloc[::num_top_muts, :]
                """bg_biggest_pos_to_plot_smoothed_corr = (
                    biggest_pos_bg_to_plot_sorted_corr.set_index('correlation_distance')
                    .rolling(smoothing_window_size_corr, center=True, min_periods=0, closed = 'left')
                    ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                    .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                    ).iloc[::num_top_muts, :]"""
                # distance
                biggest_neg_bg_to_plot_sorted_dist = biggest_neg_bg_to_plot.query(
                    "is_background == True"
                    ).sort_values(by=dist_measure, ascending=False).drop_duplicates()
                """biggest_pos_bg_to_plot_sorted_dist = biggest_pos_bg_to_plot.query(
                    "is_background == True"
                    ).sort_values(by=dist_measure, ascending=False).drop_duplicates()"""
                bg_biggest_neg_to_plot_smoothed_dist = (
                    biggest_neg_bg_to_plot_sorted_dist.set_index(dist_measure)
                    .rolling(smoothing_window_size_dist, center=True, min_periods=0, closed = 'left')
                    ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                    .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                    )
                """bg_biggest_pos_to_plot_smoothed_dist = (
                    biggest_pos_bg_to_plot_sorted_dist.set_index(dist_measure)
                    .rolling(smoothing_window_size_dist, center=True, min_periods=0,  closed = 'left')
                    ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                    .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                    )"""
                # positive corr
                """axes.plot(
                    bg_biggest_pos_to_plot_smoothed_corr['correlation_distance'],
                    bg_biggest_pos_to_plot_smoothed_corr['median'],
                    color=bg_colors[0], label = '(BG) Methylation-coherence Top 25%',
                    linewidth = 1.5
                    )
                axes.fill_between(
                    bg_biggest_pos_to_plot_smoothed_corr['correlation_distance'],
                    bg_biggest_pos_to_plot_smoothed_corr['5th'],
                    bg_biggest_pos_to_plot_smoothed_corr['95th'],
                    color=bg_colors[0], alpha=0.2, rasterized=True
                    )"""
                # negative corr
                axes.plot(
                    bg_biggest_neg_to_plot_smoothed_corr['correlation_distance'],
                    bg_biggest_neg_to_plot_smoothed_corr['median'],
                    color='steelblue', label = '(BG) Methylation-coherence Bottom 25%',
                    linewidth = 1.5
                    )
                """axes.fill_between(
                    bg_biggest_neg_to_plot_smoothed_corr['correlation_distance'],
                    bg_biggest_neg_to_plot_smoothed_corr['5th'],
                    bg_biggest_neg_to_plot_smoothed_corr['95th'],
                    color=bg_colors[2], alpha=0.2, rasterized=True
                    )"""
                # ositive dist  
                """axes_top.plot(
                    bg_biggest_pos_to_plot_smoothed_dist[dist_measure] /1000,
                    bg_biggest_pos_to_plot_smoothed_dist['median'],
                    color=bg_colors[0], label = '(BG) Genomic distance Top 25%',
                    linewidth = 1.5, linestyle = 'dashed'
                    )
                axes_top.fill_between(
                    bg_biggest_pos_to_plot_smoothed_dist[dist_measure] / 1000,
                    bg_biggest_pos_to_plot_smoothed_dist['5th'],
                    bg_biggest_pos_to_plot_smoothed_dist['95th'],
                    color=bg_colors[0], alpha=0.2, rasterized=True
                    )"""
                # negative dist
                axes_top.plot(
                    bg_biggest_neg_to_plot_smoothed_dist[dist_measure] / 1000,
                    bg_biggest_neg_to_plot_smoothed_dist['median'],
                    color='steelblue', label = '(BG) Genomic distance Bottom 25%',
                    linewidth = 1.5, linestyle = 'dashed'
                    )
                """axes_top.fill_between(
                    bg_biggest_neg_to_plot_smoothed_dist[dist_measure] / 1000,
                    bg_biggest_neg_to_plot_smoothed_dist['5th'],
                    bg_biggest_neg_to_plot_smoothed_dist['95th'],
                    color=bg_colors[2], alpha=0.2, rasterized=True
                    )"""
                # show legend
                axes.legend(loc='upper right', title='Mutation event')
                axes_top.legend(loc='upper left', title='Mutation event')
                fig.savefig(fname=out_fn, format='svg', dpi = 300, bbox_inches='tight') 
                return biggest_neg_to_plot_smoothed_corr, biggest_neg_to_plot_smoothed_dist, bg_biggest_neg_to_plot_smoothed_corr, bg_biggest_neg_to_plot_smoothed_dist
            
            fig.savefig(fname=out_fn, format='svg', dpi = 300, bbox_inches='tight') 
            return  biggest_neg_to_plot_smoothed_corr, biggest_neg_to_plot_smoothed_dist
        else:
            # sort the DataFrames by rank_order_corr and drop duplicates
            biggest_neg_to_plot_sorted = biggest_neg_to_plot.query(
                "is_background == False"
                ).sort_values(by='measured_site_dist', ascending=False).drop_duplicates()
            biggest_pos_to_plot_sorted = biggest_pos_to_plot.query(
                "is_background == False"
                ).sort_values(by='measured_site_dist', ascending=False).drop_duplicates()
            # within each rolling window, calculate the median, 40th and 60th percentile
            biggest_neg_to_plot_smoothed = (
                biggest_neg_to_plot_sorted.set_index('measured_site_dist')
                .rolling(smoothing_window_size_dist, center=True, min_periods=0, closed = 'left')
                ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                #.agg(['mean',  'std', 'count']).reset_index()
                )
            biggest_pos_to_plot_smoothed = (
                biggest_pos_to_plot_sorted.set_index('measured_site_dist')
                .rolling(smoothing_window_size_dist, center=True, min_periods=0, closed = 'left')
                ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                #.agg(['mean',  'std', 'count']).reset_index()
                )
            # round measured_site_dist to nearest 100
            """biggest_pos_to_plot_sorted['measured_site_dist'] = (
                biggest_pos_to_plot_sorted['measured_site_dist'] / 1000
                ).apply(np.floor) * 1000
            biggest_pos_to_plot_smoothed = biggest_pos_to_plot_sorted.groupby('measured_site_dist')['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index().rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
            biggest_neg_to_plot_sorted['measured_site_dist'] = (
                biggest_neg_to_plot_sorted['measured_site_dist'] / 1000
                ).apply(np.floor) * 1000
            biggest_neg_to_plot_smoothed = biggest_neg_to_plot_sorted.groupby('measured_site_dist')['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index().rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})"""
            #return biggest_neg_to_plot_sorted, biggest_neg_to_plot_smoothed
            # plot as lineplots
            # 1kb, 1k smoothing all pos/neg
            fig, axes = plt.subplots(2, 1, figsize=(6, 5), dpi = 100, sharex=False)
            # increase text size
            sns.set_context("paper")
            axes[0].plot(
                biggest_pos_to_plot_smoothed['measured_site_dist']/1000,
                biggest_pos_to_plot_smoothed['median'],
                color='maroon'
                )
            axes[0].fill_between(
                biggest_pos_to_plot_smoothed['measured_site_dist']/1000,
                biggest_pos_to_plot_smoothed['5th'],
                biggest_pos_to_plot_smoothed['95th'],
                color='maroon', alpha=0.2, rasterized=True
                )
            axes[0].set_xlabel('')
            axes[0].set_ylabel('$\Delta$MF')
            
            axes[1].plot(
                biggest_neg_to_plot_smoothed['measured_site_dist']/1000,
                biggest_neg_to_plot_smoothed['median'],
                color='maroon'
                )
            axes[1].fill_between(
                biggest_neg_to_plot_smoothed['measured_site_dist']/1000,
                biggest_neg_to_plot_smoothed['5th'],
                biggest_neg_to_plot_smoothed['95th'],
                color='maroon', alpha=0.2, rasterized=True
                )
            axes[1].set_xlabel('Distance from mutated site (kb)')
            axes[1].set_ylabel('$\Delta$MF')
            
            axes[0].set_xlim(-40, 40)
            axes[1].set_xlim(-40, 40)
            
            # plot background too
            if plot_bg:
                # and get background sorted too
                biggest_neg_to_plot.reset_index(drop= True, inplace = True)
                bg_biggest_neg_to_plot_sorted = biggest_neg_to_plot.query(
                    "is_background == True"
                    ).iloc[::100].sort_values(by='measured_site_dist', ascending=False).drop_duplicates()
                biggest_pos_to_plot.reset_index(drop= True, inplace = True)
                bg_biggest_pos_to_plot_sorted = biggest_pos_to_plot.query(
                    "is_background == True"
                    ).iloc[::100].sort_values(by='measured_site_dist', ascending=False).drop_duplicates()
                bg_biggest_neg_to_plot_smoothed = (
                    bg_biggest_neg_to_plot_sorted.set_index('measured_site_dist')
                    .rolling(4000, center=True, min_periods=0, closed = 'left')
                    ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                    .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                    )
                bg_biggest_pos_to_plot_smoothed = (
                    bg_biggest_pos_to_plot_sorted.set_index('measured_site_dist')
                    .rolling(4000, center=True, min_periods=0, closed = 'left')
                    ['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index()
                    .rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                    )
                
                
                """# round measured_site_dist to nearest 100
                bg_biggest_pos_to_plot_sorted['measured_site_dist'] = (
                    bg_biggest_pos_to_plot_sorted['measured_site_dist'] / 1000
                    ).apply(np.floor) * 1000
                bg_biggest_pos_to_plot_smoothed = bg_biggest_pos_to_plot_sorted.groupby('measured_site_dist')['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index().rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})
                bg_biggest_neg_to_plot_sorted['measured_site_dist'] = (
                    bg_biggest_neg_to_plot_sorted['measured_site_dist'] / 1000
                    ).apply(np.floor) * 1000
                bg_biggest_neg_to_plot_smoothed = bg_biggest_neg_to_plot_sorted.groupby('measured_site_dist')['delta_mf_median'].agg([custom_agg1, custom_agg2, custom_agg3]).reset_index().rename(columns={'custom_agg1': '5th', 'custom_agg2': 'median', 'custom_agg3': '95th'})"""
                
                axes[0].plot(
                    bg_biggest_pos_to_plot_smoothed['measured_site_dist']/1000,
                    bg_biggest_pos_to_plot_smoothed['median'],
                    color='steelblue'
                    )
                axes[0].fill_between(
                    bg_biggest_pos_to_plot_smoothed['measured_site_dist']/1000,
                    bg_biggest_pos_to_plot_smoothed['5th'],
                    bg_biggest_pos_to_plot_smoothed['95th'],
                    color='steelblue', alpha=0.2, rasterized=True
                    )
                axes[1].plot(
                    bg_biggest_neg_to_plot_smoothed['measured_site_dist']/1000,
                    bg_biggest_neg_to_plot_smoothed['median'],
                    color='steelblue'
                    )
                axes[1].fill_between(
                    bg_biggest_neg_to_plot_smoothed['measured_site_dist']/1000,
                    bg_biggest_neg_to_plot_smoothed['5th'],
                    bg_biggest_neg_to_plot_smoothed['95th'],
                    color='steelblue', alpha=0.2, rasterized=True
                    )
            
                return biggest_neg_to_plot_sorted, biggest_pos_to_plot_sorted, bg_biggest_neg_to_plot_sorted, bg_biggest_pos_to_plot_sorted
        plt.savefig(fname=out_fn, format='svg', dpi = 300, bbox_inches='tight')
        return biggest_neg_to_plot_sorted, biggest_pos_to_plot_sorted

        
    def plot_distance_of_effect_boxplot(
        self, 
        mut_events,
        num_bins,
        min_dist,
        max_dist,
        method = 'dist',
        log_scale = False
        ):
        sns.set_context('paper')
        # make bigger figure
        plt.figure(figsize=(6,4), dpi=100)
        # subset
        mut_events = mut_events.loc[mut_events['mutated_sample'] == True]
        # change to kb
        if method == 'dist':
            min_dist = int(min_dist / 1000)
            max_dist = int(max_dist / 1000)
        
        if log_scale:
            # Generate log-scaled bin edges
            bin_edges = np.logspace(np.log10(min_dist+1), np.log10(max_dist), num_bins + 1)
            # Assign data to bins
            mut_events['dist_bin'] = pd.cut(
                mut_events['measured_site_dist'],
                bins=bin_edges,
                labels=[
                    '[' + str(round(bin_edges[i], 2)) + ',' + str(round(bin_edges[i + 1], 2)) + ')'
                    for i in range(num_bins)
                ],
                include_lowest=True
            )
        else:
            # do binning
            mut_events['dist_bin'] = pd.cut(
                mut_events['measured_site_dist'],
                bins = num_bins,
                labels = [ 
                        '[' + str(x) + ',' + str(x + int((max_dist - min_dist) / num_bins)) + ')' 
                            for x in range(min_dist, max_dist, int((max_dist - min_dist) / num_bins))
                            ]
                )
        # rename
        mut_events = mut_events.rename(columns={'is_background': 'Locus'})
        # map Mutaiton event column values
        mut_events['Locus'] = mut_events['Locus'].map(
            {True: 'Random', False: 'Mutated'}
            )
        sns.boxplot(
            data= mut_events,
            x = 'dist_bin', y = 'delta_mf_median', 
            hue = 'Locus', showfliers = False, 
            palette=['steelblue', 'maroon']
        )
        # do not plot legend
        plt.legend([], [], frameon=False)
        # angle x labels
        plt.xticks(rotation=30)
        if method == 'dist':
            plt.xlabel('Distance from mutation (kb)')
        else:
            plt.xlabel('Correlation distance from mutated CpG')

        plt.ylabel(r'$\Delta$MF')
        
    def get_corr_site_distances(
        self, 
        illumina_cpg_locs_df, 
        comparison_sites,
        mut_cpg_name
        ):
        # get the distances between the mutated site and the comparison sites
        comparison_sites_starts = illumina_cpg_locs_df.loc[
            illumina_cpg_locs_df['#id'].isin(comparison_sites), ['#id', 'start']
            ]
        comparison_sites_starts.set_index('#id', inplace=True)
        comparison_sites_starts = comparison_sites_starts.reindex(comparison_sites)
        mut_cpg_start = illumina_cpg_locs_df.loc[
            illumina_cpg_locs_df['#id'] == mut_cpg_name, 'start'
            ].values[0]
        comparison_site_distances = comparison_sites_starts['start'] - mut_cpg_start
        return comparison_site_distances.values.tolist()

    def plot_heatmap_dist(
        self, 
        mut_event: str, 
        comparison_sites_df: pd.DataFrame,
        all_methyl_age_df_t: pd.DataFrame,
        max_abs_distance: int = 1e9, 
        max_matched_samples: int = 1e9,
        illumina_cpg_locs_df: pd.DataFrame = None,
        method: str = 'dist',
        rolling_window_size: int = 1,
        ) -> None:
        sns.set_theme(style='white', font_scale=1.3, rc={'xtick.bottom': True, 'ytick.left': True})
        
        # get mutatated sample name and comparison sites
        this_mut_event = comparison_sites_df.loc[comparison_sites_df['mut_event'] == mut_event]
        if len(this_mut_event) == 0:
            print("mut event not present")
            sys.exit(1)
        mut_sample_name = this_mut_event['case_submitter_id'].values[0]
        matched_samples = this_mut_event['matched_samples'].to_list()[0]
        comparison_sites = this_mut_event['comparison_sites'].values[0]
        distances = this_mut_event['comparison_dists'].values[0]
        mut_cpg_name = this_mut_event['#id'].values[0]
        mutated_site = 'chr' + mut_event.split('_')[1]
        # check if first entry of comparison sites is a byte string
        if isinstance(comparison_sites[0], bytes):
            # convert if so
            comparison_sites = [x.decode() for x in comparison_sites]
    
        # simultaneously sort the comparison sites and distances, by distance
        comparison_site_and_distances = pd.DataFrame(
            {'comparison_sites': comparison_sites, 'distances': distances}
            )
        comparison_site_and_distances.sort_values(by='distances', inplace=True)
        # drop rows with distance > max_abs_distance
        comparison_site_and_distances = comparison_site_and_distances.loc[
            abs(comparison_site_and_distances.distances) <= max_abs_distance
            ]
        if method == 'corr':
            actual_distances = self.get_corr_site_distances(
                illumina_cpg_locs_df, comparison_site_and_distances['comparison_sites'], mut_cpg_name
                )
            comparison_site_and_distances['actual_distances'] = actual_distances
        
        comparison_site_and_distances.reset_index(inplace=True, drop=True)
        #comparison_site_and_distances = comparison_site_and_distances.iloc[10:-34, :]
        distances = comparison_site_and_distances.distances.to_list()
        comparison_sites = comparison_site_and_distances.comparison_sites.to_list()
        
        # find where distances switches from negative to positive
        mut_pos = -1
        for i, dist in enumerate(distances):
            if dist > 0:
                mut_pos = i
                print(distances[i-1], distances[i], mut_pos)
                break
        # if all distances are negative, put the mutated site at the end
        if mut_pos == -1:
            mut_pos = len(distances)
            print(distances[i-1], distances[i], mut_pos)

        # put the mutated sample in middle
        samples_to_plot = np.concatenate(
            (utils.half(matched_samples[:int(max_matched_samples)], 'first'), 
             [mut_sample_name], 
             utils.half(matched_samples[:int(max_matched_samples)], 'second'))
            )
        
        """
        For TCGA dist example
        comparison_sites = comparison_sites[10:-37]
        print(distances[10])
        print(distances[-37])
        """
        # get mf of the comparison sites
        all_samples_comp_sites = all_methyl_age_df_t.loc[samples_to_plot, comparison_sites]
        if rolling_window_size > 1:
            # combine every adjacent rolling_window_size sites
            all_samples_comp_sites = all_samples_comp_sites.rolling(
                rolling_window_size, axis=1
                ).mean()#.iloc[:, ::rolling_window_size]
        # plot DMF
        _, axes = plt.subplots(figsize=(9,6), dpi=100)
        all_samples_comp_sites_dmf = all_samples_comp_sites.subtract(
            all_samples_comp_sites.median(axis=0), axis=1
            )
        ax = sns.heatmap(
            data = all_samples_comp_sites_dmf, annot=False, xticklabels=False, yticklabels=True, 
            cmap="icefire", vmin=-1, vmax=1, center=0,
            cbar_kws={'label': r'$\Delta$MF'}, ax=axes
            )
        # label axes
        axes.set_xlabel('Comparison sites')
        axes.set_ylabel('Samples')
        # add a y tick for the mutated sample, make tick label red, and rotate 90 degrees
        """axes.set_yticks(np.arange(.5, len(samples_to_plot)+.5, 1))
        axes.set_yticks([int(len(utils.half(matched_samples[:int(max_matched_samples)], 'first')))+.5])
        axes.set_yticklabels([mut_sample_name], color='red', rotation=90, ha='center', rotation_mode='anchor')"""
        # make the y tick labels smaller
        axes.tick_params(axis='y', which='major', labelsize=8)
        # make the mutated sample label red
        axes.yaxis.get_majorticklabels()[int(len(utils.half(matched_samples[:int(max_matched_samples)], 'first')))].set_color('red')
        axes.tick_params(axis='y', which='major', pad=5)
        # add a x tick for the mutated site, make tick label red, and rotate 90 degrees
        if method == 'corr':
            label_locations = [mut_pos - .5]
        else:
            label_locations = [mut_pos]
        
        label_locations = [0, mut_pos, len(comparison_sites)-1]
        mutated_site_chr = mutated_site.split(':')[0]
        mutated_site_start = int(mutated_site.split(':')[1])
        labels = [f'{mutated_site_chr}:'+ format(mutated_site_start + distances[0], ','), mutated_site_chr + ':' + format(mutated_site_start, ',') , f'{mutated_site_chr}:'+ format(mutated_site_start + distances[-1], ',')]
        axes.set_xticks(label_locations)
        axes.set_xticklabels(labels, color='black', ha='center', rotation_mode='anchor')
        #axes.tick_params(axis='x', which='major', labelsize=)
        # make the mutated site label red
        axes.xaxis.get_majorticklabels()[1].set_color('red')
        
        #plt.savefig('/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure2/heatmap_DMF.svg', format='svg', dpi = 300, bbox_inches='tight')
        plt.savefig('/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure4/figure4C_heatmap_DMF.svg', format='svg', dpi = 300, bbox_inches='tight')
        
        from matplotlib.colors import LinearSegmentedColormap

        cmap_segments = [
            (0.0, "white"),
            (0.2, "aliceblue"),
            (0.3, "lightskyblue"),
            (0.4, "skyblue"),
            (0.5, "deepskyblue"),
            (0.8, "dodgerblue"),
            (0.9, "mediumblue"),
            (1.0, "midnightblue"),
        ]
        # Create the colormap
        cmap = LinearSegmentedColormap.from_list("white_blue_very_dark_blue", cmap_segments, N =256)

        # plot MF
        _, axes = plt.subplots(figsize=(9,6), dpi=100)
        ax = sns.heatmap(
            data = all_samples_comp_sites, annot=False, xticklabels=False, yticklabels=True, 
            cmap='Blues', vmin=0, vmax=1, center=0.5,
            cbar_kws={'label': r'Methylation fraction'}, ax=axes
            )
        # label axes
        axes.set_xlabel('Comparison sites')
        axes.set_ylabel('Samples')
        # add a y tick for the mutated sample, make tick label red, and rotate 90 degrees
        """axes.set_yticks(np.arange(.5, len(samples_to_plot)+.5, 1))
        axes.set_yticks([int(len(utils.half(matched_samples[:int(max_matched_samples)], 'first')))+.5])
        axes.set_yticklabels([mut_sample_name], color='red', rotation=90, ha='center', rotation_mode='anchor')"""
        axes.tick_params(axis='y', which='major', labelsize=8)
        # make the mutated sample label red
        axes.yaxis.get_majorticklabels()[int(len(utils.half(matched_samples[:int(max_matched_samples)], 'first')))].set_color('red')
        axes.tick_params(axis='y', which='major', pad=5)
        # add a x tick for the mutated site, make tick label red, and rotate 90 degrees
        label_locations = [0, mut_pos, len(comparison_sites)-1]
        mutated_site_chr = mutated_site.split(':')[0]
        mutated_site_start = int(mutated_site.split(':')[1])
        labels = [f'{mutated_site_chr}:'+ format(mutated_site_start + distances[0], ','), mutated_site_chr + ':' + format(mutated_site_start, ',') , f'{mutated_site_chr}:'+ format(mutated_site_start + distances[-1], ',')]
        axes.set_xticks(label_locations)
        axes.set_xticklabels(labels, color='black', ha='center', rotation_mode='anchor')
        #axes.tick_params(axis='x', which='major', labelsize=)
        # make the mutated site label red
        axes.xaxis.get_majorticklabels()[1].set_color('red')
        
        # plot distances as a lineplot
        if method == 'corr':
            distances = comparison_site_and_distances['actual_distances'].to_list()
        #plt.savefig('/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure2/heatmap_methylFrac.svg', format='svg', dpi = 300, bbox_inches='tight')
        plt.savefig('/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure4/Figure4C_heatmap_methylFrac.svg', format='svg', dpi = 300, bbox_inches='tight')
        
        _, axes = plt.subplots(figsize=(8, 2), dpi=100)
        ax = sns.scatterplot(
            y=[np.abs(x) for x in distances], 
            x=np.arange(1, len(distances)+1, 1), 
            color='black', ax=axes
            )
        ax.set_yscale('log')
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=7))
        #ax.legend(loc='upper right')
        # plot correlation distance on the same plot with y axis on the right
        if method == 'corr':
            ax2 = ax.twinx()
            ax2 = sns.scatterplot(
                y=[x for x in range(1, len(comparison_site_and_distances['distances'].to_list()) +1, 1)], 
                x=np.arange(1, len(distances)+1, 1), 
                color='grey', ax=ax2, alpha = 0.6
                )
            #ax2.legend(loc='upper left')
            # added these three lines
            ax2.set_ylabel('Correlation\ndistance')
            
            
        # plot y = 0 line on axes
        #plt.axhline(y=0, color='black', linestyle='--')
        # plot a vertical line at mut_pos
        #plt.axvline(x=mut_pos - .5, color='red', linestyle='--')
        # add y axis label
        axes.set_ylabel('Distance from\n mutation (bp)')
        # do not plot y ticks or labels
        axes.set_xticks([])
        axes.set_xticklabels([])
        axes.set_xlabel('Comparison sites')
        #plt.savefig('/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure2/dist_of_heatmap_sites.svg', format='svg', dpi = 300, bbox_inches='tight')
        plt.savefig('/cellar/users/zkoch/methylation_and_mutation/output_dirs/final_figures/figure4/Figure4C_dist_of_heatmap_sites.svg', format='svg', dpi = 300, bbox_inches='tight')
        
        return all_samples_comp_sites, all_samples_comp_sites_dmf, comparison_site_and_distances
        
    def plot_heatmap(
        self, 
        mut_event: str, 
        comparison_sites_df: pd.DataFrame,
        all_methyl_age_df_t: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame,
        linkage_method: str,
        max_abs_distance: int = 1e9, 
        max_sites: int = 1e9,
        ) -> None:
        """
        Plot a heatmap of the mutated sample and matched sample MFs at the comparison sites
        @ mut_event: the mutation event to plot
        @ comparison_sites_df: a dataframe with the comparison sites for each mutation event
        @ linkage_method: the linkage method used to cluster the comparison sites
        @ max_abs_distance: the distance limit used to cluster the comparison sites
        """
        # get mutatated sample name and comparison sites
        this_mut_event = comparison_sites_df.loc[comparison_sites_df['mut_event'] == mut_event]
        print(this_mut_event)
        if len(this_mut_event) == 0:
            print("mut event not present")
            sys.exit(1)
        mut_sample_name = this_mut_event['case_submitter_id'].values[0]
        """try:
            comparison_sites = [i.decode('ASCII') for i in this_mut_event['comparison_sites'].values[0]]
        except:"""
        comparison_sites = this_mut_event['comparison_sites'].values[0]
        print(comparison_sites)
        mut_cpg_name = this_mut_event['#id'].values[0]
        
        # if linkage method is corr, we need to get their real distances
        if linkage_method == 'corr':
            # get the distances between the mutated site and the comparison sites
            comparison_sites_starts = illumina_cpg_locs_df.loc[
                illumina_cpg_locs_df['#id'].isin(comparison_sites), ['#id', 'start']
                ]
            comparison_sites_starts.set_index('#id', inplace=True)
            comparison_sites_starts = comparison_sites_starts.reindex(comparison_sites)
            mut_cpg_start = illumina_cpg_locs_df.loc[
                illumina_cpg_locs_df['#id'] == mut_cpg_name, 'start'
                ].values[0]
            comparison_site_distances = comparison_sites_starts['start'] - mut_cpg_start
            comparison_site_distances.sort_values(ascending = True, inplace=True)
            print(comparison_site_distances)
            comparison_site_distances = comparison_site_distances[
                (comparison_site_distances > -1*max_abs_distance) 
                & (comparison_site_distances < max_abs_distance)
                ]
            distances = comparison_site_distances.values.tolist()
            comparison_sites = comparison_site_distances.index.tolist()
        elif linkage_method == 'dist':
            distances = this_mut_event['comparison_dists'].values[0]
            # need to simultaneously sort the comparison sites and distances, by distance
            distances, comparison_sites = zip(*sorted(zip(distances, comparison_sites)))
            # convert to list
            distances = list(distances)
            comparison_sites = list(comparison_sites)
            
        # get the matched samples
        matched_samples = this_mut_event['matched_samples'].to_list()[0]
        """# and drop those with mutations nearby
        samples_to_exclude = self._detect_effect_in_other_samples(
            sites_to_test = comparison_sites, 
            mut_row = this_mut_event
            )
        before_drop = len(matched_samples)
        matched_samples = [s for s in matched_samples if s not in samples_to_exclude]
        print(f"Dropped {before_drop - len(matched_samples)} samples with mutations nearby")"""
        # get methylation fractions of the all samples at comparison sites, mut sample in middle
        samples_to_plot = np.concatenate(
            (utils.half(matched_samples, 'first'), 
             [mut_sample_name], 
             utils.half(matched_samples, 'second'))
            )
        
        
        # find the index of comparison_sites_distances which is closest to 0 
        if linkage_method == 'corr':
            magnify_mut_factor = 4
        else:
            magnify_mut_factor = 0
            mut_pos = 0
        for i in range(len(distances)):
            if distances[i] > 0:
                comparison_sites = comparison_sites[:i] \
                                    + ([mut_cpg_name] * magnify_mut_factor) \
                                    + comparison_sites[i:]
                distances = distances[:i] + [0] * magnify_mut_factor + distances[i:]
                mut_pos = i + magnify_mut_factor/2
                break
            if max(distances) < 0:
                comparison_sites.append(mut_cpg_name)
                comparison_sites = comparison_sites + [mut_cpg_name] * magnify_mut_factor
                distances = distances + [0] * magnify_mut_factor
                mut_pos = len(distances) - 1 + magnify_mut_factor/2
    
        print(samples_to_plot)
        print(comparison_sites)
        all_samples_comp_sites = all_methyl_age_df_t.loc[samples_to_plot, comparison_sites[:max_sites]]
        distances = distances[:max_sites]
        # plot MF as a heatmap
        _, axes = plt.subplots(figsize=(9,6), dpi=100)
        ax = sns.heatmap(
            data = all_samples_comp_sites, annot=False, xticklabels=False, yticklabels=False, 
            cmap="Blues", vmin=0, vmax=1, center=0.5,
            cbar_kws={'label': r'Methylation fraction'}, ax=axes
            )#, cmap="icefire", vmin=-1, vmax=1, center=0
        # label axes
        ax.set_xlabel('Comparison sites')
        ax.set_ylabel('Samples')
        # add a y tick for the mutated sample, make tick label red, and rotate 90 degrees
        ax.set_yticks(np.arange(.5, len(samples_to_plot)+.5, 1))
        ax.set_yticks([int(len(utils.half(matched_samples, 'first')))+.5])
        ax.set_yticklabels(['Mutated sample'], color='red', rotation=90, ha='center', rotation_mode='anchor')
        ax.tick_params(axis='y', which='major', pad=5)
        # slightly seperate mutated CpG
        ax.add_patch(Rectangle((mut_pos-(magnify_mut_factor/2), 0), magnify_mut_factor, len(all_samples_comp_sites), fill=False, edgecolor='white', lw=4))
        # add second around the others to make same height
        """ax.add_patch(Rectangle((100, 0), len(all_samples_comp_sites.columns), len(all_samples_comp_sites), fill=False, edgecolor='white', lw=2))"""
        # add a tick label for the mutated CpG
        tick_locs = [0, mut_pos, len(distances)]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(
            [str(int(-1*distances[0]/1000000))+'Mbp',
             'Mutated site', str(int(distances[-1]/1000000))+'Mbp'],
            ha='center', rotation_mode='anchor'
            )
        colors = ['black', 'red', 'black']
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
            
        # plot delta_mf as a heatmap
        _, axes = plt.subplots(figsize=(9,6), dpi=100)
        # median normalize columns of all_samples_comp_sites
        all_samples_comp_sites_dmf = all_samples_comp_sites.subtract(all_samples_comp_sites.median(axis=0), axis=1)
        ax = sns.heatmap(
            data = all_samples_comp_sites_dmf, annot=False, xticklabels=False, yticklabels=False, 
            cmap="icefire", vmin=-1, vmax=1, center=0,
            cbar_kws={'label': r'$\Delta$MF'}, ax=axes
            )#, cmap="icefire", vmin=-1, vmax=1, center=0
        # label axes
        ax.set_xlabel('Comparison sites')
        ax.set_ylabel('Samples')
        # add a y tick for the mutated sample, make tick label red, and rotate 90 degrees
        ax.set_yticks(np.arange(.5, len(samples_to_plot)+.5, 1))
        ax.set_yticks([int(len(utils.half(matched_samples, 'first')))+.5])
        ax.set_yticklabels(['Mutated sample'], color='red', rotation=90, ha='center', rotation_mode='anchor')
        ax.tick_params(axis='y', which='major', pad=5)
        # slightly seperate mutated CpG
        ax.add_patch(Rectangle((mut_pos-(magnify_mut_factor/2), 0), magnify_mut_factor, len(all_samples_comp_sites), fill=False, edgecolor='white', lw=4))
        # add second around the others to make same height
        """ax.add_patch(Rectangle((100, 0), len(all_samples_comp_sites.columns), len(all_samples_comp_sites), fill=False, edgecolor='white', lw=2))"""
        # add a tick label for the mutated CpG
        tick_locs = [0, mut_pos, len(distances)]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([str(int(-1*distances[0]/1000000))+'Mbp', 'Mutated site', str(int(distances[-1]/1000000))+'Mbp'], ha='center', rotation_mode='anchor')
        colors = ['black', 'red', 'black']
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
        return all_samples_comp_sites


class mutationScan:
    def __init__(
        self,
        all_mut_w_age_df: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame, 
        all_methyl_age_df_t: pd.DataFrame,
        corr_dir: str,
        age_bin_size: int, 
        max_dist: int,
        num_correl_sites: float,
        num_background_events: int,
        matched_sample_num: int,
        mut_collision_dist: int
        ) -> None:

        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.corr_dir = corr_dir
        self.age_bin_size = age_bin_size
        self.max_dist = max_dist
        self.num_correl_sites = num_correl_sites
        self.num_background_events = num_background_events
        self.matched_sample_num = matched_sample_num
        self.mut_collision_dist = mut_collision_dist
        # Preprocessing: subset to only mutations that are
        # non X and Y chromosomes and that occured in samples with measured methylation
        self.all_mut_w_age_df['mut_loc'] = self.all_mut_w_age_df['chr'] + ':' + self.all_mut_w_age_df['start'].astype(str)
        self.all_mut_w_age_df = self.all_mut_w_age_df.loc[
            # (self.all_mut_w_age_df['mutation'] == 'C>T') &
            (self.all_mut_w_age_df['chr'] != 'X')  
            & (self.all_mut_w_age_df['chr'] != 'Y')
            & (self.all_mut_w_age_df['case_submitter_id'].isin(self.all_methyl_age_df_t.index)),
            :]
        # join self.all_mut_w_age_df with the illumina_cpg_locs_df
        all_mut_w_age_illum_df = self.all_mut_w_age_df.copy(deep=True)
        all_mut_w_age_illum_df['start'] = pd.to_numeric(self.all_mut_w_age_df['start'])
        self.all_mut_w_age_illum_df = all_mut_w_age_illum_df.merge(
                                        self.illumina_cpg_locs_df, on=['chr', 'start'], how='left'
                                        )
        # subset illumina_cpg_locs_df to only the CpGs that are measured
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)
            ]
        # and remove chr X and Y
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[
            (self.illumina_cpg_locs_df['chr'] != 'X')
            & (self.illumina_cpg_locs_df['chr'] != 'Y')
            ]

    def correct_pvals(
        self, 
        all_metrics_df: pd.DataFrame,
        one_background: bool = False,
        pvals_to_correct: list = []
        ):
        """
        Correct each mutation events pvalue based on the background pvalues
        @ all_metrics_df: a dataframe with all the metrics for each mutation event
        @ one_background: if True, use all background mutation events to correct each mut event's pvals,
                            not just those which have index_event == that mut_event
        @ pvals_to_correct: a list of the pvals to correct
        @ returns: The all_metrics_df with the sig columns added
        """
        def get_sigs(mut_row, cutoffs):
            this_mut_cutoffs = cutoffs[mut_row['mut_event']]
            try:
                return mut_row[this_mut_cutoffs[0].index] < this_mut_cutoffs[0]
            except:
                print(mut_row)
                print(this_mut_cutoffs)
                print(cutoffs)
                sys.exit(1)

        def one_cutoff(background_pvals, num_mut_events):
            # for each column, get the 5th percentile value and bonf. correct it
            background_pvals_cutoffs = background_pvals[pvals_to_correct].quantile(0.05) / num_mut_events
            return background_pvals_cutoffs

        def get_cutoffs_one_background(metrics_df, mut_events):
            num_mut_events = len(mut_events)
            background_pvals = metrics_df.loc[
                (metrics_df['is_background'] == True) # get background samples
                & (metrics_df['mutated_sample'] == True) # and get just one of them
                , :].drop_duplicates(subset=['mut_event']) # get just one row per background mut event
            all_background_pvals_cutoff = one_cutoff(background_pvals, num_mut_events)
            return all_background_pvals_cutoff
            
        def get_cutoffs(metrics_df, mut_events):
            num_mut_events = len(mut_events)
            cutoffs = defaultdict(list)
            background_pvals = metrics_df.loc[
                (metrics_df['is_background'] == True) # get background samples
                & (metrics_df['mutated_sample'] == True) # and get just one of them
                , :].drop_duplicates(subset=['mut_event']) # get just one row per background mut event
            # for each mutation event
            for mut_event in mut_events:
                # get this mut event's background sites
                this_background_pvals = background_pvals.loc[
                    (background_pvals['index_event'] == mut_event),
                    :]
                # get the 5th percentile value for each pvalue of this mut event's background sites
                background_pvals_cutoffs = one_cutoff(this_background_pvals, num_mut_events)
                cutoffs[mut_event].append(background_pvals_cutoffs)
            return cutoffs

        # get real mutation events
        real_muts = all_metrics_df.loc[
            (all_metrics_df['index_event'] == 'self') # redundant, makes sure not a background site
            & (all_metrics_df['mutated_sample'] == True) # makes sure is the mutated sample, for uniqueness
            & (all_metrics_df['is_background'] == False) # makes sure not a background site
            , : ]
        # and get the unique mutation events from these
        mut_events = real_muts['mut_event'].unique()
        if one_background:
            cutoffs = get_cutoffs_one_background(all_metrics_df, mut_events)
            all_metrics_df[[pval_name + '_sig' for pval_name in pvals_to_correct]] = all_metrics_df[pvals_to_correct] < cutoffs
            return all_metrics_df
        else: # correct each mutation event's pvals based on the background pvals for that mutation event
            cutoffs = get_cutoffs(all_metrics_df, mut_events)
            real_muts[[pval_name + '_sig' for pval_name in pvals_to_correct]] = real_muts.apply(
                lambda mut_row: get_sigs(mut_row, cutoffs), axis=1
                )
            return real_muts

    def volcano_plot(
        self,
        all_metrics_df: pd.DataFrame,
        pval_col: str,
        sig_col: str,
        effect_col: str
        ) -> None:
        """
        Plot a volcano plot of the the cumul_delta_mf and pval
        """
        # TODO: deal with pvalues == 0
        # first select only the rows with measurements for the mutated sample
        real_muts_df = all_metrics_df.loc[
            (all_metrics_df['index_event'] == 'self') # redundant, makes sure not a background site
            & (all_metrics_df['mutated_sample'] == True) # makes sure is the mutated sample, for uniqueness
            & (all_metrics_df['is_background'] == False) # makes sure not a background site
            , : ]
        real_muts_df['abs_' + effect_col] = real_muts_df[effect_col].abs()

        # then get median delta_mf for each event
        med_delta_mf = real_muts_df.groupby('mut_event')[effect_col].median()
        abs_median_delta_mf = real_muts_df.groupby('mut_event')['abs_' + effect_col].median()
        # and get pvalue for each event 
        # doesn't matter if min, max, or mean cause all pvals are same for a mut event
        pvals = real_muts_df.groupby('mut_event')[pval_col].min()
        sigs = real_muts_df.groupby('mut_event')[sig_col].min()
        grouped_volc_metrics = pd.merge(abs_median_delta_mf, pd.merge(sigs, pd.merge(med_delta_mf, pvals, on='mut_event'), on='mut_event'), on='mut_event')
        grouped_volc_metrics['log10_pval'] = grouped_volc_metrics[pval_col].apply(lambda x: -np.log10(x))
        _, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100, gridspec_kw={'width_ratios': [3, 1]})
        # volcano plot
        sns.scatterplot(
            y = 'log10_pval', x = effect_col, data = grouped_volc_metrics, alpha=0.3,
            hue = sig_col, palette = {True: 'maroon', False: 'grey'}, ax = axes[0]
            )
        # round the pvalue and set coloirs to red if significant and grey if not
        axes[0].legend(loc='upper center', labels=[f'p < sig. threshold', f'p >= sig. threshold'], title='p-value')
        leg = axes[0].get_legend()
        leg.legendHandles[0].set_color('maroon')
        leg.legendHandles[1].set_color('grey') 
        axes[0].set_xlabel(r"Median $\Delta$MF")
        axes[0].set_ylabel('-log10 p-value')
        # barplot of the number of significant and non-significant mutations
        sns.countplot(x = sig_col, data = grouped_volc_metrics, palette = {True: 'maroon', False: 'grey'}, ax = axes[1])
        # set x ticks
        axes[1].set_xlabel('Corrected p-value')
        axes[1].set_xticklabels([f'Not significant', f'Significant' ])
        axes[1].set_ylabel('Count of mutation events')

        """# same but absolute cumul
        _, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100, gridspec_kw={'width_ratios': [3, 1]})
        # volcano plot
        sns.scatterplot(
            y = 'log10_pval', x = 'abs_delta_mf_median', data = grouped_volc_metrics, alpha=0.3,
            hue = sig_col, palette = {True: 'maroon', False: 'grey'}, ax = axes[0]
            )
        axes[0].legend(loc='upper left', labels=[f'p < sig. threshold', f'p >= sig. threshold'], title='Bonferroni p-value')
        leg = axes[0].get_legend()
        leg.legendHandles[0].set_color('maroon')
        leg.legendHandles[1].set_color('grey')
        axes[0].set_xlabel(r"Median absolute $\Delta$MF")
        axes[0].set_ylabel('-log10 p-value')
        # barplot of the number of significant and non-significant mutations
        sns.countplot(x = sig_col, data = grouped_volc_metrics, palette = {True: 'maroon', False: 'grey'}, ax = axes[1])
        axes[1].set_xlabel('Corrected p-value')
        axes[1].set_xticklabels([f'Not significant',f'Significant' ])
        axes[1].set_ylabel('Count of mutation events')"""
        return grouped_volc_metrics

    def _join_with_illum(
        self, 
        in_df, 
        different_illum = None
        ):
        """
        Join the dataframe with the illumina_cpg_locs_df
        """
        df = in_df.copy(deep=True)
        # split 'mut_loc' into 'chr' and 'start'
        df[['chr', 'start']] = df['mut_loc'].str.split(':', expand=True)
        # convert start column to int with to_numeric
        df['start'] = pd.to_numeric(df['start'])
        if different_illum is None:
            df_w_illum = df.merge(self.illumina_cpg_locs_df, on=['chr', 'start'], how='left')
        else:
            df_w_illum = df.merge(different_illum, on=['chr', 'start'], how='left')
        return df_w_illum

    def preproc_correls(
        self, 
        out_dir
        ) -> None:
        """
        Calculate the correlation matrix for each dataset within each chromosome and output to file
        """
        for chrom in self.all_mut_w_age_df['chr'].unique():
            this_chr_measured_cpgs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] == chrom]
            for dset in self.all_mut_w_age_df['dataset'].unique():
                this_chr_dset_methyl_df = self.all_methyl_age_df_t.loc[:, this_chr_measured_cpgs['#id'].to_list() + ['dataset']]
                this_chr_dset_methyl_df = this_chr_dset_methyl_df[this_chr_dset_methyl_df['dataset'] == dset]
                corr_df = this_chr_dset_methyl_df.corr()
                corr_df.to_parquet(os.path.join(out_dir, 'chr{}_{}.parquet'.format(chrom, dset)))
                print(chrom, dset, flush=True)

    def _same_age_and_tissue_samples(
        self, 
        sample_name: str
        ) -> list:
        """
        Return a list of samples with matched age and tissue as the sample_name
        @ sample_name
        @ returns: list of sample names
        """
        # get this sample's age and dataset
        this_age = self.all_methyl_age_df_t.loc[sample_name, 'age_at_index']
        this_dset = self.all_methyl_age_df_t.loc[sample_name, 'dataset']
        # get the mf all other samples of within age_bin_size/2 years of age on either side
        matched_samples = self.all_methyl_age_df_t.loc[
            (self.all_methyl_age_df_t['dataset'] == this_dset)
            & (np.abs(self.all_methyl_age_df_t['age_at_index'] - this_age) <= self.age_bin_size/2) 
            ].index
        # drop the mutated sample itself
        matched_samples_no_mut = matched_samples.drop(sample_name)
        return matched_samples_no_mut.to_list()

    def _detect_effect_in_other_samples(
        self,
        mut_row: pd.Series
        ) -> list:
        """
        Detect samples that may have been affected by a mutation in the same region as the mutation we are testing, to discard these samples as they are a bad comparison
        @ sites_to_test: list of sites to test, cg######### format
        @ mut_row: the row from comparison_sites_df specifying the mutation
        @ returns: list of samples that do not have any C>T mutations in max_dist from any of the sites in sites_to_test
        """
        # so it works when called from heatmap and comethylation scan 
        mut_row = mut_row.copy(deep=True).squeeze()
        
        # select rows of self.all_mut_w_age_df that have the same chr and dataset and as mut_row
        relevant_mutations = self.all_mut_w_age_df.loc[
            (self.all_mut_w_age_df['chr'] == mut_row['chr']) 
            & (self.all_mut_w_age_df['case_submitter_id'].isin(mut_row['matched_samples']))
            ]
        if len(relevant_mutations) == 0:
            return []
        """# detect samples that have a mutation in the max_dist window of any of the sites in sites_to_test
        sites_to_test = mut_row['comparison_sites'].to_list()
        sites_to_test_locs = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['#id'].isin(sites_to_test)
            ] 
        # select rows of relevant_mutations that are within max_dist of any of the sites_to_test
        try:
            have_illegal_muts = relevant_mutations.loc[
                relevant_mutations.apply(lambda row:
                    any(np.abs(row['start'] - sites_to_test_locs['start']) <= self.max_dist),
                    axis=1)
                ]
        except:
            print(relevant_mutations)
            print(relevant_mutations.apply(
                lambda row: any(np.abs(row['start'] - sites_to_test_locs['start']) <= self.max_dist), 
                axis=1))
            sys.exit(1)"""
        # detect samples that have a mutation in the mutated site or within max_dist of it
        have_illegal_muts = relevant_mutations.loc[
                 (relevant_mutations['mut_loc'] == mut_row['mut_loc']) 
                 | (np.abs(relevant_mutations['start'] - mut_row['start']) <= self.mut_collision_dist)
                 ]
        """pd.concat(
            [have_illegal_muts, 
             relevant_mutations.loc[
                 (relevant_mutations['mut_loc'] == mut_row['mut_loc']) 
                 | (np.abs(relevant_mutations['start'] - mut_row['start']) <= self.mut_collision_dist)
                 ]
             ])"""
        return have_illegal_muts['case_submitter_id'].to_list()

    def _compare_sites(
        self, 
        comparison_site_mfs: pd.DataFrame, 
        mut_sample_name: str
        ) -> pd.DataFrame:
        """
        Calculate effect size and pvalue for each comparison
        """
        # get the difference of each sample from the median of the other samples
        median_diffs = comparison_site_mfs.apply(
            lambda row: row - np.nanmedian(comparison_site_mfs.drop(row.name), axis = 0),
            axis=1
            )
        abs_median_diffs = np.abs(median_diffs)
        # zscore of the absolute difference from the median
        zscore_abs_median_diffs = abs_median_diffs.apply(
            lambda row: (row - np.nanmean(abs_median_diffs, axis=0)) / np.nanstd(abs_median_diffs, axis=0),
            axis=1
            )
        # mean diff
        mean_diffs = comparison_site_mfs.apply(
            lambda row: row - np.nanmean(comparison_site_mfs.drop(row.name), axis = 0),
            axis=1
            )
        abs_mean_diffs = np.abs(mean_diffs)
        # zscore of the difference from the median
        zscore_abs_mean_diffs = abs_mean_diffs.apply(
            lambda row: (row - np.nanmean(abs_mean_diffs, axis=0)) / np.nanstd(abs_mean_diffs, axis=0),
            axis=1
            )
        
        metrics = median_diffs.stack().reset_index()
        metrics.columns = ['sample', 'measured_site', 'delta_mf_median']
        # create column called 'mutated' that is True if the sample is the mutated sample
        metrics['mutated_sample'] = metrics['sample'] == mut_sample_name
        # add the zscore of the difference from the median
        stacked_zscore_abs_median_diffs = zscore_abs_median_diffs.stack().reset_index()
        stacked_zscore_abs_median_diffs.columns = ['sample', 'measured_site', 'zscore_abs_delta_mf_median']
        metrics = metrics.merge(
            stacked_zscore_abs_median_diffs,
            on = ['sample', 'measured_site'],
            how = 'left'
            )
        """# add mean difference
        stacked_mean_diffs = mean_diffs.stack().reset_index()
        stacked_mean_diffs.columns = ['sample', 'measured_site', 'delta_mf_mean']
        metrics = metrics.merge(
            stacked_mean_diffs,
            on = ['sample', 'measured_site'],
            how = 'left'
            )
        # add zscore of the difference from the mean
        stacked_zscore_abs_mean_diffs = zscore_abs_mean_diffs.stack().reset_index()
        stacked_zscore_abs_mean_diffs.columns = ['sample', 'measured_site', 'zscore_abs_delta_mf_mean']
        metrics = metrics.merge(
            stacked_zscore_abs_mean_diffs,
            on = ['sample', 'measured_site'],
            how = 'left'
            )"""
        metrics.columns = ['sample', 'measured_site', 'delta_mf_median', 'mutated_sample', 'zscore_abs_delta_mf_median']#, 'delta_mf_mean', 'zscore_abs_delta_mf_mean']
        return metrics

    def effect_on_each_site(
        self, 
        comparison_sites_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        For each mutation, get the effect of the mutation on each comparison site
        @ comparison_sites_df
        @ returns: df with statistical test and effect sizes
        """        
        def process_row(mut_row):
            # get the same age and dataset samples
            matched_samples = mut_row['matched_samples']
            # exclude samples that have ANY mutations within max_dist of a comparison site or the mutated site
            samples_to_exclude = self._detect_effect_in_other_samples(mut_row)
            # drop entries of matched_samples that are in samples_to_exclude 
            matched_samples = [s for s in matched_samples if s not in samples_to_exclude]
            if len(matched_samples) < self.matched_sample_num:
                return None
            """
            STOPPING THIS FOR NOW
            # limit to matched_sample_num samples, no more, no less, 
            # to avoid statistical bias towards samples with more similar samples
            if len(matched_samples) < self.matched_sample_num:
                return None
            else:
                matched_samples = matched_samples[:self.matched_sample_num]
            """
            # create a list of the matched samples and mutated sample 
            all_samples = matched_samples
            all_samples.append(mut_row['case_submitter_id'])
            comparison_site_mfs = self.all_methyl_age_df_t.loc[all_samples, mut_row['comparison_sites']]
            # measure the change in methylation between sites in the mutated samples and in other non-mutated samples of the same age
            metrics = self._compare_sites(comparison_site_mfs, mut_sample_name = mut_row['case_submitter_id'])
            
            metrics['mut_loc'], metrics['mut_event'], metrics['is_background'], metrics['index_event'] = mut_row['mut_loc'], mut_row['mut_event'], mut_row['is_background'], mut_row['index_event']
            
            cpg_to_dist_dict = dict(zip(mut_row['comparison_sites'], mut_row['comparison_dists']))
            
            metrics['measured_site_dist'] = metrics['measured_site'].map(cpg_to_dist_dict)
            # add to output
            return metrics
        
        tqdm.pandas(
            desc="Calculating effect of mutation on comparison sites",
            miniters=len(comparison_sites_df)/10
            )
        # apply process_row across each row of comparison_sites_df
        all_metrics_dfs = comparison_sites_df.progress_apply(process_row, axis=1)
        # drop none values
        print("Dropping None values", flush=True)
        before = len(all_metrics_dfs)
        all_metrics_dfs = all_metrics_dfs.dropna()
        all_metrics_dfs = all_metrics_dfs.to_list()
        after = len(all_metrics_dfs)
        print(f"dropped {before - after} mutation events bc they missed mutated sample num", flush=True)
        # concat all dfs
        all_metrics_df = pd.concat(all_metrics_dfs)
        print("Done getting effect of mutation on each site", flush=True)
        return all_metrics_df

    def _find_collisions(
        self,
        muts_df: pd.DataFrame
        ) -> pd.DataFrame:
        # Merge the two DataFrames on the 'chr' column, using an inner join
        merged_df = pd.merge(muts_df, self.illumina_cpg_locs_df, on=['chr'], how='inner')
        # Create a new DataFrame containing only the rows where the 'start' column values
        # are not within max.dist of each other
        no_close_mut = merged_df.loc[np.abs(merged_df['start_x'] - merged_df['start_y']) > self.max_dist]
        return no_close_mut

    def _locs_near_mutation(self, background_chrom, sample):
        """
        For each mutation in sample, get locations that are within max_dist of the mutation
        """
        # get the locations of mutations in this sample
        sample_mut_locs = self.all_mut_w_age_df.loc[
            (self.all_mut_w_age_df['case_submitter_id'] == sample) 
            & (self.illumina_cpg_locs_df['chr'] == background_chrom), 'start'
            ].values.tolist()
        # make a list of numbers from mut_loc - max_dist to mut_loc + max_dist for each mut_loc
        locs_near_mutation = []
        for mut_loc in sample_mut_locs:
            locs_near_mutation.extend(range(mut_loc - self.max_dist, mut_loc + self.max_dist))
        return locs_near_mutation

    def _random_site_near_cpg(
        self,
        row):
        """
        Applied across a dataframe with a 'chr' column, returns a random location near a CpG on this chr
        """
        this_chr_cpgs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] == row['chr']]
        # randomly choose a row from this_chr_cpgs
        rand_cpg = this_chr_cpgs.sample(n=1, random_state=1)
        rand_cpg_loc = rand_cpg['start'].values[0]
        # randomly choose a location near this cpg (+/- self.max_dist)
        chosen_loc = random.randrange(rand_cpg_loc - self.max_dist, rand_cpg_loc + self.max_dist)
        return chosen_loc
    
    def _choose_background_sites_meqtlDB(
        self, 
        comparison_sites_df: pd.DataFrame,
        meqtl_DB_df: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Choose num_background_events for each mutation event in comparison_sites_df from meQTL db
        meQTL's that do not actually have a mutation (in any sample)
        """
        # merge mutations with meQTL database, meQTLs without a mutaition
        mut_in_meqtl = meqtl_DB_df.merge(self.all_mut_w_age_illum_df, how='left', right_on='mut_loc', left_on = 'snp')
        # drop rows that no not have nan in mut_loc, keeping unmutated meQTLs
        mut_in_meqtl = mut_in_meqtl[mut_in_meqtl['mut_loc'].isna()]
        # drop rows with cpg not in methyl df
        mut_in_meqtl = mut_in_meqtl.loc[mut_in_meqtl['cpg'].isin(self.all_methyl_age_df_t.columns)]

        all_background_sites = []
        unique_meqtl_snps = pd.Series(mut_in_meqtl['snp'].unique()) # so no bias towards snps with more cpgs
        print("got unique meqtl snps", flush=True)
        for i, row in comparison_sites_df.iterrows():
            # choose num_background_events fake mutations in meQTLs from the sample of this row
            # randomly sample num_background_events snps from mut_in_meqtl
            chosen_snps = unique_meqtl_snps.sample(
                n=self.num_background_events, random_state=1, replace = False
                )
            # get corresponding cpgs
            background_comp_sites = mut_in_meqtl.loc[mut_in_meqtl['snp'].isin(chosen_snps)]
            # turn this into a df with the same columns as comparison_sites_df, with same values
            background_comp_sites['mut_event'] = row['case_submitter_id'] + '_' + background_comp_sites['snp']
            background_comp_sites = background_comp_sites.groupby('mut_event')['cpg'].apply(list).to_frame().reset_index()
            background_comp_sites.rename(columns={'cpg': 'comparison_sites'}, inplace=True) 
            background_comp_sites['comparison_dists'] = [[] for _ in range(len(background_comp_sites))]
            background_comp_sites['comparison_dists'] = background_comp_sites.apply(
                lambda mut_event: [x for x in range(len(mut_event['comparison_sites']))], axis=1
            )
            background_comp_sites['case_submitter_id'] = row['case_submitter_id']
            background_comp_sites['mut_loc'] = background_comp_sites['mut_event'].apply(lambda x: x.split('_')[1])
            background_comp_sites['chr'] = background_comp_sites['mut_loc'].apply(lambda x: x.split(':')[0])
            background_comp_sites['start'] = background_comp_sites['mut_loc'].apply(lambda x: int(x.split(':')[1]))
            background_comp_sites['matched_samples'] = [row['matched_samples'] for _ in range(len(background_comp_sites))] # every background event has the same matched samples
            # make it a background event
            background_comp_sites['is_background'] = True
            background_comp_sites['index_event'] = row['mut_event']
            # add to list of dataframes of each real comparison sites comp sites
            all_background_sites.append(background_comp_sites)
            print(f"finished {100*i/len(comparison_sites_df)}% of background events", flush=True)
        all_background_comp_sites = pd.concat(all_background_sites)
        # concat with comparison_sites_df to be processed together
        comparison_sites_df = pd.concat([comparison_sites_df, all_background_comp_sites])
        comparison_sites_df.reset_index(inplace=True, drop=True)
        # return the comparison sites df with the background events added
        return comparison_sites_df
    
    def _choose_background_sites_dist(
        self,
        comparison_sites_df: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Choose num_background_events for each mutation event in comparison_sites_df
        """
        num_mut_events = len(comparison_sites_df)
        # ranomly choose num_background_events from the keys of CHROM_LENGTHS
        rand_chrs = np.random.choice(
            list(CHROM_LENGTHS.keys()), 
            size= self.num_background_events * num_mut_events, replace=True
            )
        background_events = pd.DataFrame({'chr': rand_chrs})
        # randomly choose a location near a CpG on this chr by applying _random_site_near_cpg to each row 
        tqdm.pandas(desc="Getting background events near cpgs", miniters=len(background_events)/10)
        background_events['start'] = background_events.progress_apply(self._random_site_near_cpg, axis=1)
        background_events['end'] = background_events['start']
        background_events['mut_loc'] = background_events['chr'] + ':' + background_events['start'].astype(str)
        # concat the comparison sites df to itself to populate background_events columns
        repeated_comp_sites_df = pd.concat([comparison_sites_df] * self.num_background_events, ignore_index=True)
        repeated_comp_sites_df.reset_index(drop=True, inplace=True)
        background_events['dataset'] = repeated_comp_sites_df['dataset']
        background_events['case_submitter_id'] = repeated_comp_sites_df['case_submitter_id']
        background_events['matched_samples'] = repeated_comp_sites_df['matched_samples']
        background_events['is_background'] = True # redundant with index_event but keeping for now
        background_events['mut_delta_mf'] = np.nan
        # index event is the real mutation event that the background site is for
        background_events['index_event'] = repeated_comp_sites_df['mut_event']
        # mut event is the background site
        background_events['mut_event'] = background_events['case_submitter_id'] + '_' + background_events['mut_loc']
    
        # now get the comparison sites and dists for these background sites
        background_events = self._get_nearby_measured_cpgs(background_events)
        # drop those without any nearby CpGs
        background_events = background_events[background_events['comparison_sites'].apply(len) > 0]
        # check if we now have too many or too few background sites
        # we want self.num_background_events for each index event
        index_value_counts = background_events['index_event'].value_counts().to_frame()
        index_value_counts.columns = ['num_background_events']
        index_value_counts['diff_from_target'] = index_value_counts['num_background_events'] - self.num_background_events
        # if there are any nonzero diff_from_target raise an error
        if index_value_counts[index_value_counts['diff_from_target'] < 0].shape[0] != 0:
            print(index_value_counts)
            print(background_events)
            print('Not right number of background sites')

        # concat with comparison_sites_df to be processed together
        comparison_sites_df = pd.concat([comparison_sites_df, background_events])
        comparison_sites_df.reset_index(inplace=True, drop=True)
        # return the comparison sites df with the background events added
        return comparison_sites_df
        
    def _choose_background_sites_corr(
        self,
        comparison_sites_df: pd.DataFrame,
        corr_direction: str
        ) -> pd.DataFrame:
        num_mut_events = len(comparison_sites_df)

        cpgs_to_choose_from = self.illumina_cpg_locs_df['#id'].to_list()
        rand_cpgs = np.random.choice(
            cpgs_to_choose_from, size= self.num_background_events * num_mut_events, replace=True
            )
        # create dataframe of background events based on the chosen cpgs
        # mapping chr and start from illumina cpg locs
        background_events = pd.DataFrame({'#id': rand_cpgs})
        background_events = background_events.merge(
            self.illumina_cpg_locs_df[['#id', 'chr', 'start']], on='#id', how='left'
            )
        background_events['mut_loc'] = background_events['chr'] + ':' + background_events['start'].astype(str)
        # concat the comparison sites df to itself to populate background_events columns
        repeated_comp_sites_df = pd.concat(
            [comparison_sites_df] * self.num_background_events, ignore_index=True
            )
        repeated_comp_sites_df.reset_index(drop=True, inplace=True)
        background_events['dataset'] = repeated_comp_sites_df['dataset']
        background_events['case_submitter_id'] = repeated_comp_sites_df['case_submitter_id']
        background_events['matched_samples'] = repeated_comp_sites_df['matched_samples']
        background_events['is_background'] = True # redundant with index_event but keeping for now
        background_events['mut_delta_mf'] = np.nan
        # index event is the real mutation event that the background site is for
        background_events['index_event'] = repeated_comp_sites_df['mut_event']
        # mut event is the background site
        background_events['mut_event'] = background_events['case_submitter_id'] + '_' + background_events['mut_loc']
        
        # choose the most correlated sites from the preprocessed data
        tqdm.pandas(desc="Getting background comp sites", miniters=len(background_events)/10)
        background_events['comparison_sites'] = background_events.progress_apply(
            lambda mut_event: self._select_correl_sites_preproc(mut_event, corr_direction),
            axis = 1
            )
        background_events['comparison_dists'] = [
            [i for i in range(self.num_correl_sites)] 
            for _ in range(len(background_events))
            ]
        # drop those without any comparison sites CpGs
        background_events = background_events[background_events['comparison_sites'].apply(len) > 0]
        
        # concat with comparison_sites_df to be processed together
        comparison_sites_df = pd.concat([comparison_sites_df, background_events])
        comparison_sites_df.reset_index(inplace=True, drop=True)
        # return the comparison sites df with the background events added
        return comparison_sites_df
        
    def _select_correl_sites_preproc(
        self,
        mut_event: pd.Series,
        corr_direction: str
        ) -> list:
        """
        Gets the num sites that are either the most positively or negativelty correlated sites with in_cpg
        @ corr_df: correlation matrix for all sites from one dataset on one chromosome
        @ mut_cpg: the cpg that we are interested in correlating with
        @ corr_direction: 'pos' or 'neg' for positive or negative correlation
        @ returns: list of chosen sites
        """
        # read in the the correlation matrix for this chrom and dataset
        corrs = pd.read_parquet(
            os.path.join(self.corr_dir, 'chr{}_{}.parquet'.format(mut_event['chr'], mut_event['dataset'])),
            columns = [mut_event['#id']]
            )
        # convert corrs to series
        corrs = corrs.iloc[:, 0]
        # get the value of the 'percentile' highest correlation
        if corr_direction == 'pos':
            q = corrs.quantile(1, interpolation='lower')
            # select num sites closest to q, but not including mut_cpg (which will have a correlation of 1)
            return corrs.iloc[(corrs - q).abs().argsort().iloc[1: self.num_correl_sites + 1]].index.to_list()
        elif corr_direction == 'neg':
            q = corrs.quantile(0, interpolation='higher')
            # here do not need to exclude most correlated site, since it will not be the mut_cpg
            return corrs.iloc[(corrs - q).abs().argsort().iloc[:self.num_correl_sites]].index.to_list()

    def _mut_site_delta_mf(
        self,
        mut_event: pd.Series,
        ) -> float:
        """
        @ mut_event: a row from comparison_sites_df
        @ returns: the delta MF of the mutated site
        """
        # get the MF of the mutation in matched samples
        mut_sample_mf = self.all_methyl_age_df_t.loc[mut_event['case_submitter_id'], mut_event['#id']]
        # get the mutated CpG's MF in the matched samples
        matched_mfs = self.all_methyl_age_df_t.loc[mut_event['matched_samples'], mut_event['#id']]
        # delta_mf
        return mut_sample_mf - matched_mfs.median()

    def _select_correl_sites(
        self,
        mut_event: pd.Series,
        corr_direction: str
        ) -> list:
        """
        Just in time correlation to find the most correlated sites to the mutation event CpG in matched samples
        """
        # get the mutated CpG's MF in the matched samples
        mut_cpg_mf = self.all_methyl_age_df_t.loc[mut_event['matched_samples'], mut_event['#id']]
        # get the MF of all same chrom CpGs in matched samples
        same_chrom_cpgs = self.illumina_cpg_locs_df.loc[
            (self.illumina_cpg_locs_df['chr'] == mut_event['chr']) 
            & (self.illumina_cpg_locs_df['#id'] != mut_event['#id']), # exclude the mut_cpg
            '#id'].values
        same_chrom_cpgs_mf = self.all_methyl_age_df_t.loc[mut_event['matched_samples'], same_chrom_cpgs]
        # get correlation between mut_cpg and all same chrom CpGs
        corrs = same_chrom_cpgs_mf.corrwith(mut_cpg_mf, axis=0)
        if corr_direction == 'pos':
            q = corrs.quantile(1, interpolation='lower')
            return corrs.iloc[(corrs - q).abs().argsort().iloc[:self.num_correl_sites]].index.to_list()
        elif corr_direction == 'neg':
            q = corrs.quantile(0, interpolation='higher')
            # here do not need to exclude most correlated site, since it will not be the mut_cpg
            return corrs.iloc[(corrs - q).abs().argsort().iloc[:self.num_correl_sites]].index.to_list()

    def _get_correl_based_comp_sites(
        self, 
        start_num_mut_to_process: int,
        end_num_mut_to_process: int,
        corr_direction: str
        ) -> pd.DataFrame:
        """
        Find the mutation events that meet the criteria (min_VAF, max_delta_mf) and get choose
        comparison sites to be those CpGs which are most correlated with the mutation event CpG 
        in matched samples
        @ min_VAF_percentile: the minimum VAF percentile for a mutation event to be considered
        @ max_delta_mf_percentile: the maximum delta MF percentile for a mutation event to be considered
        @ corr_direction: 'pos' or 'neg' for positive or negative correlation
        """
        pd.options.mode.chained_assignment = None  # default='warn'
        # subset to mutations in the measured CpGs (i.e. in the illumina CpGs)
        valid_muts_w_illum = self.all_mut_w_age_illum_df.loc[
                self.all_mut_w_age_illum_df['#id'].isin(self.illumina_cpg_locs_df['#id'])
                ]
        print("Number mutation events being processed: {}".format(len(valid_muts_w_illum)), flush=True)
        # get same age and tissue samples for each, keeping only mutations with at least self.matched_sample_num
        tqdm.pandas(desc="Getting matched samples", miniters=len(valid_muts_w_illum)/10)
        valid_muts_w_illum['matched_samples'] = valid_muts_w_illum.progress_apply(
            lambda mut_event: self._same_age_and_tissue_samples(mut_event['case_submitter_id']), axis = 1
            )
        valid_muts_w_illum = valid_muts_w_illum.loc[
            valid_muts_w_illum['matched_samples'].apply(len) >= self.matched_sample_num
            ]
        # TECHNICALLY DEPRECATED< NO LONGER USED
        # get the delta MF of the mutated site
        tqdm.pandas(desc="Getting mut site delta MF", miniters=len(valid_muts_w_illum)/10)
        valid_muts_w_illum['mut_delta_mf'] = valid_muts_w_illum.progress_apply(
            lambda mut_event: self._mut_site_delta_mf(mut_event), axis=1
            )
        # sort mutations high to low by DNA_VAF
        valid_muts_w_illum = valid_muts_w_illum.sort_values(
            by='DNA_VAF',
            ascending=False
            #by=['mut_delta_mf', 'DNA_VAF'],
            #ascending=[True, False]
            )
        # select top mutations for further processing
        valid_muts_w_illum = valid_muts_w_illum.iloc[
            start_num_mut_to_process : end_num_mut_to_process,
            :]
        print(
            "Number mutation events being processed after filtering for matched sample number: {}".format(len(valid_muts_w_illum)), flush=True
            )
        # choose comparison sites
        tqdm.pandas(desc="Getting comparison sites", miniters=len(valid_muts_w_illum)/10)
        # changed from _select_correl_sites to match BG and FG methods
        valid_muts_w_illum['comparison_sites'] = valid_muts_w_illum.progress_apply(
            lambda mut_event: self._select_correl_sites_preproc(mut_event, corr_direction), axis = 1
            )
        valid_muts_w_illum['comparison_dists'] = [
            [i for i in range(self.num_correl_sites)] 
            for _ in range(len(valid_muts_w_illum))
            ]
        valid_muts_w_illum['mut_loc'] = valid_muts_w_illum['chr'] + ':' + valid_muts_w_illum['start'].astype(str)
        valid_muts_w_illum['mut_event'] = valid_muts_w_illum['case_submitter_id'] + '_' + valid_muts_w_illum['mut_loc']
        valid_muts_w_illum['is_background'] = False
        valid_muts_w_illum['index_event'] = 'self'
        valid_muts_w_illum.reset_index(drop = True, inplace = True)
        pd.options.mode.chained_assignment = 'warn'
        return valid_muts_w_illum
    
    def _get_nearby_measured_cpgs(self, muts_df):
        """
        Given a df of mutations, get the measured cpgs within max_dist of each mutation
        @ muts_df: a df of mutations 
        @ returns: muts_df, a df of mutations with a 'comparison_sites' and 'comparison_dists' columns
        """
        # do this chrom by chrom to speed up
        for chrom in track(
                muts_df['chr'].unique(), 
                total = len(muts_df['chr'].unique()), 
                description = 'Finding nearby measured cpgs'
            ):
            # get the measured cpgs on this chromsome
            cpg_locs = self.illumina_cpg_locs_df[self.illumina_cpg_locs_df['chr'] == chrom]
            # get the mutations on this chromsome
            mut_locs = muts_df.loc[(muts_df['chr'] == chrom)]
            # for each mutation, get a list of the measured CpGs that are within max_dist 
            # but not the same CpG as the mutation (distance 0)
            muts_df.loc[mut_locs.index, 'comparison_sites'] = mut_locs.apply(
                lambda x: list(cpg_locs[(np.abs(x['start'] - cpg_locs['start']) <= self.max_dist) 
                                        & (x['start'] - cpg_locs['start'] != 0)]['#id']), axis = 1
                )
            # also get a list of the distances of these sites
            muts_df.loc[mut_locs.index, 'comparison_dists'] = mut_locs.apply(
                lambda x: list(cpg_locs[(np.abs(x['start'] - cpg_locs['start']) <= self.max_dist) 
                                        & (x['start'] - cpg_locs['start'] != 0)]['start'] - x['start']),
                                        axis = 1
                )
        return muts_df
    
    def _get_distance_based_comp_sites(
        self, 
        start_num_mut_to_process: int,
        end_num_mut_to_process: int,
        ) -> pd.DataFrame:
        """
        Find the measured cpgs within max_dist of each mutation to use as comparison sites
        @ start_num_mut_to_process: the number of mutations to start processing at
        @ end_num_mut_to_process: the number of mutations to end processing at
        @ returns: valid_muts_w_illum, a df of mutations that have at least one measured CpG within max_dist of the mutation. 'comparison_sites' column is a list of the measured cpgs within max_dist of the mutation.
        """
        pd.options.mode.chained_assignment = None  # default='warn'q
        # get df of all mutations
        valid_muts_w_illum = self.all_mut_w_age_illum_df
        # sort mutations high to low by DNA_VAF
        valid_muts_w_illum = valid_muts_w_illum.sort_values(by='DNA_VAF', ascending=False)
        # choose the 100,000 top VAF, which is >.6 VAF 
        subset_num = 5000
        valid_muts_w_illum = valid_muts_w_illum.iloc[:subset_num, :]
        min_vaf = valid_muts_w_illum['DNA_VAF'].min()
        print(f"First subsetting to {subset_num} mutations with highest VAF, corresponding to {min_vaf}, processing {len(valid_muts_w_illum)} for comparison sites and matched samples")
        # initialize empty cols
        valid_muts_w_illum['comparison_sites'] = [[] for _ in range(len(valid_muts_w_illum))]
        valid_muts_w_illum['comparison_dists'] = [[] for _ in range(len(valid_muts_w_illum))]
        # get comparison sites as the measured cpgs within max_dist of each mutation
        valid_muts_w_illum = self._get_nearby_measured_cpgs(valid_muts_w_illum)
        # drop all rows of valid_muts_w_illum where comparison_sites is empty
        valid_muts_w_illum = valid_muts_w_illum[
            valid_muts_w_illum['comparison_sites'].apply(lambda x: len(x) > 0)
            ]
        print(
            f"Number of mutation events with at least one comparison site within {self.max_dist}: {len(valid_muts_w_illum)}", flush=True
            )
        # get matched samples for each, keeping only mutations with at least self.matched_sample_num
        tqdm.pandas(desc="Getting matched samples", miniters=len(valid_muts_w_illum)/10)
        valid_muts_w_illum['matched_samples'] = valid_muts_w_illum.progress_apply(
            lambda mut_event: self._same_age_and_tissue_samples(mut_event['case_submitter_id']), axis = 1
            )
        # print distribution of number of matched samples across mutation events
        valid_muts_w_illum['num_matched_samples'] = valid_muts_w_illum['matched_samples'].apply(len)
        valid_muts_w_illum = valid_muts_w_illum.loc[
            valid_muts_w_illum['matched_samples'].apply(len) >= self.matched_sample_num
            ]
        print(
            f"Number mutation events being processed after filtering for matched sample number of {self.matched_sample_num}: {len(valid_muts_w_illum)}", flush=True
            )
        # select top mutations for further processing
        valid_muts_w_illum = valid_muts_w_illum.iloc[start_num_mut_to_process:end_num_mut_to_process, :]
        print(f"Number of mutation events being processed based on start_num_mut_to_process and end_num_mut_to_process: {len(valid_muts_w_illum)}")
        # set other column values
        valid_muts_w_illum.loc[:, 'mut_loc'] = valid_muts_w_illum['chr'] + ':' + valid_muts_w_illum['start'].astype(str)
        valid_muts_w_illum['mut_event'] = valid_muts_w_illum['case_submitter_id'] + '_' + valid_muts_w_illum['mut_loc']
        valid_muts_w_illum['is_background'] = False
        valid_muts_w_illum['index_event'] = 'self'
        valid_muts_w_illum.reset_index(drop = True, inplace = True)
        pd.options.mode.chained_assignment = 'warn'
        return valid_muts_w_illum 
    
    def _get_meqtlDB_based_comp_sites(
        self,
        meqtl_DB_df: pd.DataFrame, 
        start_num_mut_to_process: int, 
        end_num_mut_to_process: int
        ) -> pd.DataFrame:
        """
        Choose mutation events that are in meQTLs and choose comparison sites as the CpGs related to that meQTL
        """
        print("Getting comparison sites from meQTL database...")
        # merge mutations with meQTL database, keeping only mutations that are in meQTLs
        mut_in_meqtl = meqtl_DB_df.merge(self.all_mut_w_age_illum_df, how='left', right_on='mut_loc', left_on = 'snp')
        mut_in_meqtl.dropna(inplace=True, subset=['mut_loc'])
        print("number of meQTLs with mutations in them: ", len(mut_in_meqtl))
        # drop rows where CpG is not in self.all_methyl_age_df_t
        mut_in_meqtl = mut_in_meqtl.loc[mut_in_meqtl['cpg'].isin(self.all_methyl_age_df_t.columns)]
        print("number of meQTLs with mutations in them and CpGs in methyl data: ", len(mut_in_meqtl))
        # if there are no mutations in meQTL database, exit
        if len (mut_in_meqtl) == 0:
            sys.exit("No somatic mutations in meQTL database, exiting")
        
        # groupby mut_event, aggregating cpg values into a list
        mut_in_meqtl['mut_event'] = mut_in_meqtl['case_submitter_id'] + '_' + mut_in_meqtl['mut_loc']
        mut_in_meqtl = mut_in_meqtl.groupby('mut_event')['cpg'].apply(list).to_frame().reset_index()
        mut_in_meqtl.rename(columns={'cpg': 'comparison_sites'}, inplace=True) 
        # for each mutation event, add the distances to the comparison sites
        mut_in_meqtl['comparison_dists'] = [[] for _ in range(len(mut_in_meqtl))]
        mut_in_meqtl['comparison_dists'] = mut_in_meqtl.apply(
            lambda mut_event: [x for x in range(len(mut_event['comparison_sites']))], axis=1
        )
        # get columns back
        mut_in_meqtl['case_submitter_id'] = mut_in_meqtl['mut_event'].apply(lambda x: x.split('_')[0])
        mut_in_meqtl['mut_loc'] = mut_in_meqtl['mut_event'].apply(lambda x: x.split('_')[1])
        mut_in_meqtl['chr'] = mut_in_meqtl['mut_loc'].apply(lambda x: x.split(':')[0])
        mut_in_meqtl['start'] = mut_in_meqtl['mut_loc'].apply(lambda x: int(x.split(':')[1]))
        print(
            f"Number of mutation events with that are meQTL : {len(mut_in_meqtl)}", flush=True
            )
        # get matched samples
        tqdm.pandas(desc="Getting matched samples", miniters=len(mut_in_meqtl)/10)
        mut_in_meqtl['matched_samples'] = mut_in_meqtl.progress_apply(
            lambda mut_event: self._same_age_and_tissue_samples(mut_event['case_submitter_id']), axis = 1
            )
        # print distribution of number of matched samples across mutation events
        mut_in_meqtl['num_matched_samples'] = mut_in_meqtl['matched_samples'].apply(len)
        mut_in_meqtl = mut_in_meqtl.loc[
            mut_in_meqtl['matched_samples'].apply(len) >= self.matched_sample_num
            ]
        print(
            f"Number mutation events being processed after filtering for matched sample number of {self.matched_sample_num}: {len(mut_in_meqtl)}", flush=True
            )
        mut_in_meqtl['is_background'] = False
        mut_in_meqtl['index_event'] = 'self'
        mut_in_meqtl = mut_in_meqtl.iloc[start_num_mut_to_process:end_num_mut_to_process, :]
        print(
            f"Number of mutation events being processed based on start_num_mut_to_process and end_num_mut_to_process: {len(mut_in_meqtl)}"
            )
        mut_in_meqtl.reset_index(drop = True, inplace = True)
        return mut_in_meqtl
        
    
    def look_for_disturbances(
        self, 
        start_num_mut_to_process: int,
        end_num_mut_to_process: int,
        linkage_method: str,
        out_dir: str,
        corr_direction: str,
        comparison_sites_df: pd.DataFrame = None,
        meqtl_db_df: pd.DataFrame = None
        ) -> tuple:
        """
        Driver for the analysis. Finds mutations with VAF >= min_VAF_percentile that have a measured CpG within max_dist of the mutation and then looks for disturbances in the methylation of these CpGs.
        @ start_num_mut_to_process: 
        @ end_num_mut_to_process: 
        @ linkage_method: 'dist' or 'corr' or 'db'
        @ out_dir: directory to write output files to
        @ corr_direction: 'pos' or 'neg' or 'both'
        @ comparison_sites_df: optional dataframe with columns 'mut_event', 'comparison_sites', 'comparison_dists', 'case_submitter_id', 'mut_loc', 'chr', 'start', 'matched_samples', 'is_background', 'index_event'
        @ meqtl_db_df: optional dataframe with columns 'cpg': cpg id's, 'snp': mutation location chr:start, 'beta': beta values for meQTL
        """
        # PHASE 1: choose mutation events and comparison sites
        ######################################################
        # for each mutation, get a list of the CpGs #id in illum_locs that are 
        # within max_dist of the mutation 'start' or top correlated, depending on linkage_method
        if comparison_sites_df is None:
            if linkage_method == 'dist':
                comparison_sites_df = self._get_distance_based_comp_sites(
                    start_num_mut_to_process, end_num_mut_to_process
                    )
            elif linkage_method == 'corr':
                # does just in time correlation of the methylation data in memory
                comparison_sites_df = self._get_correl_based_comp_sites(
                    start_num_mut_to_process, end_num_mut_to_process, corr_direction
                    )
            elif linkage_method == 'db':
                comparison_sites_df = self._get_meqtlDB_based_comp_sites(
                    meqtl_db_df, start_num_mut_to_process, end_num_mut_to_process
                )
            else:
                raise ValueError('linkage_method must be "dist", "correl", or "db"')
            # choose background sites
            if self.num_background_events > 0:
                print("Getting background sites...", flush=True)
                # update comparison_sites_df with background sites
                if linkage_method == 'db':
                    comparison_sites_df = self._choose_background_sites_meqtlDB(
                        comparison_sites_df, meqtl_db_df
                        )
                elif linkage_method == 'dist':
                    comparison_sites_df = self._choose_background_sites_dist(
                        comparison_sites_df
                        )
                else: # corr 
                    comparison_sites_df = self._choose_background_sites_corr(
                        comparison_sites_df, corr_direction
                        )
                print(f"{comparison_sites_df.shape[0]} mutation events total (background and foreground) to be processed", flush=True)
            # convert comparison_sites_df to dask and write to multiple parquet files
            comparison_sites_dd = dask.dataframe.from_pandas(comparison_sites_df, npartitions = 25)
            comparison_sites_fn = os.path.join(
                out_dir, "comparison_sites_{}-{}Muts_{}-linked_qnorm3SD_{}background".format(
                start_num_mut_to_process, end_num_mut_to_process, linkage_method, self.num_background_events
                    )
                )
            comparison_sites_dd.to_parquet(
                comparison_sites_fn, engine = 'pyarrow',
                schema={
                    "comparison_sites": pa.list_(pa.string()),
                    "comparison_dists": pa.list_(pa.int64()),
                    "matched_samples": pa.list_(pa.string())
                    }
                )
            print(f"Wrote comparison sites df to {comparison_sites_fn}", flush=True)
            
        # PHASE 2: compare methylation fractions at comparison sites
        #########################################################
        # for each mutation with nearby measured site, compare the methylation of the nearby measured sites
        # in the mutated sample to the other samples of same age and dataset
        all_metrics_df = self.effect_on_each_site(comparison_sites_df)
        print("got all metrics", flush=True)
        #all_metrics_df.reset_index(inplace=True, drop=True)
        # write out to parquet using dask
        all_metrics_dd = dask.dataframe.from_pandas(all_metrics_df, npartitions = 20)
        all_metrics_fn = os.path.join(
            out_dir, "all_metrics_{}-{}Muts_{}-linked_qnorm3SD_{}background".format(
                start_num_mut_to_process, end_num_mut_to_process, linkage_method, self.num_background_events
                )
            )
        all_metrics_dd.to_parquet(all_metrics_fn, engine = 'pyarrow')
        print(f"Wrote results to {all_metrics_fn}", flush=True)
        return comparison_sites_df, all_metrics_df
