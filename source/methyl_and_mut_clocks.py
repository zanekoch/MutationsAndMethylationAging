import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import get_data
import pandas as pd
import numpy as np
import os
import argparse
from glob import glob
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from xgboost import XGBRegressor
from scipy.sparse import csr_matrix



class methylationClock:
    def __init__(
        self,
        methyl_age_df,
        ):
        """Constructor for methylationClock class"""
        self.methyl_age_df = methyl_age_df
        
    def methylomic_age_prediction(
        self,
        output_col_label:str,
        include_dataset_covariate:bool,
        just_this_cv_num:int = -1,
        specific_features:list = [],
        dataset_subset:list = [],
        include_wg_mean: bool = False,
        out_dir: str = ""
        ):
        """
        Expects self.methyl_age_df to have columns 'age_at_index' and 'tissue'
        @ output_col_label: str, name of column to save predictions to
        @ include_dataset_covariate: bool, whether to include tissue as a one-hot covariate
        @ just_this_cv_num: int, if not -1, only train and predict on this CV fold
        @ specific_features: list, if not empty, only use these features
        @ dataset_subset: list, if not empty, only use these tissues
        @ include_wg_mean: bool, whether to include a mean feature
        @ out_dir: str, if not empty, save models and predictions to this directory
        """
        # print all arguments
        print(
            output_col_label, include_dataset_covariate, just_this_cv_num, specific_features, dataset_subset, include_wg_mean, out_dir
            )
        # create a copy so we don;t mess up original
        methyl_data_mat = self.methyl_age_df.copy(deep = True)
        # optionally subset to specific featyres
        if specific_features == "":
            features_to_use = methyl_data_mat.columns[
                    methyl_data_mat.columns.str.startswith('cg') 
                    | methyl_data_mat.columns.str.startswith('ch')
                    ]
        elif len(specific_features) > 0:
            features_to_use = np.array(specific_features)
        else:
            features_to_use = []
        # optionally add mean features
        if include_wg_mean:
            methyl_data_mat['mean'] = methyl_data_mat[features_to_use].mean(axis = 1)
            features_to_use = np.append(features_to_use, 'mean')
        # optionally subset to only specified tissues
        if len(dataset_subset) > 0:
            methyl_data_mat = methyl_data_mat[methyl_data_mat['dataset'].isin(dataset_subset)]
        # optionally one-hot encode tissue 
        if include_dataset_covariate:
            dataset_dummies = pd.get_dummies(methyl_data_mat['dataset'])
            methyl_data_mat = pd.concat([methyl_data_mat, dataset_dummies], axis = 1)
            features_to_use = np.append(features_to_use, dataset_dummies.columns)
            
        # split into train and test balanced by age using sklearn
        all_cv_predictions = {}
        cv_split = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        stratify_col = pd.qcut(methyl_data_mat['age_at_index'], 10, labels = False)
        # check if n_splits will be greater than the number of members in each class
        if len(np.unique(stratify_col)) < 5:
            # do a normal train test split
            cv_split = KFold(n_splits = 5, shuffle = True, random_state = 0)
            cv = cv_split.split(methyl_data_mat)
        else:
            cv = cv_split.split(methyl_data_mat, stratify_col)
            
        for cv_num, (train_index, test_index) in enumerate(cv):
            print(f"Starting CV {cv_num}...", flush = True)
            # skip if we're only training on CV
            if just_this_cv_num != -1 and cv_num != just_this_cv_num:
                continue
            # split into train and test
            X_train = methyl_data_mat.iloc[train_index][features_to_use]
            X_test = methyl_data_mat.iloc[test_index][features_to_use]
            y_train = methyl_data_mat.iloc[train_index]['age_at_index']
            y_test = methyl_data_mat.iloc[test_index]['age_at_index']
            # convert to sparse matrices
            X_train_sparse = csr_matrix(X_train.values)
            X_test_sparse = csr_matrix(X_test.values)
            xgb = XGBRegressor(
                learning_rate=0.1, verbosity=1, 
                objective='reg:squarederror', n_jobs=-1,
                booster = 'dart' 
                )
            # set xgb feature names to features_to_use
            xgb.feature_names = features_to_use.tolist()
            xgb.fit(X_train_sparse, y_train)
            # predict
            this_cv_predictions = xgb.predict(X_test_sparse)
            # create df with sample name, prediction, and true age
            this_cv_predictions = pd.DataFrame(
                this_cv_predictions, 
                columns = [output_col_label]
                )
            this_cv_predictions['age_at_index'] = y_test.values
            this_cv_predictions['case_submitter_id'] = methyl_data_mat.iloc[test_index].index
            this_cv_predictions.set_index('case_submitter_id', inplace = True)
            this_cv_predictions['cv_num'] = cv_num
            this_cv_predictions['dataset'] = methyl_data_mat.iloc[test_index]['dataset']
            # save predictions
            all_cv_predictions[cv_num] = this_cv_predictions
            # save model if an output directory is specified
            if out_dir != "":
                pickle.dump(xgb, open(f"{out_dir}/methyl_clock_{cv_num}.pkl", "wb"))
                # write out feature names as parquet
                # write out feature names as parquet
                feat_df =  pd.DataFrame(features_to_use)
                feat_df.columns = ['feature_names']
                # convert to string datatypes
                feat_df['feature_names'] = feat_df['feature_names'].astype(str)
                feat_df.to_parquet(f"{out_dir}/methyl_clock_{cv_num}_features.parquet")
        if out_dir != "":
            # save predictions
            if just_this_cv_num != -1:
                all_cv_predictions[just_this_cv_num].to_parquet(f"{out_dir}/methyl_clock_predictions_{just_this_cv_num}CV.parquet")
            else:
                # combine and save all
                all_cv_predictions_df = pd.concat(all_cv_predictions.values())
                print(all_cv_predictions_df)
                # convert column dtypes
                all_cv_predictions_df[output_col_label] = all_cv_predictions_df[output_col_label].astype(float)
                all_cv_predictions_df['age_at_index'] = all_cv_predictions_df['age_at_index'].astype(float)
                all_cv_predictions_df['cv_num'] = all_cv_predictions_df['cv_num'].astype(int)
                all_cv_predictions_df['dataset'] = all_cv_predictions_df['dataset'].astype(str)
                all_cv_predictions_df.to_parquet(f"{out_dir}/methyl_clock_predictions.parquet")
        else:
            # combine into one df
            all_cv_predictions_df = pd.concat(all_cv_predictions.values())
            # if output col label column already exists, drop it
            if output_col_label in self.methyl_age_df.columns:
                self.methyl_age_df.drop(columns=[output_col_label], inplace=True)
            # add to self.mut_burden_df
            self.methyl_age_df = self.methyl_age_df.merge(
                all_cv_predictions_df[[output_col_label]],
                how = 'left',
                left_index = True, right_index = True
                )
            return all_cv_predictions_df
        
     
class mutationClock:   
    def __init__(
        self,
        mut_age_df,
        meta_df,
        illumina_cpg_locs_w_methyl_df, 
        tumor_data = False
        ):
        """Constructor for mutationClock class"""
        self.mut_age_df = mut_age_df
        self.meta_df = meta_df
        self.illumina_cpg_locs_w_methyl_df = illumina_cpg_locs_w_methyl_df
        self.whole_genome_feats = []
        self.tumor_data = tumor_data
    
    def create_mutation_feature_mat(
        self,
        min_coverage:int = -1,
        mutation_subset:list = [],
        one_chr:str = "",
        ):
        """Take the df of mutations and create a sample x mutations matrix
        @ min_coverage: int, minimum coverage to include a mutation
        @ mutation_subset: list, if not empty, only include these kinds of mutations, eg ['C>T', 'C>A']
        @ one_chr: str, if not empty, only include mutations on this chromosome
        """
        if one_chr != "":
            self.mut_age_df = self.mut_age_df.query("chr == @one_chr")
        # optionally filter by coverage
        if min_coverage > 0:
            filtered_mut_w_age_df = self.mut_age_df.query("st_coverage > @min_coverage").reset_index(drop=True)
        else:
            filtered_mut_w_age_df = self.mut_age_df
        # filter by mutation type
        if len(mutation_subset) > 0:
            filtered_mut_w_age_df = filtered_mut_w_age_df.query("mutation.isin(@mutation_subset)").reset_index(drop=True)
        # create a mutation location column
        filtered_mut_w_age_df['mut_loc'] = filtered_mut_w_age_df['chr'].astype(str) + ':' + filtered_mut_w_age_df['start'].astype(str)
        # pivot to wide 
        if not self.tumor_data:
            mut_w_age_wide_df = pd.pivot_table(
                        filtered_mut_w_age_df, index='sample_pair', columns='mut_loc',
                        values='DNA_VAF', fill_value = 0
                        )
        else:
            mut_w_age_wide_df = pd.pivot_table(
                        filtered_mut_w_age_df, index='sample', columns='mut_loc',
                        values='DNA_VAF', fill_value = 0
                        )
        vaf_sum = mut_w_age_wide_df.sum(axis=1)
        # count of nonzero cols in each row
        mut_count = mut_w_age_wide_df.apply(lambda x: x.astype(bool).sum(), axis=1)
        # mean VAF
        mean_vaf = mut_w_age_wide_df.mean(axis=1)
        # mean vaf of nonzero cols
        mean_vaf_nonzero = mut_w_age_wide_df.apply(
            lambda x: x[x > 0].mean(), axis=1
            )
        # add these to the df
        mut_w_age_wide_df['mutation_count'] = mut_count
        mut_w_age_wide_df['vaf_sum'] = vaf_sum
        mut_w_age_wide_df['mean_vaf'] = mean_vaf
        mut_w_age_wide_df['mean_vaf_nonzero'] = mean_vaf_nonzero
        self.whole_genome_feats = self.whole_genome_feats + ['mutation_count', 'vaf_sum', 'mean_vaf', 'mean_vaf_nonzero']
        # convert mutation column in filtered_mut_w_age_df to always start with pyrimidine
        filtered_mut_w_age_df['mutation_mapped'] = filtered_mut_w_age_df['mutation'].map(
            {
                'C>T': 'C>T', 'C>A': 'C>A', 'C>G': 'C>G',
                'G>A': 'C>T', 'G>T': 'C>A', 'G>C': 'C>G',
                'T>C': 'T>C', 'T>A': 'T>A', 'T>G': 'T>G',
                'A>G': 'T>C', 'A>T': 'T>A', 'A>C': 'T>G'
            }
        )
        # also add the count of each mutation type
        for mut_type in ['C>T', 'C>A', 'C>G', 'T>C', 'T>A', 'T>G']:
            if not self.tumor_data:
                grouped_by_pair = filtered_mut_w_age_df.query(
                    "mutation_mapped == @mut_type"
                    ).groupby('sample_pair')
            else:
                grouped_by_pair = filtered_mut_w_age_df.query(
                    "mutation_mapped == @mut_type"
                    ).groupby('sample')
            mut_w_age_wide_df[mut_type] = grouped_by_pair.size()
            mut_w_age_wide_df[mut_type + '_vaf_sum'] = grouped_by_pair['DNA_VAF'].sum()
            mut_w_age_wide_df[mut_type + '_mean_vaf'] = grouped_by_pair['DNA_VAF'].mean()
            mut_w_age_wide_df[mut_type + '_mean_vaf_nonzero'] = grouped_by_pair['DNA_VAF'].apply(
                lambda x: x[x > 0].mean()
                )
            # fill na with 0
            mut_w_age_wide_df[mut_type].fillna(0, inplace=True)
            mut_w_age_wide_df[mut_type + '_vaf_sum'].fillna(0, inplace=True)
            mut_w_age_wide_df[mut_type + '_mean_vaf'].fillna(0, inplace=True)
            mut_w_age_wide_df[mut_type + '_mean_vaf_nonzero'].fillna(0, inplace=True)
            self.whole_genome_feats = self.whole_genome_feats + [mut_type, mut_type + '_vaf_sum', mut_type + '_mean_vaf', mut_type + '_mean_vaf_nonzero']
            
        # add sample column
        if not self.tumor_data:
            mut_w_age_wide_df['sample'] = mut_w_age_wide_df.index.str[:12]
        # merge back in all_meta_df to get age and such
        mut_w_age_wide_df = mut_w_age_wide_df.merge(
            self.meta_df, how = 'left',
            left_on = 'sample', right_index = True
            )
        # reset index, creating sample_pair col
        mut_w_age_wide_df.reset_index(inplace=True, drop=False)
        self.mut_w_age_wide_df = mut_w_age_wide_df
        
    def filter_outliers(
        self,
        filter_using_replicates:bool,
        max_mutation_count:int,
        min_mutation_count:int,
        max_sd_from_mean:int = -1,
        max_above_median: int = -1,
        ):
        """Filter samples with outlier mutation counts
        @ filter_using_replicates: bool, whether to filter using replicates (individuals with multiple blood-normal pairs)
        @ max_mutation_count: int, maximum mutation count to include
        @ min_mutation_count: int, minimum mutation count to include
        @ max_sd_from_mean: int, maximum number of standard deviations above the mean to include
        @ max_above_median: int, maximum number of times above the median to include
        """
        # create mutation_count_by_sample df to use for filterings
        if not self.tumor_data:
            mutation_count_by_sample = self.mut_age_df.groupby('sample_pair').size().reset_index(name='mutation_count')
            mutation_count_by_sample['sample'] = mutation_count_by_sample['sample_pair'].str[:12]
        else:
            mutation_count_by_sample = self.mut_age_df.groupby('sample').size().reset_index(name='mutation_count')
        # merge back in all_meta_df to get age and such
        mutation_count_by_sample = mutation_count_by_sample.merge(
            self.meta_df, how = 'left',
            left_on = 'sample', right_index = True
            )
        
        # drop samples with greater than max_mutation_count or less than min_mutation_count
        start_shape = mutation_count_by_sample.shape
        mutation_count_by_sample.query(
            "not (mutation_count > @max_mutation_count) & not(mutation_count < @min_mutation_count)",
            )
        print(
            "dropped", start_shape[0] - mutation_count_by_sample.shape[0], "rows based on min/max mutation count"
            )
        
        # within each dataset, drop outliers
        def filter_outliers_group(dataset_group):
            start_shape2 = dataset_group.shape[0]
            if max_sd_from_mean != -1:
                dataset_group_mean = dataset_group['mutation_count'].mean()
                dataset_group_std = dataset_group['mutation_count'].std()
                dataset_group = dataset_group.query(
                    "(mutation_count < @dataset_group_mean + @max_sd_from_mean * @dataset_group_std) and (mutation_count > @dataset_group_mean - @max_sd_from_mean * @dataset_group_std)"
                    )
            if max_above_median != -1:
                dataset_group_median = dataset_group['mutation_count'].median()
                dataset_group = dataset_group.query(
                    "mutation_count < @dataset_group_median + @max_above_median * @dataset_group_median"
                    )
            # print dataset name
            print(dataset_group['dataset'].values[0])
            print("dropped", start_shape2 - dataset_group.shape[0], "rows based on dataset-specific mutation count")
            return dataset_group
        
        if max_sd_from_mean != -1 or max_above_median != -1:
            mutation_count_by_sample =mutation_count_by_sample.groupby('dataset').apply(filter_outliers_group)
            # get rid of the dataset index that was created during grouping
            mutation_count_by_sample.reset_index(inplace = True, drop=True)
        # after doing other filtering, possibly do filtering by replicates
        if filter_using_replicates:
            # for each of these samples groups, either choose the one with the highest mutation count if 2 samples, or the middle if >=3
            def resolve_replicates(sample_group):
                if sample_group.shape[0] == 2:
                    sample_group = sample_group.sort_values('mutation_count', ascending = True)
                    return sample_group.iloc[1]
                elif sample_group.shape[0] > 2:
                    sample_group = sample_group.sort_values('mutation_count', ascending = True)
                    return sample_group.iloc[round(sample_group.shape[0] / 2)]
                else:
                    return sample_group.iloc[0]
                
            start_shape3 = mutation_count_by_sample.shape
            mutation_count_by_sample = mutation_count_by_sample.groupby(
                'sample'
                ).apply(resolve_replicates)#.dropna()
            print("dropped", start_shape3[0] - mutation_count_by_sample.shape[0], "samples across all datasets based on replicates")
            # get rid of the sample index that was created during grouping
            self.mutation_count_by_sample = mutation_count_by_sample.reset_index(inplace = False, drop=True)
        # now drop rows of self.mut_age_df that have a sample_pair not in mutation_count_by_sample
        if not self.tumor_data:
            self.mut_age_df = self.mut_age_df[
                self.mut_age_df['sample_pair'].isin(mutation_count_by_sample['sample_pair'])
                ]
        else:
            self.mut_age_df = self.mut_age_df[
                self.mut_age_df['sample'].isin(mutation_count_by_sample['sample'])
                ]
        
    def create_multiindex_and_sparsify(self):
        """
        Splits column names into a MultiIndex and converts the DataFrame to a sparse matrix format.
        """
        # subset to only the mutation columns, those starting with 'chr'
        mut_w_age_wide_mut_only_df = self.mut_w_age_wide_df.loc[
            :, self.mut_w_age_wide_df.columns.str.startswith('chr')
            ]
        # Splitting the column names into a MultiIndex
        split_cols = mut_w_age_wide_mut_only_df.columns.str.split(':', expand=True)
        # convert the second column to an integer
        # set these as the columns
        mut_w_age_wide_mut_only_df.columns = split_cols
        # convert to sparse type
        dtype = pd.SparseDtype(float, fill_value=0)
        mut_w_age_wide_sparse= mut_w_age_wide_mut_only_df.astype(dtype)
        # add level names to the multiindex
        mut_w_age_wide_sparse.columns.names = ['chr', 'start']
        # convert the start level of the MultiIndex to an integer
        mut_w_age_wide_sparse.columns.set_levels(
            mut_w_age_wide_sparse.columns.levels[1].astype(int), level=1, inplace=True
            )
        
        self.mut_w_age_wide_sparse = mut_w_age_wide_sparse
        
    def get_nearby_distance_burden_per_cpg(
        self,
        max_dist: int,
        output_mut_burden_dir: str, 
        chrom: str = None
        ):
        """
        Sum mutations in nearby CpGs for each CpG, outputting as parquet files
        @ param max_dist: maximum distance to consider for nearby CpGs (+/- max_dist)
        @ param output_mut_burden_dir: directory to save output
        @ param chrom: if not None, only do this chromosome
        """
        # Function to sum mutations within max_dist of a given start position for a chromosome
        def sum_nearby_mutations(chrom: str, cpg_start: int, max_dist: int = 5000):
            # Identify the range of start positions to consider
            start_from = cpg_start - max_dist
            start_to = cpg_start + max_dist
        
            # Use the MultiIndex to efficiently select nearby mutations
            mask = (self.mut_w_age_wide_sparse.columns.get_level_values('chr') == chrom) & \
                   (self.mut_w_age_wide_sparse.columns.get_level_values('start') >= start_from) & \
                   (self.mut_w_age_wide_sparse.columns.get_level_values('start') <= start_to)
            selected_columns = self.mut_w_age_wide_sparse.loc[:, mask]
            #print(selected_columns.shape[1], "mutations within", max_dist, "bp of", cpg_start, "on chromosome", chrom, flush=True)
            # Sum across selected columns
            return selected_columns.sum(axis=1)
        
        # make output directory if it doesn't exist
        if output_mut_burden_dir != "":
            if not os.path.exists(output_mut_burden_dir):
                os.makedirs(output_mut_burden_dir)
            
        if chrom is not None:
            chroms = [chrom]
        else:
            chroms = range(1,23)
        # iterate across chromosomes, or just do one
        all_mut_burdens = []
        for chrom in chroms:
            print("Starting chromosome", chrom, flush=True)
            chrom = str(chrom)
            # get cpgs on this chromosome
            this_chr_cpg_locs_df = self.illumina_cpg_locs_w_methyl_df.query(
                "chr == @chrom"
                ).set_index('#id', drop = True)
            # for each of these cpgs, get the sum of mutation counts in the nearby CpGs (within max_dist) 
            dist_mut_burdens_df = this_chr_cpg_locs_df.apply(
                lambda cpg_row: sum_nearby_mutations(
                    'chr' + chrom, cpg_row['start'], max_dist=max_dist
                    ), axis=1
                )
            all_mut_burdens.append(dist_mut_burdens_df)
            # save as parquet
            if output_mut_burden_dir != "":
                dist_mut_burdens_df.to_parquet(
                    f"{output_mut_burden_dir}/mut_burden_{max_dist*2}_nearby_cpgs_chr{chrom}.parquet"
                    )
            print("Done with chromosome", chrom, flush=True)
        self.mut_burden_df = pd.concat(all_mut_burdens)
        
    def read_in_mut_wide(
        self,
        output_mut_burden_dir: str, 
        ):
        """Read in the mut wide dfs per chromosome
        @ param output_mut_burden_dir: directory where output is saved
        """
        # read in all the parquet files
        whole_genome_feats_dfs = []
        for fn in glob(f"{output_mut_burden_dir}/mut_w_age_wide*.parquet"):
            df = pd.read_parquet(fn)
            df = df.loc[:, ~df.columns.str.startswith('chr')]
            df['chrom'] = fn.split('_')[-1].split('.')[0]
            whole_genome_feats_dfs.append(df)
            print(fn, flush=True)
        # concatenate into one df
        self.whole_genome_feats_df = pd.concat(whole_genome_feats_dfs)
        
        # aggregate whole genome features across chromosomes
        # these columns we can just sum
        to_sum_cols = ['mutation_count', 'vaf_sum', 'C>T', 'C>A', 'C>G', 'T>C', 'T>A', 'T>G']
        self.mut_w_age_wide_df = self.whole_genome_feats_df.groupby('sample').agg({
            # sum mutation_count
            'mutation_count': 'sum',
            'vaf_sum': 'sum',
            'C>T':'sum',
            'C>A':'sum', 
            'C>G':'sum',
            'T>C':'sum', 
            'T>A':'sum', 
            'T>G':'sum'
        })
        # these columns we need to take a weighted average
        def weighted_average(values, weights):
            return np.average(values, weights=weights)
        to_weighted_avg_cols = self.whole_genome_feats_df.columns.difference(to_sum_cols)
        # remove 'chrom', 'sample', 'dataset', and 'gender' 
        to_weighted_avg_cols = to_weighted_avg_cols.difference(['chrom', 'sample', 'dataset', 'gender', 'age_at_index'])
        # iterate across columns, getting 
        for col in to_weighted_avg_cols:
            new_weighted_col = self.whole_genome_feats_df.groupby('sample').apply(
                lambda sample_group: weighted_average(sample_group[col], sample_group['mutation_count'])
            )
            self.mut_w_age_wide_df[col] = new_weighted_col
        self.whole_genome_feats = self.mut_w_age_wide_df.columns.to_list()
        
        # merge in self.meta_df
        self.mut_w_age_wide_df = self.mut_w_age_wide_df.merge(
            self.meta_df, how = 'left', left_index= True, right_index = True
            )        
        
    def read_in_nearby_distance_burden_per_cpg(
        self,
        max_dist: int,
        output_mut_burden_dir: str, 
        ):
        """ Read in the mutation burden per CpG from parquet files
        @ param max_dist: maximum distance to consider for nearby CpGs (+/- max_dist)
        @ param output_mut_burden_dir: directory  where output is saved
        """
        """# check if there is a cached version of this
        if os.path.exists(f"{output_mut_burden_dir}/mut_burden_{max_dist*2}_all.parquet"):
            self.mut_burden_df = pd.read_parquet(f"{output_mut_burden_dir}/mut_burden_{max_dist*2}_all.parquet")
            return"""
        # read in all the parquet files
        all_mut_burdens = []
        for fn in glob(f"{output_mut_burden_dir}/*{max_dist*2}*.parquet"):
            all_mut_burdens.append(pd.read_parquet(fn))
            print(fn, flush=True)
        # concatenate into one df
        self.mut_burden_df = pd.concat(all_mut_burdens)
        self.mut_burden_df = self.mut_burden_df.T
        # convet all columns to float
        self.mut_burden_df = self.mut_burden_df.astype(float)
        # merge with certain columns of mut_w_age_wide_df
        self.mut_burden_df = self.mut_burden_df.merge(
            self.mut_w_age_wide_df[self.whole_genome_feats + ['dataset', 'age_at_index']],
            left_index=True, right_index=True, how = 'left'
            )
        print("filling na", flush=True)
        self.mut_burden_df.fillna(0, inplace=True)
        
        # only need to do the below for 50kb windows bc of some weirdness with the parquet files
        """if not self.tumor_data:
            # set sample as index
            self.mut_burden_df.reset_index(inplace=True)
            # get from sample pair
            self.mut_burden_df['sample'] = self.mut_burden_df['index'].str[:12]
            self.mut_burden_df.set_index('sample', inplace=True)
            self.mut_burden_df.drop(columns=['index'], inplace=True)
            # remove last 3 rows, chr, start and Strand
            self.mut_burden_df = self.mut_burden_df.iloc[:-3, :]"""
        """# cache this mutation burden df
        self.mut_burden_df.to_parquet(f"{output_mut_burden_dir}/mut_burden_{max_dist*2}_all.parquet")"""
        
    def mutational_age_prediction(
        self,
        output_col_label:str,
        include_dataset_covariate:bool = True,
        categorical_dataset_feat:bool = False, # cannot use when using sparse matrices
        just_this_cv_num:int = -1,
        specific_features:list = [],
        dataset_subset:list = [],
        include_whole_genome_feats: bool = False,
        include_callable_bp: bool = False,
        stratify_by: str = 'dataset',
        booster = 'gbtree',
        out_dir: str = ""
        ):
        """
        Expects self.methyl_age_df to have columns 'age_at_index' and 'tissue'
        @ output_col_label: str, name of column to save predictions to
        @ include_dataset_covariate: bool, whether to include tissue as a one-hot covariate
        @ just_this_cv_num: int, if not -1, only train and predict on this CV fold
        @ specific_features: list, if not empty, only use these features
        @ dataset_subset: list, if not empty, only use these tissues
        @ include_wgmb: bool, whether to include a wgmb feature
        @ include_callable_bp: bool, whether to include a callable bp feature
        @ stratify_by: str, stratify by 'dataset' or 'age'
        @ out_dir: str, if not empty, save models and predictions to this directory
        """
        # create a copy so we don;t mess up original
        mut_data_mat = self.mut_burden_df.copy(deep = True)
        # optionally subset to certain features
        if specific_features == "":
            features_to_use = mut_data_mat.columns[
                    mut_data_mat.columns.str.startswith('cg') 
                    | mut_data_mat.columns.str.startswith('ch')
                    ]
        elif len(specific_features) > 0:
            # convert to np array
            features_to_use = np.array(specific_features)
        else:
            features_to_use = []
        # optionally add whole genome features
        if include_whole_genome_feats:
            features_to_use = np.append(features_to_use, self.whole_genome_feats)
            # drop callable_bp if it's in the features, so we can control whether to include it
            if 'callable_bp' in features_to_use:
                features_to_use = np.delete(features_to_use, np.where(features_to_use == 'callable_bp'))
        # optionally add callable bp feature
        if include_callable_bp:
            mut_data_mat['mutation_count_per_kb'] = mut_data_mat['mutation_count'] / (mut_data_mat['callable_bp'] * 1000)
            features_to_use = np.append(features_to_use, 'mutation_count_per_kb')
            features_to_use = np.append(features_to_use, 'callable_bp')
        # optionally subset to only specified tissues
        if len(dataset_subset) > 0:
            mut_data_mat = mut_data_mat[mut_data_mat['dataset'].isin(dataset_subset)]
        # optionally one-hot encode tissue 
        if include_dataset_covariate:
            if categorical_dataset_feat:
                mut_data_mat['dataset'] = mut_data_mat['dataset'].astype('category')
                features_to_use = np.append(features_to_use, 'dataset')
            else:
                dataset_dummies = pd.get_dummies(mut_data_mat['dataset'])
                mut_data_mat = pd.concat([mut_data_mat, dataset_dummies], axis = 1)
                features_to_use = np.append(features_to_use, dataset_dummies.columns)
        # split into train and test balanced by dataset using sklearn
        cv_split = StratifiedKFold(n_splits = 5, shuffle = True, random_state =1 )
        if stratify_by == 'age':
            # split age_at_index into 5 bins
            stratify_col = pd.qcut(mut_data_mat['age_at_index'], 10, labels = False)
        elif stratify_by == 'dataset':
            stratify_col = mut_data_mat['dataset']
            
        
        try:
            cv = cv_split.split(mut_data_mat, stratify_col)
            # testing
            for cv_num, (train_index, test_index) in enumerate(cv):
                pass
            cv = cv_split.split(mut_data_mat, stratify_col)

        except:
            print("StratifiedKFold failed, using KFold instead", flush=True)
            cv_split = KFold(n_splits = 5, shuffle = True, random_state = 1)
            try:
                cv = cv_split.split(mut_data_mat)
                # testing
                for cv_num, (train_index, test_index) in enumerate(cv):
                    pass
                cv = cv_split.split(mut_data_mat)
            except:
                print("KFold failed, skipping...")
                return 
        all_cv_predictions = {}
        for cv_num, (train_index, test_index) in enumerate(cv):
            print(f"Starting CV {cv_num}...", flush = True)
            # skip if we're only training on CV
            if just_this_cv_num != -1 and cv_num != just_this_cv_num:
                continue
            # split into train and test
            X_train = mut_data_mat.iloc[train_index][features_to_use]
            X_test = mut_data_mat.iloc[test_index][features_to_use]
            y_train = mut_data_mat.iloc[train_index]['age_at_index']
            y_test = mut_data_mat.iloc[test_index]['age_at_index']
            # convert to sparse matrices
            X_train_sparse = csr_matrix(X_train.values)
            X_test_sparse = csr_matrix(X_test.values)
            # fit model
            if categorical_dataset_feat:
                xgb = XGBRegressor(
                    learning_rate=0.1, verbosity=1, 
                    objective='reg:squarederror', n_jobs=-1, 
                    enable_categorical=True, tree_method = 'hist',
                    booster = booster
                    )
            else:
                xgb = XGBRegressor(
                    learning_rate=0.1, verbosity=1, 
                    objective='reg:squarederror', n_jobs=-1,
                    booster = booster 
                    )
            # set xgb feature names to features_to_use
            xgb.feature_names = features_to_use.tolist()
            xgb.fit(X_train_sparse, y_train)
            # predict
            this_cv_predictions = xgb.predict(X_test_sparse)
            # create df with sample name, prediction, and true age
            this_cv_predictions = pd.DataFrame(
                this_cv_predictions, 
                columns = [output_col_label]
                )
            this_cv_predictions['age_at_index'] = y_test.values
            this_cv_predictions['case_submitter_id'] = mut_data_mat.iloc[test_index].index
            this_cv_predictions.set_index('case_submitter_id', inplace = True)
            this_cv_predictions['cv_num'] = cv_num
            this_cv_predictions['dataset'] = mut_data_mat.iloc[test_index]['dataset']
            # save predictions
            all_cv_predictions[cv_num] = this_cv_predictions
            # save model if an output directory is specified
            if out_dir != "":
                pickle.dump(xgb, open(f"{out_dir}/mutation_clock_{cv_num}.pkl", "wb"))
                # write out feature names as parquet
                feat_df =  pd.DataFrame(features_to_use)
                feat_df.columns = ['feature_names']
                feat_df['feature_names'] = feat_df['feature_names'].astype(str)
                feat_df.to_parquet(f"{out_dir}/mutation_clock_{cv_num}_features.parquet")
        if out_dir != "":
            # save predictions
            if just_this_cv_num != -1:
                all_cv_predictions[just_this_cv_num].to_parquet(f"{out_dir}/mut_clock_predictions_{just_this_cv_num}CV.parquet")
            else:
                # combine and save all
                all_cv_predictions_df = pd.concat(all_cv_predictions.values())
                print(all_cv_predictions_df)
                # convert column dtypes
                all_cv_predictions_df[output_col_label] = all_cv_predictions_df[output_col_label].astype(float)
                all_cv_predictions_df['age_at_index'] = all_cv_predictions_df['age_at_index'].astype(float)
                all_cv_predictions_df['cv_num'] = all_cv_predictions_df['cv_num'].astype(int)
                all_cv_predictions_df['dataset'] = all_cv_predictions_df['dataset'].astype(str)
                all_cv_predictions_df.to_parquet(f"{out_dir}/mut_clock_predictions.parquet")
        else:
            # combine into one df
            all_cv_predictions_df = pd.concat(all_cv_predictions.values())
            # if output col label column already exists, drop it
            if output_col_label in self.mut_burden_df.columns:
                self.mut_burden_df.drop(columns=[output_col_label], inplace=True)
            # add to self.mut_burden_df
            self.mut_burden_df = self.mut_burden_df.merge(
                all_cv_predictions_df[[output_col_label]],
                how = 'left',
                left_index = True, right_index = True
                )
            return all_cv_predictions_df, xgb
         
class combined_mutationClock(mutationClock):
    """
    Takes in two mutationClock objects and combines them, then allowing training of a combined model
    """
    def __init__(
        self, 
        normal_mut_clock: mutationClock,
        tumor_mut_clock: mutationClock
        ):
        self.normal_mut_clock = normal_mut_clock
        self.tumor_mut_clock = tumor_mut_clock
        # add _cancer to index of 
        self.tumor_mut_clock.mut_burden_df.index = self.tumor_mut_clock.mut_burden_df.index + '_cancer'
        self.normal_mut_clock.mut_burden_df.index = self.normal_mut_clock.mut_burden_df.index + '_normal'
        # combine mutation, dropping not-shared columns
        self.normal_mut_clock.mut_burden_df['tissue_type'] = 'normal'
        self.tumor_mut_clock.mut_burden_df['tissue_type'] = 'cancer'
        self.mut_burden_df = pd.concat([normal_mut_clock.mut_burden_df, tumor_mut_clock.mut_burden_df])
        self.mut_burden_df.dropna(how = 'any', axis = 1, inplace = True)
        # use tumor wg feats for whole genome feats, doesn't matter which
        self.whole_genome_feats = tumor_mut_clock.whole_genome_feats
