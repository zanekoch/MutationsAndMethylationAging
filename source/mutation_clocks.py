
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


def create_mutation_features(
    mutation_df: pd.DataFrame,
    mutation_matrix_out_dir: str,
    tissues: list, 
    consoritum: str,
    only_c_to_t: bool = True
    ):
    """
    Given a dataframe of mutations across all samples, create a wide matrix of mutations for each chromosome
    @ param mutation_df: dataframe of mutations across all samples, 
    @ param mutation_matrix_out_dir: directory to save wide mutation matrices
    @ param tissues: list of tissues whose samples should be included in the mutation matrix
    @ param consoritum: consortium to use, either 'tcga' or 'icgc'
    @ return: None
    """
    if not os.path.exists(mutation_matrix_out_dir):
        os.makedirs(mutation_matrix_out_dir)
    if 'mut_loc' not in mutation_df.columns:
        mutation_df['mut_loc'] = mutation_df['chr'].astype(str) + ':' + mutation_df['start'].astype(str)
    if only_c_to_t:
        mutation_df = mutation_df.query("mutation == 'C>T' and dataset in @tissues")
    # iterate across chromosomes, create wide matrices of mutation data
    for chrom in range(1,23):
        chrom = str(chrom)
        mut_w_age_wide_df = pd.pivot_table(
            mutation_df.query("chr == @chrom"), index='case_submitter_id', columns='mut_loc',
            values='DNA_VAF', fill_value = 0
            )
        mut_w_age_wide_df.to_parquet(f"{mutation_matrix_out_dir}/{consoritum}_mut_w_age_wide_df_chr{chrom}.parquet")
        print(chrom)

class mutationBurdenClock:
    
    def __init__(
        self,
        all_methyl_age_df_t: pd.DataFrame,
        mut_burden_dir: str,
        corr_dir: str,
        consoritum: str,
        tissues: list,
        max_dist: int
        ):
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.mut_burden_dir = mut_burden_dir
        self.corr_dir = corr_dir
        self.consoritum = consoritum
        self.tissues = tissues
        self.max_dist = max_dist
        self.all_mut_burden_df = pd.DataFrame()
        self.mut_count_w_pred_results_df = pd.DataFrame()
        
    def load_mut_burden_df(
        self,
        chrom: int
        ):
        dist_mut_burdens_df = pd.read_parquet(
            f"{self.mut_burden_dir}/{self.consoritum}_mut_burden_{int(self.max_dist*2)}_nearby_cpgs_chr{chrom}.parquet"
            )
        return dist_mut_burdens_df.T
    
    def load_all_mut_burden_df(self):
        """
        Load the burden dfs of all chrs and concat them
        """
        all_chr_burdens = []
        for chrom in range(1,23):
            all_chr_burdens.append(self.load_mut_burden_df(chrom))
        # merge all dfs on their index
        all_mut_burden_df = pd.concat(all_chr_burdens, axis=1)
        # fill nan values with 0
        all_mut_burden_df = all_mut_burden_df.fillna(0)
        self.all_mut_burden_df = all_mut_burden_df

    def load_burden_mf_corr(
        self,
        tissue: str,
        corr_type: str
        ):
        """
        Load correlation between mutation burden and methylation fraction
        @ param tissue: tissue to use
        @ param corr_type: correlation type, either 'spearman' or 'pearson'
        @ return: correlation dataframe
        """
        return pd.read_parquet(f"{self.corr_dir}/{tissue}_{corr_type}_corrs.parquet")
    
    def get_mutation_burden_methyl_fract_corr(self):
        # make output directory if it doesn't exist
        if not os.path.exists(self.corr_dir):
            os.makedirs(self.corr_dir)
        # iterate across tissues
        for tissue in self.tissues:
            spearman_corr_df_list = []  
            pearson_corr_df_list = []
            for chrom in range(1,23):
                # read in mutation burden data
                dist_mut_burdens_df = self.load_mut_burden_df(chrom)
                # select corresponding methylation data
                methylation_df = self.all_methyl_age_df_t[dist_mut_burdens_df.columns.tolist() + ['dataset']]
                # select the samples from tissue that have both methylation and mutation burden data
                samples_w_both_data = dist_mut_burdens_df.index.intersection(
                    methylation_df.query("dataset == @tissue").index
                    )
                # subset both matrices to only include samples with both data
                dist_mut_burdens_df = dist_mut_burdens_df.loc[samples_w_both_data]
                methylation_df = methylation_df.loc[samples_w_both_data].drop(columns=['dataset'])
                # get pairwise correlations between mutation burden and methylation
                spearman_corrs = dist_mut_burdens_df.corrwith(methylation_df, method='spearman')
                pearson_corrs = dist_mut_burdens_df.corrwith(methylation_df, method='pearson')
                spearman_corr_df_list.append(spearman_corrs)
                pearson_corr_df_list.append(pearson_corrs)
                print(f"Done with chromosome {chrom} for tissue {tissue}")
            # combine correlations across chromosomes
            spearman_corr_df = pd.concat(spearman_corr_df_list, axis=0).to_frame().rename(columns={0:'spearman'})
            pearson_corr_df = pd.concat(pearson_corr_df_list, axis=0).to_frame().rename(columns={0:'pearson'})
            # save as parquet
            spearman_corr_df.to_parquet(f"{self.corr_dir}/{tissue}_spearman_corrs.parquet")
            pearson_corr_df.to_parquet(f"{self.corr_dir}/{tissue}_pearson_corrs.parquet")
    
    def get_cpgs_w_most_mutations(
        self,
        num_cpgs, 
        skip_first
        ):
        cpg_w_methylation_and_burden = list(
            set(self.all_methyl_age_df_t.columns).intersection(set(self.all_mut_burden_df.columns))
            )
        samples_w_methylation_and_burden = list(
            set(self.all_methyl_age_df_t.index).intersection(set(self.all_mut_burden_df.index))
            )
        burden_df = self.all_mut_burden_df.loc[
            samples_w_methylation_and_burden, cpg_w_methylation_and_burden
            ]
        # get the most freq mutated sites
        site_mut_freq = burden_df.sum().sort_values(ascending=False) / burden_df.shape[0]
        top_mut_freq = site_mut_freq[skip_first:num_cpgs + skip_first].index
        # get the burden across these CpGs in each sample
        mut_count_in_chosen_cpgs = burden_df[top_mut_freq]
        mut_count_in_chosen_cpgs['sum'] = mut_count_in_chosen_cpgs.sum(axis=1)
        # merge with methylation data to get ages and gender
        mut_count_in_chosen_cpgs = mut_count_in_chosen_cpgs.merge(
            self.all_methyl_age_df_t[['age_at_index', 'gender']], left_index=True, right_index=True, how = 'left'
            )
        return mut_count_in_chosen_cpgs
              
    def choose_cpg_and_samples_one_tissue(
        self,
        tissue,
        num_cpgs,
        frac_test_samples = 0.1,
        skip_first = 10,
        output_fn = ""
        ):
        """
        @ burden_df: mutation burden dataframe, count for each sample and each site
        @ all_methyl_age_df_t: methylation dataframe
        @ tissue: tissue to use
        @ num_cpgs: number of CpGs to select
        @ frac_test_samples: fraction of samples to use as test
        @ skip_first: number of CpGs to skip (to avoid CpGs with very high mutation burden, e.g. cancer driver genes)
        """
        if tissue != 'all':
            # select only samples of this tissue with methylation and sites with methylation
            all_methyl_age_df_t = self.all_methyl_age_df_t.query('dataset == @tissue')
        else:
            all_methyl_age_df_t = self.all_methyl_age_df_t
        # get cpgs with methylation and mutation burden
        mut_count_in_chosen_cpgs = self.get_cpgs_w_most_mutations(num_cpgs, skip_first)
        
        if tissue != 'all':
            mut_count_in_chosen_cpgs.sort_values('sum', ascending=False, inplace=True)
            # set the first frac_test_samples% of samples as test
            total_samples = mut_count_in_chosen_cpgs.shape[0]
            num_test_samples = int(total_samples * frac_test_samples)
            mut_count_in_chosen_cpgs['test'] = [1] * num_test_samples + [0] * (total_samples - num_test_samples)
            
            # randomly select another 10% of samples as test, balancing ages
            """age_weights = mut_count_in_chosen_cpgs.query("test == 1")['age_at_index'].value_counts() / num_test_samples
            mut_count_in_chosen_cpgs['weights'] = mut_count_in_chosen_cpgs['age_at_index'].map(age_weights)"""
            non_high_burden_test_samples = mut_count_in_chosen_cpgs.query(
                "test == 0"
                ).sample(num_test_samples).index#, weights = 'weights').index
            # set average burden test samples to 2
            mut_count_in_chosen_cpgs.loc[non_high_burden_test_samples, 'test'] = 2
        else:
            # sort the dataframe
            mut_count_in_chosen_cpgs.sort_values('sum', ascending=False, inplace=True)
            # group by tissue and set the first frac_test_samples% of samples as test
            mut_count_in_chosen_cpgs['test'] = 0
            mut_count_in_chosen_cpgs['weights'] = 0
            for tissue in self.tissues:
                tissue_samples = list(
                    set(all_methyl_age_df_t.query("dataset == @tissue").index
                        ).intersection(set(mut_count_in_chosen_cpgs.index))
                    )
                if len(tissue_samples) == 0:
                    print("no samples in tissue {}".format(tissue))
                    continue
                total_samples = len(tissue_samples)
                num_test_samples = int(total_samples * frac_test_samples)
                ordered_tissue_samples = mut_count_in_chosen_cpgs.loc[tissue_samples, :].sort_values('sum', ascending=False).index
                mut_count_in_chosen_cpgs.loc[ordered_tissue_samples[:num_test_samples], 'test'] = 1
                """# also set another 10% of samples as test, balancing ages
                age_weights = mut_count_in_chosen_cpgs.loc[ordered_tissue_samples[:num_test_samples],'age_at_index'].value_counts() / num_test_samples
                print(age_weights)
                print(mut_count_in_chosen_cpgs.loc[ordered_tissue_samples[:num_test_samples], 'age_at_index'].value_counts())
                print(mut_count_in_chosen_cpgs.loc[ordered_tissue_samples[:num_test_samples]])
                mut_count_in_chosen_cpgs.loc[tissue_samples, 'weights'] = mut_count_in_chosen_cpgs.loc[tissue_samples, 'age_at_index'].map(age_weights)"""
                # select samples using weights
                non_high_burden_test_samples = mut_count_in_chosen_cpgs.loc[tissue_samples, :].query(
                    "test == 0"
                    ).sample(num_test_samples).index
                # set average burden test samples to 2
                mut_count_in_chosen_cpgs.loc[non_high_burden_test_samples, 'test'] = 2
                print("Done with tissue {}".format(tissue), flush=True)
        mut_count_in_chosen_cpgs.to_parquet(output_fn)
        return mut_count_in_chosen_cpgs
    
    def read_mut_count_w_pred_results(
        self, 
        mut_count_w_pred_results_glob,
        five_fold
        ):
        """
        Read in the results of training the mutation burden clocks
        """
        if five_fold:
            mut_count_w_pred_fns = glob(
                os.path.join(mut_count_w_pred_results_dir, "*chosen_cpg_burden_w_pred_5CV.parquet")
                )
        else:
            mut_count_w_pred_fns = glob(
                os.path.join(mut_count_w_pred_results_dir, "*chosen_cpg_burden_w_pred.parquet")
                )
        mut_count_w_pred_results = []
        for mut_count_w_pred_fn in mut_count_w_pred_fns:
            if 'allTissues' in mut_count_w_pred_fn:
                continue
            tissue = os.path.basename(mut_count_w_pred_fn).split("chosen_cpg_burden_w_pred")[0]
            mut_count_w_pred_results_df= pd.read_parquet(mut_count_w_pred_fn)
            mut_count_w_pred_results_df['tissue'] = tissue
            mut_count_w_pred_results.append(mut_count_w_pred_results_df)
        mut_count_w_pred_results_df = pd.concat(mut_count_w_pred_results, axis=0)
        mut_count_w_pred_results_df['residual'] = mut_count_w_pred_results_df['pred_age'] - mut_count_w_pred_results_df['age_at_index']
        mut_count_w_pred_results_df['abs_residual'] = np.abs(mut_count_w_pred_results_df['residual'])
        if five_fold:
            mut_count_w_pred_results_df.set_index('case_submitter_id', inplace=True, drop = True)
        self.mut_count_w_pred_results_df = mut_count_w_pred_results_df
    
    def get_cpgs_from_clock(
        self, 
        mut_count_w_pred_results_dir,
        five_fold: bool
        ):
        """
        Get the cpgs used in each clock
        """
        clock_fns = glob(os.path.join(mut_count_w_pred_results_dir, "*.pkl"))
        clock_cpgs_by_tissue = {}
        clock_coefs_by_tissue = {}
        for clock_fn in clock_fns:
            if five_fold:
                cv_num = os.path.basename(clock_fn).split("_enet_model")[1].split(".pkl")[0]
                if cv_num == '':
                    # skip the non 5CV clocks
                    continue
                tissue = os.path.basename(clock_fn).split("_enet_model")[0]
                clock = pickle.load(open(clock_fn, 'rb'))
                # get the indices of nonzero coefficients
                nonzero_cpgs = clock.coef_.nonzero()[0]
                clock_coefs_by_tissue[tissue + '_' + cv_num + 'CV'] = clock.coef_[nonzero_cpgs]
                # get the corresponding feature names
                nonzero_cpgs = clock.feature_names_in_[nonzero_cpgs]
                clock_cpgs_by_tissue[tissue  + '_' + cv_num + 'CV'] = nonzero_cpgs
            else:
                tissue = os.path.basename(clock_fn).split("_enet_model")[0]
                clock = pickle.load(open(clock_fn, 'rb'))
                # get the indices of nonzero coefficients
                nonzero_cpgs = clock.coef_.nonzero()[0]
                clock_coefs_by_tissue[tissue] = clock.coef_[nonzero_cpgs]
                # get the corresponding feature names
                nonzero_cpgs = clock.feature_names_in_[nonzero_cpgs]
                clock_cpgs_by_tissue[tissue] = nonzero_cpgs
        self.clock_cpgs_by_tissue = clock_cpgs_by_tissue
        self.clock_coefs_by_tissue = clock_coefs_by_tissue
    
    
    def calculate_burdens_of_clock_cpgs_pancan(
        self,
        model_fns_glob,
        all_mut_w_age_df: pd.DataFrame
        ):
        # get the CpGs used in each CV
        model_fns = glob(model_fns_glob)
        model_fns.sort(key = lambda x: int(x[-5]))
        model_each_cv = {i:pd.read_pickle(fn) for i, fn in enumerate(model_fns)}
        self.clock_coefs_by_tissue = {
            i:model_each_cv.coef_[model_each_cv.coef_.nonzero()] for i, model_each_cv in model_each_cv.items()
            }
        self.clock_cpgs_by_tissue = {
            i:model_each_cv.feature_names_in_[model_each_cv.coef_.nonzero()] 
            for i, model_each_cv in model_each_cv.items()
            }
        # calculate the new burden of the clock CpGs
        self.mut_count_w_pred_results_df['clock_burden'] = 0
        for cv_num, this_clock_cpgs in self.clock_cpgs_by_tissue.items():
            samples = self.mut_count_w_pred_results_df.query("cv_number == @cv_num").index.unique().tolist()
            samples_w_burden = list(set(samples).intersection(set(self.all_mut_burden_df.index)))
            # select burdens from all_cpg_burden_df
            self.mut_count_w_pred_results_df.loc[
                (self.mut_count_w_pred_results_df['cv_number'] == cv_num), 'clock_burden'
                ] = self.all_mut_burden_df.loc[
                            samples_w_burden, this_clock_cpgs
                        ].sum(axis=1)
        # log scale
        self.mut_count_w_pred_results_df['log_clock_burden'] = np.log(self.mut_count_w_pred_results_df['clock_burden'] + 1)
        # also add the burden of all cpgs
        self.mut_count_w_pred_results_df['wg_burden'] = 0
        unique_samples = self.mut_count_w_pred_results_df.index.unique().tolist()
        sample_wg_burdens = all_mut_w_age_df['case_submitter_id'].value_counts()
        # create dictionary
        sample_wg_burden_dict = dict(zip(sample_wg_burdens.index, sample_wg_burdens.values))
        # map from index to wg burden using sample_wg_burden_dict
        self.mut_count_w_pred_results_df['wg_burden'] = self.mut_count_w_pred_results_df.index.map(sample_wg_burden_dict)
        self.mut_count_w_pred_results_df['log_wg_burden'] = np.log(self.mut_count_w_pred_results_df['wg_burden'] + 1)
        
    
    def epi_age_prediction_xgb(
        self,
        output_col_label:str,
        include_tissue_covariate:bool,
        target:str = 'age_at_index',
        ):
        """
        Train a xgboost regressor to predict chronological age from CpGs
        """
        from xgboost import XGBRegressor
        from scipy.sparse import csr_matrix
        self.mut_count_w_pred_results_df[output_col_label] = -1
        for cv_num in range(5):
            features_to_use = self.mut_count_w_pred_results_df.columns[
                    self.mut_count_w_pred_results_df.columns.str.startswith('cg') 
                    | self.mut_count_w_pred_results_df.columns.str.startswith('ch')
                ]
            if include_tissue_covariate:
                features_to_use = np.append(features_to_use, 'dataset')
            # get cpgs and training samples
            this_cv_train_samples = self.mut_count_w_pred_results_df.query(
                "cv_number == @cv_num and this_cv_test_sample == False"
                ).index
            this_cv_all_samples = self.mut_count_w_pred_results_df.query(
                "cv_number == @cv_num"
                ).index
            # select mutation burden data for training samples and clock cpgs
            methyl_mat_train = self.all_methyl_age_df_t.loc[
                this_cv_train_samples, features_to_use
                ]
            methyl_mat_all = self.all_methyl_age_df_t.loc[
                this_cv_all_samples, features_to_use
                ]
            # one hot encode tissue if include_tissue_covariate
            if include_tissue_covariate:
                # one hot encode tissue
                tissue_one_hot = pd.get_dummies(
                    methyl_mat_all.loc[this_cv_all_samples, 'dataset']
                    )
                # add tissue to methyl_mat_all
                methyl_mat_all = pd.concat([methyl_mat_all, tissue_one_hot], axis=1)
                # add tissue to methyl_mat_train
                tissue_one_hot = pd.get_dummies(
                    methyl_mat_train.loc[this_cv_train_samples, 'dataset']
                    )
                methyl_mat_train = pd.concat([methyl_mat_train, tissue_one_hot], axis=1)
                # remove old tissue column
                methyl_mat_all.drop(columns=['dataset'], inplace=True)
                methyl_mat_train.drop(columns=['dataset'], inplace=True)
            # convert to sparse matrices
            methyl_mat_train = csr_matrix(methyl_mat_train.values)
            methyl_mat_all = csr_matrix(methyl_mat_all.values)
            # get labels
            label_train = self.all_methyl_age_df_t.loc[
                this_cv_train_samples, target
                ].values
            # train a xgboost regressor
            xgb = XGBRegressor(
                learning_rate=0.1, verbosity=1, 
                objective='reg:squarederror', n_jobs=-1, 
                )
            xgb.fit(methyl_mat_train, label_train)
            # predict ages for test samples
            pred = xgb.predict(methyl_mat_all)
            # select rows of mut_count_w_pred_results_df that have cv_number == cv_num and are in this_cv_all_samples
            self.mut_count_w_pred_results_df.reset_index(inplace=True, drop = False)
            
            rows = self.mut_count_w_pred_results_df.query(
                "cv_number == @cv_num and index in @this_cv_all_samples"
                ).index
            self.mut_count_w_pred_results_df.loc[
                rows, output_col_label
                ] = pred
            # set index back to samples
            self.mut_count_w_pred_results_df.set_index('index', inplace=True, drop=True)
            print(f"Done with CV {cv_num}", flush=True)
        
    def methylomic_age_prediction(
        self,
        output_col_label:str,
        include_tissue_covariate:bool,
        include_entropy:bool,
        just_this_cv_num:int = -1,
        specific_features:list = [],
        tissue_subset:list = [],
        drop_outliers:bool = False,
        include_wg_mean: bool = False,
        model_out_dir: str = ""
        ):
        from xgboost import XGBRegressor
        from scipy.sparse import csr_matrix
        # create output col
        self.methyl_mat_w_pred_results_df[output_col_label] = -1
        for cv_num in range(5):
            if just_this_cv_num != -1:
                if cv_num != just_this_cv_num:
                    continue
            if len(specific_features) > 0:
                features_to_use = specific_features
            else:
                features_to_use = self.methyl_mat_w_pred_results_df.columns[
                        self.methyl_mat_w_pred_results_df.columns.str.startswith('cg') 
                        | self.methyl_mat_w_pred_results_df.columns.str.startswith('ch')
                        ]
            if include_tissue_covariate:
                features_to_use = np.append(features_to_use, 'tissue')
            if include_entropy:
                features_to_use = np.append(features_to_use, 'entropy')
            if include_wg_mean:
                features_to_use = np.append(features_to_use, 'mean')
                
            # make index a column to allow subset tissue selection
            self.methyl_mat_w_pred_results_df.reset_index(inplace=True, drop = False)
            if len(tissue_subset) > 0:
                if drop_outliers == True:
                    this_cv_train_samples = self.methyl_mat_w_pred_results_df.query(
                        "cv_number == @cv_num and this_cv_test_sample == False and tissue in @tissue_subset and is_outlier == False"
                        ).index
                    this_cv_all_samples = self.methyl_mat_w_pred_results_df.query(
                        "cv_number == @cv_num and tissue in @tissue_subset and is_outlier == False"
                        ).index
                else:
                    this_cv_train_samples = self.methyl_mat_w_pred_results_df.query(
                        "cv_number == @cv_num and this_cv_test_sample == False and tissue in @tissue_subset"
                        ).index
                    this_cv_all_samples = self.methyl_mat_w_pred_results_df.query(
                        "cv_number == @cv_num and tissue in @tissue_subset"
                        ).index
            else:
                if drop_outliers == True:
                    this_cv_train_samples = self.methyl_mat_w_pred_results_df.query(
                        "cv_number == @cv_num and this_cv_test_sample == False and is_outlier == False"
                        ).index
                    this_cv_all_samples = self.methyl_mat_w_pred_results_df.query(
                        "cv_number == @cv_num and is_outlier == False"
                        ).index
                else:
                    this_cv_train_samples = self.methyl_mat_w_pred_results_df.query(
                        "cv_number == @cv_num and this_cv_test_sample == False"
                        ).index
                    this_cv_all_samples = self.methyl_mat_w_pred_results_df.query(
                        "cv_number == @cv_num"
                        ).index
            # select mutation burden data for training samples and clock cpgs
            methyl_mat_train = self.methyl_mat_w_pred_results_df.loc[
                this_cv_train_samples, features_to_use
                ]
            methyl_mat_all = self.methyl_mat_w_pred_results_df.loc[
                this_cv_all_samples, features_to_use
                ]
            # one hot encode tissue if include_tissue_covariate
            if include_tissue_covariate:
                # one hot encode tissue
                tissue_one_hot = pd.get_dummies(
                    self.methyl_mat_w_pred_results_df.loc[this_cv_all_samples, 'tissue']
                    )
                # add tissue to methyl_mat_all
                methyl_mat_all = pd.concat([methyl_mat_all, tissue_one_hot], axis=1)
                # add tissue to methyl_mat_train
                tissue_one_hot = pd.get_dummies(
                    self.methyl_mat_w_pred_results_df.loc[this_cv_train_samples, 'tissue']
                    )
                methyl_mat_train = pd.concat([methyl_mat_train, tissue_one_hot], axis=1)
                # remove old tissue column
                methyl_mat_all.drop(columns=['tissue'], inplace=True)
                methyl_mat_train.drop(columns=['tissue'], inplace=True)
            # get labels
            label_train = self.methyl_mat_w_pred_results_df.loc[
                this_cv_train_samples, 'age_at_index'
                ].values
            
            # do model training
            xgb = XGBRegressor(
                learning_rate=0.1, verbosity=1, 
                objective='reg:squarederror', n_jobs=-1, 
                )
            xgb.fit(methyl_mat_train, label_train)
            #predict ages for test samples
            pred = xgb.predict(methyl_mat_all) 
            
            """# grid search
            xgb = XGBRegressor(
                learning_rate=0.1, verbosity=1, 
                objective='reg:squarederror', n_jobs=-1, 
                )
            ## grid search
            param_grid = {
                #'n_estimators': range(50, 750, 200), # 10
                'max_depth': [2, 6, 10], 
                #'min_child_weight': range(1, 10, 2), # 3
                #'gamma': np.linspace(0, 0.5, 50),
                #'subsample': np.linspace(0.5, 1, 5),
                #'colsample_bytree': np.linspace(0.5, 1, 5),
                'reg_alpha': [.5, 1], # 3
                'reg_lambda': [.5, 1] # 3
            }
            from sklearn.model_selection import GridSearchCV
            # grid search
            grid_search = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                n_jobs=1,
                cv=5,
                verbose=2
            )
            grid_search.fit(methyl_mat_train, label_train)
            xgb = grid_search.best_estimator_
            # print best choice of parameters
            print(grid_search.best_params_)
            pred = xgb.predict(methyl_mat_all)"""
            # add to the dataframe
            self.methyl_mat_w_pred_results_df.loc[
                this_cv_all_samples, output_col_label
                ] = pred
            # if model_out_dir is not empty, save the model
            if model_out_dir != "":
                pickle.dump(
                    xgb, open(os.path.join(model_out_dir, f"xgb_methylClock_CV{cv_num}.pkl"), 'wb')
                    )
            print(f"Done with CV {cv_num}", flush=True)
            self.methyl_mat_w_pred_results_df.set_index(
                'case_submitter_id', inplace = True
                )
            if just_this_cv_num != -1:
                return xgb
    
    def mutational_age_prediction(
        self,
        target:str,
        output_col_label:str,
        include_tissue_covariate:bool,
        include_wg_burden_covariate:bool,
        include_entropy:bool = False,
        include_log_wg_burden_covariate:bool = False,
        just_clock_cpgs:bool = True,
        log_scale_feats:bool = False,
        log_scale_labels:bool = False,
        specific_features:list = [],
        model_out_dir:str = "",
        use_enet:bool = False,
        just_this_cv_num:int = -1,
        tissue_subset:list = [],
        drop_outliers:bool = False,
        rank_order_feats:bool = False,
        ):
        """
        Train a xgboost regressor to predict chronological age from mutation burden
        around clock CpGs
        """
        from xgboost import XGBRegressor
        from scipy.sparse import csr_matrix
        # assert that not just_clock_cpgs true and specific_features is not empty
        if just_clock_cpgs:
            assert len(specific_features) == 0
        # create output col
        self.mut_count_w_pred_results_df[output_col_label] = -1
        
        #for cv_num, this_cv_clock_cpgs in self.clock_cpgs_by_tissue.items():
        for cv_num in range(5):
            if just_this_cv_num != -1:
                if cv_num != just_this_cv_num:
                    continue
            if just_clock_cpgs:
                sys.exit("just_clock_cpgs not implemented yet")
                if include_tissue_covariate:
                    # this_cv_clock_cpgs is a numpy array so add 'tissue' by 
                    this_cv_clock_cpgs = np.append(this_cv_clock_cpgs, 'tissue')
                if include_wg_burden_covariate:
                    this_cv_clock_cpgs = np.append(this_cv_clock_cpgs, 'wg_burden')
                # get cpgs and training samples
                features_to_use = list(
                    set(this_cv_clock_cpgs).intersection(set(self.mut_count_w_pred_results_df.columns))
                    )
            else:
                if len(specific_features) > 0:
                    features_to_use = specific_features
                else:
                    features_to_use = self.mut_count_w_pred_results_df.columns[
                        self.mut_count_w_pred_results_df.columns.str.startswith('cg') 
                        | self.mut_count_w_pred_results_df.columns.str.startswith('ch')
                        ]
                if include_tissue_covariate:
                    features_to_use = np.append(features_to_use, 'tissue')
                if include_wg_burden_covariate:
                    features_to_use = np.append(features_to_use, 'wg_burden')
                if include_log_wg_burden_covariate:
                    features_to_use = np.append(features_to_use, 'wg_burden_log')
                    features_to_use = np.append(features_to_use, 'mean_vaf')
                    features_to_use = np.append(features_to_use, 'std_vaf')
                    features_to_use = np.append(features_to_use, 'median_vaf')
                    features_to_use = np.append(features_to_use, 'max_vaf')
                    import itertools
                    l = ['mut_type_'+''.join(x) for x in itertools.product('ATCG', repeat = 3)]
                    # add mut_type features
                    features_to_use = np.append(features_to_use, l)
                if include_entropy:
                    features_to_use = np.append(features_to_use, 'entropy_vaf')
                    
            
            self.mut_count_w_pred_results_df.reset_index(inplace=True, drop = False)
            if len(tissue_subset) > 0:
                if drop_outliers == True:
                    this_cv_train_samples = self.mut_count_w_pred_results_df.query(
                        "cv_number == @cv_num and this_cv_test_sample == False and tissue in @tissue_subset and is_outlier == False"
                        ).index
                    this_cv_all_samples = self.mut_count_w_pred_results_df.query(
                        "cv_number == @cv_num and tissue in @tissue_subset and is_outlier == False"
                        ).index
                else:
                    this_cv_train_samples = self.mut_count_w_pred_results_df.query(
                        "cv_number == @cv_num and this_cv_test_sample == False and tissue in @tissue_subset"
                        ).index
                    this_cv_all_samples = self.mut_count_w_pred_results_df.query(
                        "cv_number == @cv_num and tissue in @tissue_subset"
                        ).index
            else:
                if drop_outliers == True:
                    this_cv_train_samples = self.mut_count_w_pred_results_df.query(
                        "cv_number == @cv_num and this_cv_test_sample == False and is_outlier == False"
                        ).index
                    this_cv_all_samples = self.mut_count_w_pred_results_df.query(
                        "cv_number == @cv_num and is_outlier == False"
                        ).index
                else:
                    this_cv_train_samples = self.mut_count_w_pred_results_df.query(
                        "cv_number == @cv_num and this_cv_test_sample == False"
                        ).index
                    this_cv_all_samples = self.mut_count_w_pred_results_df.query(
                        "cv_number == @cv_num"
                        ).index
            
            # select mutation burden data for training samples and clock cpgs
            mut_mat_train = self.mut_count_w_pred_results_df.loc[
                this_cv_train_samples, features_to_use
                ]
            mut_mat_all = self.mut_count_w_pred_results_df.loc[
                this_cv_all_samples, features_to_use
                ]
            # log scale numeric columns
            if log_scale_feats:
                if include_tissue_covariate:
                    # take off tissue column, log scale, add back tissue column
                    tissue_col = mut_mat_train['tissue']
                    mut_mat_train.drop(columns=['tissue'], inplace=True)
                    mut_mat_train = np.log(mut_mat_train + 1)
                    mut_mat_train['tissue'] = tissue_col
                    tissue_col = mut_mat_all['tissue']
                    mut_mat_all.drop(columns=['tissue'], inplace=True)
                    mut_mat_all = np.log(mut_mat_all + 1)
                    mut_mat_all['tissue'] = tissue_col
                else:
                    mut_mat_train = np.log(mut_mat_train + 1)
                    mut_mat_all = np.log(mut_mat_all + 1)

            if rank_order_feats:
                # rank order features with respect to training set, converting the test set to the same scale

                mut_mat_train = mut_mat_train.groupby('tissue').apply(
                    lambda x: x.rank()
                    )
                    
                
            # one hot encode tissue if include_tissue_covariate
            if include_tissue_covariate:
                # one hot encode tissue
                tissue_one_hot = pd.get_dummies(
                    self.mut_count_w_pred_results_df.loc[this_cv_all_samples, 'tissue']
                    )
                # add tissue to mut_mat_all
                mut_mat_all = pd.concat([mut_mat_all, tissue_one_hot], axis=1)
                # add tissue to mut_mat_train
                tissue_one_hot = pd.get_dummies(
                    self.mut_count_w_pred_results_df.loc[this_cv_train_samples, 'tissue']
                    )
                mut_mat_train = pd.concat([mut_mat_train, tissue_one_hot], axis=1)
                # remove old tissue column
                mut_mat_all.drop(columns=['tissue'], inplace=True)
                mut_mat_train.drop(columns=['tissue'], inplace=True)

            # get labels
            label_train = self.mut_count_w_pred_results_df.loc[
                this_cv_train_samples, target
                ].values
            if log_scale_labels:
                label_train = np.log(label_train + 1)
            if not use_enet:
                
                # not grid search
                xgb = XGBRegressor(
                    learning_rate=0.1, verbosity=1, 
                    objective='reg:squarederror', n_jobs=-1, 
                    )
                xgb.fit(mut_mat_train, label_train)
                #predict ages for test samples
                pred = xgb.predict(mut_mat_all) 
                """
                # grid search
                xgb = XGBRegressor(
                    learning_rate=0.1, verbosity=1, 
                    objective='reg:squarederror', n_jobs=-1, 
                    )
                ## grid search
                param_grid = {
                    #'n_estimators': range(50, 750, 200), # 10
                    'max_depth': [2, 6, 10], 
                    #'min_child_weight': range(1, 10, 2), # 3
                    #'gamma': np.linspace(0, 0.5, 50),
                    #'subsample': np.linspace(0.5, 1, 5),
                    #'colsample_bytree': np.linspace(0.5, 1, 5),
                    'reg_alpha': [.5, 1], # 3
                    'reg_lambda': [.5, 1] # 3
                }
                from sklearn.model_selection import GridSearchCV
                # grid search
                grid_search = GridSearchCV(
                    estimator=xgb,
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    n_jobs=1,
                    cv=5,
                    verbose=2
                )
                grid_search.fit(mut_mat_train, label_train)
                xgb = grid_search.best_estimator_
                # print best choice of parameters
                print(grid_search.best_params_)
                pred = xgb.predict(mut_mat_all)
                # transform back to normal scale
                if log_scale_labels:
                    pred = np.exp(pred) - 1
                """
            else:
                """enet = linear_model.ElasticNetCV(
                    cv = 5, n_jobs = -1, max_iter = 10000,
                    verbose = 1, tol = 1e-3, l1_ratio=[.5, .9, .99], selection='random'
                    )"""
                enet = linear_model.ElasticNet(
                    alpha = 1, l1_ratio = 0.5, max_iter = 10000, tol = 1e-3, selection='random'
                    )
                enet.fit(mut_mat_train, label_train)
                pred = enet.predict(mut_mat_all)
                # transform back to normal scale
                if log_scale_labels:
                    pred = np.exp(pred) - 1
            # add to the dataframe
            self.mut_count_w_pred_results_df.loc[
                this_cv_all_samples, output_col_label
                ] = pred
            # if model_out_dir is not empty, save the model
            if model_out_dir != "":
                if not use_enet:
                    pickle.dump(xgb, open(os.path.join(model_out_dir, f"xgb_mutClock_mutSeqs_CV{cv_num}.pkl"), 'wb'))
                else:
                    pickle.dump(enet, open(os.path.join(model_out_dir, f"BRCA_CESC_COAD_LGG_LIHC_THCA_UCEC_mutation_enet_model{cv_num}.pkl"), 'wb'))

            print(f"Done with CV {cv_num}", flush=True)
            # set index back
            self.mut_count_w_pred_results_df.set_index(
                'case_submitter_id', inplace = True
                )
            if just_this_cv_num != -1:
                if not use_enet:
                    return xgb
                else:
                    return enet
    
    
    def calculate_burdens_of_clock_cpgs(
        self, 
        mut_count_w_pred_results_dir,
        five_fold: bool, 
        all_mut_w_age_df: pd.DataFrame
        ):
        """
        Add the burdens of the clock CpGs to the self.mut_count_w_pred_results_df
        """
        # get the cpgs used in each clock
        self.get_cpgs_from_clock(
            mut_count_w_pred_results_dir, five_fold
            )
        # check if already pre computed
        if os.path.exists(os.path.join(mut_count_w_pred_results_dir, "all_mut_count_w_pred_results_df.parquet")):
            # read in
            self.mut_count_w_pred_results_df = pd.read_parquet(
                os.path.join(mut_count_w_pred_results_dir, "all_mut_count_w_pred_results_df.parquet")
                )
            print("read in all_mut_count_w_pred_results_df.parquet from file")
            return
        # and the results of training the clocks
        if self.mut_count_w_pred_results_df.shape[0] == 0:
            self.read_mut_count_w_pred_results(
                mut_count_w_pred_results_dir, five_fold
                )
        # calculate the new burden of the clock CpGs
        clock_burdens = []
        weighted_clock_burdens = []
        abs_weighted_clock_burdens = []
        self.mut_count_w_pred_results_df['clock_burden'] = 0
        self.mut_count_w_pred_results_df['weighted_clock_burden'] = 0
        self.mut_count_w_pred_results_df['abs_weighted_clock_burden'] = 0
        # iterating across tissues, or tissue-cv combinations
        for tissue in self.clock_cpgs_by_tissue.keys():
            # get the samples used in this clock
            if tissue == 'all':
                this_tissue_samples = self.mut_count_w_pred_results_df.index.to_list() 
            elif five_fold:
                cv_num = int(tissue.split('_')[1].split('CV')[0])
                tissue_no_cv = tissue.split('_')[0]
                this_tissue_samples =  self.mut_count_w_pred_results_df.query(
                    "tissue == @tissue_no_cv and cv_number == @cv_num"
                    ).index.to_list()
            else:
                this_tissue_samples = self.mut_count_w_pred_results_df.query("tissue == @tissue").index.to_list()
            # get the cpgs used in this clock
            this_clock_cpgs = self.clock_cpgs_by_tissue[tissue]
            # get clock cpgs with calculated burdens 
            this_clock_cpgs = list(
                set(this_clock_cpgs).intersection(set(self.mut_count_w_pred_results_df.columns))
                )
            if five_fold:
                mut_count_this_tissue_and_clock = self.mut_count_w_pred_results_df.query(
                    "cv_number == @cv_num"
                    ).loc[this_tissue_samples, this_clock_cpgs]
            else:
                mut_count_this_tissue_and_clock = self.mut_count_w_pred_results_df.loc[
                    this_tissue_samples, this_clock_cpgs
                    ]
            # select burdens from all_cpg_burden_df
            this_tissue_clock_burden = mut_count_this_tissue_and_clock.sum(axis=1)
            clock_burdens.append(this_tissue_clock_burden)
            # calculate clock burden, where each cpg is weighted by its coefficient
            this_tissue_weighted_clock_burden = mut_count_this_tissue_and_clock.multiply(
                self.clock_coefs_by_tissue[tissue], axis=1
                ).sum(axis=1)
            weighted_clock_burdens.append(this_tissue_weighted_clock_burden)
            this_tissue_abs_weighted_clock_burden = mut_count_this_tissue_and_clock.multiply(
                np.abs(self.clock_coefs_by_tissue[tissue]), axis=1
                ).sum(axis=1)
            abs_weighted_clock_burdens.append(this_tissue_abs_weighted_clock_burden)
            # add these to the mut_count_w_pred_results_df
            if five_fold:
                self.mut_count_w_pred_results_df.loc[
                    (self.mut_count_w_pred_results_df['cv_number'] == cv_num) \
                    & (self.mut_count_w_pred_results_df['tissue'] == tissue_no_cv), 'clock_burden'
                    ] = this_tissue_clock_burden.values
                self.mut_count_w_pred_results_df.loc[
                    (self.mut_count_w_pred_results_df['cv_number'] == cv_num) \
                    & (self.mut_count_w_pred_results_df['tissue'] == tissue_no_cv), 'weighted_clock_burden'
                    ] = this_tissue_weighted_clock_burden.values
                self.mut_count_w_pred_results_df.loc[
                    (self.mut_count_w_pred_results_df['cv_number'] == cv_num) \
                    & (self.mut_count_w_pred_results_df['tissue'] == tissue_no_cv), 'abs_weighted_clock_burden'
                    ] = this_tissue_abs_weighted_clock_burden.values
            else:
                self.mut_count_w_pred_results_df.loc[
                    this_tissue_samples, 'clock_burden'
                    ] = this_tissue_clock_burden
                self.mut_count_w_pred_results_df.loc[
                    this_tissue_samples, 'weighted_clock_burden'
                    ] = this_tissue_weighted_clock_burden
                self.mut_count_w_pred_results_df.loc[
                    this_tissue_samples, 'abs_weighted_clock_burden'
                    ] = this_tissue_abs_weighted_clock_burden
            print(f"Done with tissue {tissue}")
            
        clock_burdens = pd.concat(clock_burdens, axis=0)
        weighted_clock_burdens = pd.concat(weighted_clock_burdens, axis=0)
        abs_weighted_clock_burdens = pd.concat(abs_weighted_clock_burdens, axis=0)
        # add to the mut_count_w_pred_results_df
        self.mut_count_w_pred_results_df['clock_burden'] = 0
        self.mut_count_w_pred_results_df.loc[clock_burdens.index, 'clock_burden'] = clock_burdens.values
        # also add the weighted burden of the clock cpgs
        self.mut_count_w_pred_results_df['weighted_clock_burden'] = 0
        self.mut_count_w_pred_results_df.loc[weighted_clock_burdens.index, 'weighted_clock_burden'] = weighted_clock_burdens.values
        self.mut_count_w_pred_results_df['abs_weighted_clock_burden'] = 0
        self.mut_count_w_pred_results_df.loc[abs_weighted_clock_burdens.index, 'abs_weighted_clock_burden'] = abs_weighted_clock_burdens.values
        # also add the burden of all cpgs
        self.mut_count_w_pred_results_df['wg_burden'] = 0
        unique_samples = self.mut_count_w_pred_results_df.index.unique().tolist()
        sample_wg_burdens = all_mut_w_age_df['case_submitter_id'].value_counts()
        # create dictionary
        sample_wg_burden_dict = dict(zip(sample_wg_burdens.index, sample_wg_burdens.values))
        # map from index to wg burden using sample_wg_burden_dict
        self.mut_count_w_pred_results_df['wg_burden'] = self.mut_count_w_pred_results_df.index.map(sample_wg_burden_dict)
        self.mut_count_w_pred_results_df['log_wg_burden'] = np.log(self.mut_count_w_pred_results_df['wg_burden'] + 1)
        # write out 
        # select columns that do not start with cg or ch
        self.mut_count_w_pred_results_df[self.mut_count_w_pred_results_df.columns[
            ~self.mut_count_w_pred_results_df.columns.str.startswith('cg') & ~self.mut_count_w_pred_results_df.columns.str.startswith('ch')
            ]].to_parquet(
                    os.path.join(mut_count_w_pred_results_dir, "all_mut_count_w_pred_results_df.parquet")
                    )
        

    def random_selections_of_clock_burdens(
        self,
        n: int, # number of random selections for each tissue-cv combination
        num_top_cpgs: int
        ):
        """
        Exluding clock CpGs, for each tissue choose n sets of the same number of CpGs as in clock (from CpGs with at least 1 mutation near them) and calculate the clock burden
        """
        from random import sample
        # create empty columns for each random selection
        for i in range(n):
            self.mut_count_w_pred_results_df[f'random_{i}'] = 0
            
        mut_count_in_chosen_cpgs = self.get_cpgs_w_most_mutations(num_top_cpgs, skip_first=10)
        cpgs_to_choose_from = mut_count_in_chosen_cpgs.columns.to_list()
        # drop entries ['sum', 'age_at_index', 'gender'] from cpgs_to_choose_from
        cpgs_to_choose_from.remove('sum')
        cpgs_to_choose_from.remove('age_at_index')
        cpgs_to_choose_from.remove('gender')
        # iterate across tissue-cv combinations
        for tissue in self.clock_cpgs_by_tissue.keys():
            cv_num = int(tissue.split('_')[1].split('CV')[0])
            tissue_no_cv = tissue.split('_')[0]
            # skip if this tissue is not in the well predicted tissues
            """if tissue_no_cv not in self.well_pred_tissues:
                continue"""
            # get the samples from this tissue
            this_tissue_samples =  self.mut_count_w_pred_results_df.query(
                "tissue == @tissue_no_cv and cv_number == @cv_num"
                ).index.to_list()
            # select this tissue's samples
            this_tissue_mut_burdens = self.all_mut_burden_df.loc[this_tissue_samples]        
            # get the CpGs used in this clock
            this_clock_cpgs = self.clock_cpgs_by_tissue[tissue]
            # and allow to choose from all CpGs except these
            not_this_clock_cpgs = list(
                set(cpgs_to_choose_from).difference(set(this_clock_cpgs))
                )
            # n times, randomly select len(this_clock_cpgs) CpGs from not_this_clock_cpgs
            try:
                random_cpgs = [sample(not_this_clock_cpgs, len(this_clock_cpgs)) for _ in range(n)]
            except:
                print(tissue)
                print(len(this_clock_cpgs))
                print(len(not_this_clock_cpgs))
                print(this_tissue_mut_burdens)
            # get sums of these random CpGs
            random_burdens = [this_tissue_mut_burdens[rand_cpgs].sum(axis=1) for rand_cpgs in random_cpgs]
            # add each of these as a column to all_mut_burden_in_well_pred_tissues
            for i in range(n):
                self.mut_count_w_pred_results_df.loc[
                    (self.mut_count_w_pred_results_df['cv_number'] == cv_num) \
                    & (self.mut_count_w_pred_results_df['tissue'] == tissue_no_cv), f'random_{i}'
                    ] = random_burdens[i]
                self.mut_count_w_pred_results_df.loc[
                    (self.mut_count_w_pred_results_df['cv_number'] == cv_num) \
                    & (self.mut_count_w_pred_results_df['tissue'] == tissue_no_cv), f'log_random_{i}'
                    ] = np.log(1 + random_burdens[i])
        
    def select_well_pred_tissues(
        self, 
        mae_cutoff, 
        pearson_cutoff,
        top_X_tissues_by_med_wg_burden = None
        ):
        fig, ax = plt.subplots(figsize=(6, 4))
        # plot correlation vs MAE
        corr_by_tissue = self.mut_count_w_pred_results_df.query(
            "this_cv_test_sample == True"
            ).groupby(['tissue'])[['pred_age', 'age_at_index']].corr(method = 'pearson').reset_index().iloc[::2, [0,1,3]]
        corr_by_tissue.rename(columns = {'age_at_index': 'Pearson correlation'}, inplace = True)
        # merge with MAE    
        mae_by_tissue = self.mut_count_w_pred_results_df.query(
            "this_cv_test_sample == True"
            ).groupby('tissue')['abs_residual'].mean().reset_index().rename(columns = {'abs_residual': 'MAE'})
        self.perf_by_tissue = corr_by_tissue.merge(mae_by_tissue, on = 'tissue')
        # filter to only include tissues with high median burden
        if top_X_tissues_by_med_wg_burden is not None:
            burden_by_tissue = self.mut_count_w_pred_results_df.groupby('tissue')['wg_burden'].describe().sort_values('50%', ascending = False)
            top_burden_by_tissue = burden_by_tissue.head(top_X_tissues_by_med_wg_burden).index.tolist()
            # add a column for if the tissue is in the top top_X_tissues_by_med_wg_burden by median burden
            self.perf_by_tissue['top_ten_burden'] = self.perf_by_tissue['tissue'].isin(top_burden_by_tissue)
            # plot
            sns.scatterplot(
                data = self.perf_by_tissue, x = 'Pearson correlation', y = 'MAE', hue = 'top_ten_burden', legend = False,
                ax = ax
            )
        else:
            top_burden_by_tissue = self.perf_by_tissue['tissue'].unique().tolist()
            # plot
            sns.scatterplot(
                data = self.perf_by_tissue, x = 'Pearson correlation', y = 'MAE', hue = 'tissue', 
                ax = ax
            )
        plt.xlabel('Pearson correlation (predicted vs. actual age)')
        plt.ylabel('Mean absolute error ((predicted vs. actual age, years)')  
        for line in range(0,self.perf_by_tissue.shape[0]):
            plt.text(
                self.perf_by_tissue['Pearson correlation'][line]+0.01, 
                self.perf_by_tissue['MAE'][line], 
                self.perf_by_tissue['tissue'][line], 
                horizontalalignment='left', 
                size='small', 
                color='black'
            )
        # make x and y axies go from 0 to 1
        plt.xlim(0,1)
        plt.ylim(0,max(self.perf_by_tissue['MAE'].max(), 15))
        # draw vertical line at .5 and horizal at 9
        plt.axvline(x = pearson_cutoff, color = 'black', linestyle = '--')
        plt.axhline(y = mae_cutoff, color = 'black', linestyle = '--')
        
        
        # select the tissues that have a correlation > .5 and MAE < 9
        self.well_pred_tissues = self.perf_by_tissue.query(
            "MAE < @mae_cutoff & `Pearson correlation` > @pearson_cutoff & tissue in @top_burden_by_tissue"
            )['tissue'].values.tolist()
        
    def plot_burden_vs_residual(
        self,
        bin_edges = [],
        test_groups = [],
        burden_type = 'clock_burden'
        ):
        # make labels in the format [r'$10^0 - 10^1$', r'$10^1 - 10^2$', r'$10^2 - 10^3$', r'$10^3 - 10^4$'] from the bin edges
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # age accel by mutation burden
        """self.mut_count_w_pred_results_df[f'{burden_type}_bin'] = pd.qcut(
            self.mut_count_w_pred_results_df.query(
                "test in @test_groups and tissue in @self.well_pred_tissues"
                )[burden_type], q = bin_edges,
            )
        sns.swarmplot(
            data = self.mut_count_w_pred_results_df.query(
                "test in @test_groups and tissue in @self.well_pred_tissues"
                ),
            x = f'{burden_type}_bin', y = 'residual', hue = 'tissue', ax = ax
            )
        sns.violinplot(
            data = self.mut_count_w_pred_results_df.query(
                "test in @test_groups and tissue in @self.well_pred_tissues"
                ),
            x = f'{burden_type}_bin', y = 'residual', palette='Reds', ax = ax
            )"""
        self.mut_count_w_pred_results_df[f'{burden_type}_bin'] = pd.qcut(
        self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
                )[burden_type], q = bin_edges, duplicates = 'drop'
            )
        sns.violinplot(
            data = self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
                ),
            x = f'{burden_type}_bin', y = 'residual', palette='Reds', ax = ax
            )
            
            
        # agle x labels
        _  = plt.setp(ax.get_xticklabels(), rotation=45, ha = 'right')
        
    def get_burden_residual_corr_within_age_groups(
        self,
        age_group_size = 5,
        test_groups = [],
        corr_method = 'spearman'
        ):
        fig, ax = plt.subplots(figsize=(6, 4))
        # bin ages into 5 year bins
        self.mut_count_w_pred_results_df['age_bin'] = pd.cut(
            self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
                )['age_at_index'], bins = [x for x in range(0, 100, age_group_size)],
            labels = [f'{i} - {i+age_group_size}' for i in range(0, 100 - age_group_size, age_group_size)]
        )
        corr_df = self.mut_count_w_pred_results_df.query(
            "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
            ).groupby(['age_bin'])[['clock_burden', 'wg_burden', 'weighted_clock_burden','abs_weighted_clock_burden', 'residual','age_at_index','pred_age']].corr(method = corr_method)
        # drop rows with all nan values
        corr_df = corr_df.dropna(axis=0, how='all')
        # melt and only keep the correlations between burden and residual
        corr_df = corr_df.reset_index().query("level_1 == 'residual'").melt(
            id_vars = 'age_bin', value_vars = ['wg_burden', 'clock_burden', 'weighted_clock_burden', 'abs_weighted_clock_burden', 'age_at_index']
            ).rename(columns = {'variable': 'burden_type', 'value': 'correlation'})
        # plot
        sns.pointplot(
            data = corr_df, x = 'age_bin', y = 'correlation', hue = 'burden_type',
            ax = ax
        )
        plt.ylabel("Correlation with residual")
        
        # angle x labels
        _  = plt.setp(ax.get_xticklabels(), rotation=45)
        return corr_df
    
    def get_burden_residual_corr_within_wgmb_groups(
        self,
        num_bins = 10,
        test_groups = [],
        corr_method = 'spearman'
        ):
        fig, ax = plt.subplots(figsize=(6, 4))
        # bin wgmb into 10 bins
        self.mut_count_w_pred_results_df['wg_burden_bin'] = pd.qcut(
            self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
                )['wg_burden'], q = num_bins,
            )
        # get correlation between clock burden and residual within each wgmb bin
        corr_df = self.mut_count_w_pred_results_df.query(
            "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
            ).groupby(['wg_burden_bin'])[['clock_burden', 'wg_burden', 'weighted_clock_burden','abs_weighted_clock_burden', 'residual','age_at_index','pred_age']].corr(method = corr_method)
        # drop rows with all nan values
        corr_df = corr_df.dropna(axis=0, how='all')
        # melt and only keep the correlations between burden and residual
        corr_df = corr_df.reset_index().query("level_1 == 'residual'").melt(
            id_vars = 'wg_burden_bin', value_vars = ['wg_burden', 'clock_burden','weighted_clock_burden', 'abs_weighted_clock_burden', 'age_at_index']
            ).rename(columns = {'variable': 'burden_type', 'value': 'correlation'})
        # plot
        sns.pointplot(
            data = corr_df, x = 'wg_burden_bin', y = 'correlation', hue = 'burden_type',
            ax = ax
        )
        # set y axis label 
        plt.ylabel("Correlation with residual")
        # angle x labels
        _  = plt.setp(ax.get_xticklabels(), rotation=45, ha = 'right')
        return corr_df
    
    def fit_mixed_linear_model(
        self, 
        formula:str, 
        re_formula:str = "",
        tissue:str = 'all',
        pred_name:str = 'lm'
        ):
        """
        Fit a mixed linear model
        @ formula: the formula for the fixed effects
        @ re_formula: the formula for the random effects
        """
        import statsmodels.formula.api as smf 
        if tissue == 'all':
            lm = smf.mixedlm(
                formula, 
                self.mut_count_w_pred_results_df.query(
                    "this_cv_test_sample == True"
                    ),
                re_formula = re_formula,
                groups = self.mut_count_w_pred_results_df.query(
                    "this_cv_test_sample == True"
                    )['tissue']
                ) 
        else:
            # random intercept for each group
            lm = smf.mixedlm(
                formula, 
                self.mut_count_w_pred_results_df.query(
                    "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
                    ),
                re_formula = re_formula,
                groups = self.mut_count_w_pred_results_df.query(
                    "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
                    )['tissue']
                ) 
        self.mlm_results = lm.fit()
        # get the fitted values
        self.mut_count_w_pred_results_df[pred_name] = self.mlm_results.fittedvalues
        return self.mlm_results
       
    def get_linear_equation_from_mlm(
          self
        ):
        fixed_params = self.mlm_results.params
        fixed_params_repeated = [fixed_params.values for i in range(len(self.well_pred_tissues))]
        equation_df = pd.DataFrame(fixed_params_repeated, index = self.well_pred_tissues, columns = fixed_params.index)
        # adjust each by the random effects
        intercept_effects = {}
        clock_burden_effects = {}
        for tissue, array in self.mlm_results.random_effects.items():
            intercept_effects[tissue] = array[0] # group
            clock_burden_effects[tissue] = array[1] # clock_burden
        equation_df['Intercept'] = equation_df['Intercept'] + equation_df.index.map(intercept_effects)
        equation_df['log_clock_burden'] =  equation_df['log_clock_burden'] + equation_df.index.map(clock_burden_effects)
        # drop columns that end in Cov or Var
        equation_df = equation_df[equation_df.columns[~equation_df.columns.str.endswith('Cov') & ~equation_df.columns.str.endswith('Var')]]
        self.mlm_equation_df = equation_df
        
    def calculate_predictions_of_mlm(self):
        eq_cols_in_data = list(
            set(self.mlm_equation_df.columns).intersection(set(self.mut_count_w_pred_results_df.columns))
            )
        cols_to_create = list(
            set(self.mlm_equation_df.columns).difference(set(self.mut_count_w_pred_results_df.columns))
        )

        all_tissue_data_dfs = []
        for tissue in self.well_pred_tissues:
            this_tissue_data_df = self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue == @tissue"
                )[eq_cols_in_data]
            for col in cols_to_create:
                if col == 'Intercept':
                    this_tissue_data_df[col] = 1
                elif col == 'log_clock_burden:age_at_index':
                    this_tissue_data_df[col] = this_tissue_data_df['log_clock_burden'] * this_tissue_data_df['age_at_index']
                elif col == 'gender[T.MALE]':
                    this_tissue_data_df[col] = self.mut_count_w_pred_results_df.query(
                                        "this_cv_test_sample == True and tissue == @tissue"
                                    )['gender'] == 'MALE'
            this_tissue_data_df = this_tissue_data_df[self.mlm_equation_df.columns]
            this_tissue_data_df['mlm_prediction'] = this_tissue_data_df.dot(self.mlm_equation_df.loc[tissue])
            this_tissue_data_df['tissue'] = tissue
            all_tissue_data_dfs.append(this_tissue_data_df)
        all_tissue_data_df = pd.concat(all_tissue_data_dfs)
        self.mlm_data_pred_df = all_tissue_data_df
        # merge with self.mut_count_w_pred_results_df
        self.mut_count_w_pred_results_df = self.mut_count_w_pred_results_df.merge(
            self.mlm_data_pred_df[['mlm_prediction']], left_index=True, right_index=True, how = 'left',
            )
        
    def fit_linear_model(
        self, 
        formula:str, 
        tissue:str,
        pred_name:str
        ):
        import statsmodels.formula.api as smf 
        if tissue == 'all':
            lm = smf.ols(
                formula, 
                self.mut_count_w_pred_results_df.query(
                    "this_cv_test_sample == False"
                    )
                )
            lm_results = lm.fit()
            # if pred_name already in df, drop 
            if pred_name in self.mut_count_w_pred_results_df.columns:
                self.mut_count_w_pred_results_df.drop(columns = pred_name, inplace = True)
            # apply to all samples (now including test)
            self.mut_count_w_pred_results_df[pred_name] = lm_results.predict(self.mut_count_w_pred_results_df)
        else:
            lm = smf.ols(
                formula, 
                self.mut_count_w_pred_results_df.query(
                    "this_cv_test_sample == False and tissue == @tissue"
                    )
                )
            lm_results = lm.fit()
            # if pred_name already in df, drop 
            if pred_name in self.mut_count_w_pred_results_df.columns:
                self.mut_count_w_pred_results_df.drop(columns = pred_name, inplace = True)
            # apply to all samples (now including test)
            self.mut_count_w_pred_results_df[pred_name] = lm_results.predict(self.mut_count_w_pred_results_df)
            
        return lm_results
        
        
      
        
    def plot_lm_pred_residual_vs_residual(
        self
        ):
        sns.lmplot(
            data = self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
                ),
                x = 'residual', y = 'lm_pred_residual', hue = 'tissue', scatter = False
            )
        # plot y = x line based on min and max of residual and lm_pred_residual
        plt.plot(
            [self.mut_count_w_pred_results_df['residual'].min(), self.mut_count_w_pred_results_df['residual'].max()],
            [self.mut_count_w_pred_results_df['residual'].min(), self.mut_count_w_pred_results_df['residual'].max()],
            color = 'black', linestyle = '--'
            )
        
    def plot_lm_pred_residual_changing_one_variable(
        self,
        variable_to_change: str,
        range_of_values: list,
        variables_to_keep_constant: list,
        tissues: list 
        ):
        """
        Leaving the other variables as their median values, plot the variable_to_change vs. lm_pred_residual
        """
        # get the variables and coefs output by the model
        variables_w_coefs = self.mlm_results.params
        variable_names = variables_w_coefs.index.tolist()
        # get variables that are in the df
        variables_in_df = list(set(variable_names).intersection(set(self.mut_count_w_pred_results_df.columns)))
        # get the median value of each variable across all samples
        median_variable_vals_by_tissue = self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @self.well_pred_tissues"
                ).groupby('tissue')[variables_in_df].median().reset_index()
        # set sex to male
        median_variable_vals_by_tissue['C(gender)[T.MALE]'] = 1
        # set interaction term
        median_variable_vals_by_tissue['clock_burden:age_at_index'] = median_variable_vals_by_tissue['clock_burden'] * median_variable_vals_by_tissue['age_at_index']
        # get the random intercepts for each tissue
        random_intercepts = self.mlm_results.random_effects
        
        pred_resid_by_tissue = {}
        # iterate across tissues
        for tissue in tissues:
            # get the median values of the variables for this tissue
            this_tissue_median_vals = median_variable_vals_by_tissue.query("tissue == @tissue")
            # get the random intercept for this tissue
            this_tissue_intercept = random_intercepts[tissue].values
            # the constant value is the sum of the median values and intercept
            constant_value_sum = np.dot(
                    this_tissue_median_vals[variables_to_keep_constant].values,
                    variables_w_coefs[variables_to_keep_constant].values
                    )[0] + this_tissue_intercept
            # calculate the predicted residuals for each value of the variable_to_change
            predicted_residuals = [(value * variables_w_coefs[variable_to_change] + constant_value_sum)[0] for value in range_of_values]
            pred_resid_by_tissue[tissue] = predicted_residuals
        # convert to df
        pred_resid_by_tissue_df  = pd.DataFrame(pred_resid_by_tissue)
        pred_resid_by_tissue_df[variable_to_change] = range_of_values
        # stack
        pred_resid_by_tissue_df = pred_resid_by_tissue_df.melt(
            id_vars = variable_to_change, value_vars = tissues
            )
        
        return pred_resid_by_tissue_df
                
                
    def plot_age_prediction_scatterplots(
        self, 
        pred_column:str, 
        methylation: bool = False,
        tissues: list = [],
        all_together: bool = False,
        out_fn:str = "",
        robust: bool = False
        ):
        if not methylation:
            self.mut_count_w_pred_results_df.rename(
                columns = {'tissue':'Tissue', pred_column:'Mutation Age', 'age_at_index':'Chronological age (years)'}, inplace = True
                )
            pred_age_name = 'Mutation Age'
        else:
            self.methyl_mat_w_pred_results_df.rename(
                columns = {'tissue':'Tissue', pred_column:'Methylation Age', 'age_at_index':'Chronological age (years)'}, inplace = True
                )
            pred_age_name = 'Methylation Age'
        
        """if 'Brain' not in self.mut_count_w_pred_results_df['Tissue'].unique().tolist():
            tissue_name_map = {'BRCA':'Breast', 'LIHC':'Liver', 'LGG':'Brain', 'COAD':'Colon', 'CESC':'Cervix', 'THCA':'Thyroid', 'UCEC':'Uterus'}
            self.mut_count_w_pred_results_df['Tissue'] = self.mut_count_w_pred_results_df['Tissue'].map(tissue_name_map)"""
            
        if all_together: # all
            if not methylation:
                subset_tissues = self.mut_count_w_pred_results_df.query(
                                "this_cv_test_sample == True and Tissue in @tissues"
                                )
            else:
                subset_tissues = self.methyl_mat_w_pred_results_df.query(
                                "this_cv_test_sample == True and Tissue in @tissues"
                                )
            subset_tissues['Tissue'] = 'All'
            my_palette = ['darkgrey', 'steelblue', 'maroon', 'darkgoldenrod', 'lightgrey', 'lightblue', 'salmon', 'wheat']

            sns.lmplot(
                data = subset_tissues, x = 'Chronological age (years)', y = pred_age_name, hue = 'Tissue', scatter_kws = {'alpha':0.5,'edgecolor':'black', 'rasterized':True},
                palette=my_palette, robust = robust,
                #hue_order = hue_order[1:], 
                line_kws={'color':'black'}
                )
            # get pearson corr within each tissue
            pearson_corrs = subset_tissues.groupby('Tissue').apply(lambda x: pearsonr(x['Chronological age (years)'], x[pred_age_name])[0])
            mae_by_tissue = subset_tissues.groupby('Tissue').apply(lambda x: mean_absolute_error(x['Chronological age (years)'], x[pred_age_name]))
            # add pearson and mae to each plot
            for ax, tissue in zip(plt.gcf().axes, pearson_corrs.index.tolist()):
                ax.annotate(f"Pearson r = {pearson_corrs.loc[tissue]:.2f}", xy = (0.05, 0.9), xycoords = 'axes fraction')
                ax.set_title(tissue)
                ax.annotate(f"MAE = {mae_by_tissue.loc[tissue]:.2f} years", xy = (0.05, 0.85), xycoords = 'axes fraction')
            # set x and y ranges
            plt.xlim(10,95)
            plt.ylim(23,82)
            
        else: 
            if not methylation:
                subset_tissues = self.mut_count_w_pred_results_df.query(
                                "this_cv_test_sample == True and Tissue in @tissues"
                                )
            else:
                subset_tissues = self.methyl_mat_w_pred_results_df.query(
                                "this_cv_test_sample == True and Tissue in @tissues"
                                )
            #hue_order = ['All', 'Brain', 'Colon', 'Liver', 'Cervix', 'Thyroid', 'Uterus', 'Breast']
            my_palette = ['darkgrey', 'steelblue', 'maroon', 'darkgoldenrod', 'lightgrey', 'lightblue', 'salmon', 'wheat']
            # sort to_plot by hue_order
            subset_tissues['Tissue'] = pd.Categorical(subset_tissues['Tissue'], tissues)
            sns.lmplot(
                data = subset_tissues, x = 'Chronological age (years)', y = pred_age_name, hue = 'Tissue', scatter_kws = {'alpha':0.5,'edgecolor':'black', 'rasterized':True} , col = 'Tissue', col_wrap = 3, palette=my_palette[1:], robust = robust,
                #hue_order = tissues, 
                line_kws={'color':'black'}
                )
            # get pearson corr within each tissue
            pearson_corrs = subset_tissues.groupby('Tissue').apply(lambda x: pearsonr(x['Chronological age (years)'], x[pred_age_name])[0])
            pearson_pvals = subset_tissues.groupby('Tissue').apply(lambda x: pearsonr(x['Chronological age (years)'], x[pred_age_name])[1])

            mae_by_tissue = subset_tissues.groupby('Tissue').apply(lambda x: mean_absolute_error(x['Chronological age (years)'], x[pred_age_name]))
            # add pearson and mae to each plot
            for ax, tissue in zip(plt.gcf().axes, tissues):
                ax.annotate(f"Pearson r = {pearson_corrs.loc[tissue]:.2f}, p = {pearson_pvals.loc[tissue]:.2e}", xy = (0.05, 0.9), xycoords = 'axes fraction')
                ax.set_title(tissue)
                ax.annotate(f"MAE = {mae_by_tissue.loc[tissue]:.2f} years", xy = (0.05, 0.85), xycoords = 'axes fraction')
                
        if not methylation:
            self.mut_count_w_pred_results_df.rename(columns = {'Tissue':'tissue', 'Mutation Age': pred_column, 'Chronological age (years)':'age_at_index'}, inplace = True)
        else:
            self.methyl_mat_w_pred_results_df.rename(columns = {'Tissue':'tissue', 'Methylation Age': pred_column, 'Chronological age (years)':'age_at_index'}, inplace = True)
        if out_fn != "":
            plt.savefig(out_fn, dpi = 300, format = 'svg')
           
    
           
            
    def plot_perf_by_tissue_bar(
        self,
        mut_both:str,
        mut_global:str,
        methyl_both:str,
        methyl_global:str,
        xlabels:list,
        tissues: list = [],
        one_cv_only: int = -1,
        out_fn:str = "",
        ):
        """self.mut_count_w_pred_results_df.rename(
            {mut_both:'Mutation clock',  mut_global:'Mut. global features'},
            axis = 1, inplace = True
            )
        self.methyl_mat_w_pred_results_df.rename(
            {methyl_both:'Methylation clock', methyl_global:'Methyl. global features'},
            axis = 1, inplace = True
            )"""
        # combine the relevant columns into one dataframe
        if one_cv_only == -1:
            mut_test_sample_df = self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @tissues"
                ).reset_index(drop = False)
            methyl_test_sample_df = self.methyl_mat_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @tissues"
                ).reset_index(drop = False)
        else:
            mut_test_sample_df = self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @tissues and cv_number == @one_cv_only"
                ).reset_index(drop = False)
            methyl_test_sample_df = self.methyl_mat_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @tissues and cv_number == @one_cv_only"
                ).reset_index(drop = False)
            
        test_sample_df = mut_test_sample_df.merge(
            methyl_test_sample_df, 
            left_on = ['case_submitter_id', 'cv_number'], 
            right_on = ['case_submitter_id', 'cv_number'], 
            how = 'left',
            suffixes = ('_mut', '_methyl')
            )
        
        # explaining age
        var_explained_all = test_sample_df[
                ['age_at_index_mut', methyl_both, mut_both, mut_global, methyl_global]
            ].corr(method = 'pearson').iloc[0, :]
        var_explained_all['tissue'] = 'All'
        var_explained_all.rename({'tissue':'tissue_mut'}, inplace = True)
        # get variance explained by tissue
        var_explained_by_tissue = test_sample_df.groupby('tissue_mut')[
            ['age_at_index_mut', methyl_both, mut_both, mut_global, methyl_global]
            ].corr(method = 'pearson').iloc[0::5, :]
        var_explained_by_tissue.reset_index(inplace = True)
        var_explained_by_tissue.drop(columns = ['level_1', 'age_at_index_mut'], inplace = True)
        # combine all and each tissue
        var_explained_to_plot = var_explained_by_tissue.append(var_explained_all, ignore_index = False)
        var_explained_to_plot.set_index('tissue_mut', inplace = True)
        # stack
        var_explained_to_plot = var_explained_to_plot.stack().to_frame().reset_index(drop = False).rename(columns = {0:'var_explained', 'level_1':'model'})
        # do plotting
        x_order = [mut_both, mut_global, methyl_both, methyl_global]
        my_palette= ['darkgrey', 'steelblue', 'maroon', 'darkgoldenrod', 'lightgrey', 'lightblue', 'salmon', 'wheat']
        fig, axes = plt.subplots(figsize = (8, 3), sharey=True)
        # order tissues by mutation clock perf
        hue_order = var_explained_to_plot.query("tissue_mut != 'All' and model == @mut_both").sort_values(
            by = 'var_explained', ascending = False
            )['tissue_mut'].tolist()
        # prepend 'All' to hue_order
        hue_order = ['All'] + hue_order
        # plot
        sns.barplot(
            var_explained_to_plot, x= 'model', y = 'var_explained', hue = 'tissue_mut',
            palette = my_palette, ax = axes, order = x_order, hue_order = hue_order,
            edgecolor = 'black'
        )
        axes.set_ylabel("Pearson r with chron. age")
        axes.set_xlabel("")
        axes.set_ylim(0, var_explained_to_plot.query("model != 'age_at_index_mut'")['var_explained'].max() + 0.02)
        sns.despine()
        # set x tick labels
        axes.set_xticklabels(xlabels)
        if out_fn != "":
            plt.savefig(out_fn, dpi = 300, format = 'svg')
        
        
        
    def plot_single_year_boxplots(
        self,
        mut_clock_col:str,
        epi_clock_col:str,
        tissues: list = [],
        out_fn: str = ""
        ):
        bin_size = 1
        self.mut_count_w_pred_results_df['age_bin'] = pd.cut(
            self.mut_count_w_pred_results_df.query(
                    "this_cv_test_sample == True and tissue in @tissues"
                    )['age_at_index'], # 40, 56
            bins = [x for x in range(42, 68, bin_size)], 
            labels = [f'{i}' for i in range(42, 68-bin_size, bin_size)]
            )

        self.mut_count_w_pred_results_df['mut_clock_bin'] = self.mut_count_w_pred_results_df.query(
            "this_cv_test_sample == True and tissue in @tissues"
            ).groupby(['age_bin'])[mut_clock_col].transform(
                lambda x: pd.qcut(x, q = 3, duplicates = 'drop', labels = ['Low', 'Intermediate' ,'High'])
                )
        to_plot = self.mut_count_w_pred_results_df.query(
                    "this_cv_test_sample == True and tissue in @tissues"
                    ).copy(deep = True)
        # merge in epi_clock_col from self.methyl_mat_w_pred_results_df
        to_plot = to_plot.merge(
            self.methyl_mat_w_pred_results_df.query("this_cv_test_sample == True and tissue in @tissues")[epi_clock_col], left_index=True, right_index=True, how = 'left'
        )
        to_plot.dropna(subset = ['age_bin'], inplace = True)
        to_plot['age_bin'] = to_plot['age_bin'].astype(int)
        to_plot.query("age_bin % 2 != 0", inplace = True)
        fig, axes = plt.subplots(figsize = (16, 6))
        sns.boxplot(
            data = to_plot, x='age_bin', hue = 'mut_clock_bin', y = epi_clock_col, ax = axes, palette='Reds', showfliers = False
                    )
        
        plt.ylabel('Methylation Age')
        plt.xlabel('Chronological age (years)')
        sns.despine()
        # change legend name
        _ = axes.legend(title = 'Mutation Age', loc = 'upper left')
        # angle x tick labels
        if out_fn != "":
            plt.savefig(out_fn, dpi = 300, format = 'svg')
        
    def plot_residuals_violin(
        self,
        mut_clock_resid_col:str,
        epi_clock_resid_col:str,
        tissues:list = [],
        out_fn:str = ""
        ):
        import pingouin as pg
        # convert to z score within each age bin
        # bin into standard deviation bins
        to_plot = self.mut_count_w_pred_results_df.query(
                    "this_cv_test_sample == True and tissue in @tissues"
                    ).copy(deep = True)
        to_plot['mut_clock_resid_bin'] = to_plot[mut_clock_resid_col].transform(
                lambda x: pd.cut(x, bins = [-40, -30, -20, -10, 10, 20, 30, 40], 
                                 right=False, duplicates = 'drop',)
                )
        
        # merge in epi_clock_col from self.methyl_mat_w_pred_results_df
        to_plot = to_plot.merge(
            self.methyl_mat_w_pred_results_df.query("this_cv_test_sample == True and tissue in @tissues")[[epi_clock_resid_col, 'methyl_age_xgb_w_tissue_entropy']], left_index=True, right_index=True, how = 'left'
        )
        fig, axes = plt.subplots(figsize = (6, 5))
        sns.violinplot(
            data = to_plot,
            x='mut_clock_resid_bin', y =epi_clock_resid_col, 
            ax = axes, palette='Reds',
            scale='width'
            )
        axes.set_ylabel('Methylation age – chronological age')
        axes.set_xlabel('Mutation age – chronological age')
        # plot y = 0 axhline
        plt.axhline(y = 0, color = 'black', linestyle = '--')
        sns.despine()
        
        
        partial_corr_results = pg.partial_corr(
            data = to_plot, x = mut_clock_resid_col, y=epi_clock_resid_col, covar = 'age_at_index'
        )
        plt.annotate(f"Pearson r = {partial_corr_results['r'].values[0]:.2f}", xy = (0.05, 0.9), xycoords = 'axes fraction')
        # add pvalue too
        plt.annotate(f"p = {partial_corr_results['p-val'].values[0]:.2e}", xy = (0.05, 0.85), xycoords = 'axes fraction')
        
        print(pearsonr(to_plot['mut_age_xgb_w_tissue_entropy'], to_plot['methyl_age_xgb_w_tissue_entropy']))
        if out_fn != "":
            plt.savefig(out_fn, dpi = 300, format = 'svg')
        return to_plot
        
        
    def calc_outliers_median(self, quantile):
        ninety_percentile  = self.mut_count_w_pred_results_df.groupby('tissue')['wg_burden'].quantile(quantile)
        self.mut_count_w_pred_results_df['tissue_wg_quantile'] = self.mut_count_w_pred_results_df['tissue'].map(ninety_percentile)        
        self.mut_count_w_pred_results_df['is_outlier_q'] = (
            self.mut_count_w_pred_results_df['wg_burden'] > self.mut_count_w_pred_results_df['tissue_wg_quantile']
            )

        
        
    def calc_outliers(self, stand_devs):
        mean_by_tissue = self.mut_count_w_pred_results_df.groupby('tissue')['wg_burden'].mean()
        std_by_tissue = self.mut_count_w_pred_results_df.groupby('tissue')['wg_burden'].std()
        # add mean and std to mut_clock_tcga_hunnidk_all_tenkb_pancan_big.mut_count_w_pred_results_df
        self.mut_count_w_pred_results_df['tissue_mean_wg_burden'] = self.mut_count_w_pred_results_df['tissue'].map(mean_by_tissue)
        self.mut_count_w_pred_results_df['tissue_std_wg_burden'] = self.mut_count_w_pred_results_df['tissue'].map(std_by_tissue)
        self.mut_count_w_pred_results_df['top_outlier_bound'] = self.mut_count_w_pred_results_df['tissue_mean_wg_burden'] + stand_devs*self.mut_count_w_pred_results_df['tissue_std_wg_burden']
        self.mut_count_w_pred_results_df['bottom_outlier_bound'] = self.mut_count_w_pred_results_df['tissue_mean_wg_burden'] - stand_devs*self.mut_count_w_pred_results_df['tissue_std_wg_burden']
        self.mut_count_w_pred_results_df['is_outlier'] = (self.mut_count_w_pred_results_df['wg_burden'] > self.mut_count_w_pred_results_df['top_outlier_bound']) | (self.mut_count_w_pred_results_df['wg_burden'] < self.mut_count_w_pred_results_df['bottom_outlier_bound'])


    def new_plot_perf_by_tissue_bar(self, mut_both: str, mut_global: str, methyl_both: str, methyl_global: str, tissues: list = [],
                                one_cv_only: int = -1, out_fn: str = "", mae_instead: bool = False):
        self.mut_count_w_pred_results_df.rename({mut_both: 'Mutation clock', mut_global: 'Mut. global features'},
                                                axis=1, inplace=True)
        self.methyl_mat_w_pred_results_df.rename({methyl_both: 'Methylation clock', methyl_global: 'Methyl. global features'},
                                                 axis=1, inplace=True)

        if one_cv_only == -1:
            mut_test_sample_df = self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @tissues").reset_index(drop=False)
            methyl_test_sample_df = self.methyl_mat_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @tissues").reset_index(drop=False)
        else:
            mut_test_sample_df = self.mut_count_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @tissues and cv_number == @one_cv_only").reset_index(
                drop=False)
            methyl_test_sample_df = self.methyl_mat_w_pred_results_df.query(
                "this_cv_test_sample == True and tissue in @tissues and cv_number == @one_cv_only").reset_index(
                drop=False)

        test_sample_df = mut_test_sample_df.merge(
            methyl_test_sample_df,
            left_on=['case_submitter_id', 'cv_number'],
            right_on=['case_submitter_id', 'cv_number'],
            how='left',
            suffixes=('_mut', '_methyl')
        )

        if not mae_instead:
            # Pearson r
            var_explained_all = test_sample_df[
                ['age_at_index_mut', 'Methylation clock', 'Mutation clock', 'Mut. global features', 'Methyl. global features']
            ].corr(method='pearson').iloc[0, :]
            var_explained_all['tissue'] = 'All'
            var_explained_all.rename({'tissue': 'tissue_mut'}, inplace=True)

            var_explained_by_tissue = test_sample_df.groupby('tissue_mut')[
                ['age_at_index_mut', 'Methylation clock', 'Mutation clock', 'Mut. global features', 'Methyl. global features']
            ].corr(method='pearson').iloc[0::5, :]
            var_explained_by_tissue.reset_index(inplace=True)
            var_explained_by_tissue.drop(columns=['level_1', 'age_at_index_mut'], inplace=True)
        else:
            # MAE
            var_explained_all = pd.DataFrame()
            for col in ['Methylation clock', 'Mutation clock', 'Mut. global features', 'Methyl. global features']:
                mae_value = mean_absolute_error(test_sample_df['age_at_index_mut'], test_sample_df[col])
                var_explained_all[col] = [mae_value]
            var_explained_all['tissue'] = 'All'
            var_explained_all.rename(columns={'tissue': 'tissue_mut'}, inplace=True)

            var_explained_by_tissue = pd.DataFrame()
            for tissue in test_sample_df['tissue_mut'].unique():
                tissue_data = test_sample_df[test_sample_df['tissue_mut'] == tissue]
                mae_values = {}
                for col in ['Methylation clock', 'Mutation clock', 'Mut. global features', 'Methyl. global features']:
                    mae_value = mean_absolute_error(tissue_data['age_at_index_mut'], tissue_data[col])
                    mae_values[col] = mae_value
                mae_values['tissue_mut'] = tissue
                var_explained_by_tissue = var_explained_by_tissue.append(mae_values, ignore_index=True)

        # Combine all and each tissue
        var_explained_to_plot = var_explained_by_tissue.append(var_explained_all, ignore_index=True)
        var_explained_to_plot.set_index('tissue_mut', inplace=True)
        var_explained_to_plot = var_explained_to_plot.stack().to_frame().reset_index(drop=False).rename(
            columns={0: 'var_explained', 'level_1': 'model'})

        # Do plotting
        x_order = ['Mutation clock', 'Mut. global features', 'Methylation clock', 'Methyl. global features']
        my_palette = ['darkgrey', 'steelblue', 'maroon', 'darkgoldenrod', 'lightgrey', 'lightblue', 'salmon', 'wheat']
        fig, axes = plt.subplots(figsize=(8, 3), sharey=True)
        hue_order = var_explained_to_plot.query("tissue_mut != 'All' and model == 'Mutation clock'").sort_values(
            by='var_explained', ascending=False)['tissue_mut'].tolist()
        hue_order = ['All'] + hue_order

        sns.barplot(data=var_explained_to_plot, x='model', y='var_explained', hue='tissue_mut',
                    palette=my_palette, ax=axes, order=x_order, hue_order=hue_order,
                    edgecolor='black')

        ylabel = "Pearson r with chron. age" if not mae_instead else "MAE with chron. age"
        axes.set_ylabel(ylabel)
        axes.set_xlabel("")
        if not mae_instead:
            axes.set_ylim(0, var_explained_to_plot.query("model != 'age_at_index_mut'")['var_explained'].max() + 0.02)
        else:
            axes.set_ylim(0, var_explained_to_plot['var_explained'].max() + 1)

        sns.despine()

        # Rename back
        self.mut_count_w_pred_results_df.rename({mut_both: 'Mutation Age', mut_global: 'Mut. global features'},
                                                axis=1, inplace=True)
        self.methyl_mat_w_pred_results_df.rename({methyl_both: 'Methylation Age', methyl_global: 'Methyl. global features'},
                                                 axis=1, inplace=True)
