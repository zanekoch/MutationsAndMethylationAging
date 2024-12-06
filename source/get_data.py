import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import glob
import dask.dataframe as dd
# set working directory to the directory of this file
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_SCRIPT_DIR)
# CONSTANTS
VALID_MUTATIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G", "G>C","G>A", "A>T", "A>G" , "A>C", "G>T", "C>-"]


def infer_fns_from_data_dirs(data_dirs):
    """
    @ data_dirs: list of data directories
    @ returns: dict of dicts of filenames
    """
    data_files_by_name = {}
    dataset_names_list = []
    for data_dir in data_dirs:
        this_files_dict = {}
        data_set_name = data_dir.split('/')[-1].split('_')[1]
        dataset_names_list.append(data_set_name)
        # if there is a parquet version of methylation use that one
        if len(glob.glob( os.path.join(data_dir, "TCGA.{}.sampleMap2FHumanMethylation450.parquet".format(data_set_name.upper())) , recursive=False)) >= 1:
            methyl_fn = os.path.join(data_dir, "TCGA.{}.sampleMap2FHumanMethylation450.parquet".format(data_set_name.upper()))
        else:
            methyl_fn = os.path.join(data_dir, "TCGA.{}.sampleMap2FHumanMethylation450.gz".format(data_set_name.upper()))
        this_files_dict['methyl_fn'] = methyl_fn
        mut_fn = os.path.join(data_dir, "mc32F{}_mc3.txt.gz".format(data_set_name.upper()))
        this_files_dict['mut_fn'] = mut_fn
        # get ALL clinical files because there may be multiple as for coadread
        clinical_meta_fns = []
        for clinical_meta_fn in glob.glob( os.path.join(data_dir, "clinical*.tsv") , recursive=False):
            clinical_meta_fns.append(clinical_meta_fn)
        this_files_dict['clinical_meta_fns'] = clinical_meta_fns
        # add this_set_files dict to data_files_by_name under the name data_set_name
        data_files_by_name[data_set_name] = this_files_dict
    return data_files_by_name, dataset_names_list

def get_mutations(mut_fn, is_icgc = False):
    """
    @ data_files_by_name: dict of dicts of filenames
    @ returns: pandas dataframe of mutations
    """
    mut_df = pd.read_csv(mut_fn, sep='\t')
    if not is_icgc:
        # change sample names to not have '-01' at end
        mut_df['sample'] = mut_df['sample'].str[:-3]
    # subset cols
    mut_df = mut_df[['sample', 'chr', 'start', 'end', 'reference', 'alt', 'DNA_VAF']]
    mut_df["mutation"] = mut_df["reference"] + '>' + mut_df["alt"]
    # only keep rows with valid mutations
    mut_df = mut_df[mut_df["mutation"].isin(VALID_MUTATIONS)]
    
    return mut_df

def get_methylation(methylation_dir, is_icgc = False):
    """
    Read in the already preprocessed methylation data
    @ methylation_dir: directory of methylation data, or filename if is_icgc
    @ returns: pandas dataframe of methylation data
    """
    try: 
        if is_icgc:
            methyl_df = pd.read_parquet(methylation_dir)
        else:
            methyl_dd = dd.read_parquet(methylation_dir)
            print("Converting Dask df to pandas df, takes ~10min", flush=True)
            methyl_df = methyl_dd.compute()
    except Exception as e:
        print("ERROR: methylation data may not be downloaded, follow instructions in ./download_data/download_external.sh to download")
        print(e)
        raise e
    return methyl_df

def get_metadata(meta_fn, is_icgc = False):
    """
    @ metadata_fn: filename of metadata
    @ returns: 
        @ meta_df: pandas dataframe of metadata for all samples with duplicates removed and ages as ints
        @ dataset_names_list: list of dataset names
    """
    if is_icgc:
        meta_df = pd.read_csv(meta_fn, sep='\t')
        meta_df.set_index('sample', inplace=True)
        return meta_df, list(meta_df['dataset'].unique())
    else:
        # get metadata
        meta_df = pd.read_csv(meta_fn, sep='\t')
        meta_df = meta_df[['sample', 'age_at_initial_pathologic_diagnosis', 'cancer type abbreviation', 'gender']].drop_duplicates()
        meta_df['sample'] = meta_df['sample'].str[:-3]
        meta_df.set_index('sample', inplace=True)
        # drop nans
        meta_df.dropna(inplace=True)
        # rename to TCGA names
        meta_df = meta_df.rename(columns={"age_at_initial_pathologic_diagnosis":"age_at_index", "cancer type abbreviation":"dataset"})
        # drop ages that can't be formated as ints
        meta_df['age_at_index'] = meta_df['age_at_index'].astype(str)
        meta_df['age_at_index'] = meta_df[meta_df['age_at_index'].str.contains(r'\d+')]['age_at_index']
        dataset_names_list = list(meta_df['dataset'].unique())
        # make sure to duplicates still
        meta_df = meta_df.loc[meta_df.index.drop_duplicates()]
        # convert back to int, through float so e.g. '58.0' -> 58.0 -> 58
        meta_df['age_at_index'] = meta_df['age_at_index'].astype(float).astype(int)
        # drop rows with duplicate index
        meta_df = meta_df[~meta_df.index.duplicated(keep='first')]
        return meta_df, dataset_names_list

def transpose_methylation(all_methyl_df):
    """
    @ all_methyl_df: pandas dataframe of methylation fractions for all samples
    @ returns: pandas dataframe of methylation fractions for all samples, transposed
    """
    # turn methylation to numpy for fast transpose
    # save row and col names
    cpg_names = all_methyl_df.index
    sample_names = all_methyl_df.columns
    all_methyl_arr = all_methyl_df.to_numpy()
    all_methyl_arr_t = np.transpose(all_methyl_arr)
    # convert back
    all_methyl_df_t = pd.DataFrame(all_methyl_arr_t, index = sample_names, columns=cpg_names)
    return all_methyl_df_t

def get_illum_locs(illum_cpg_locs_fn):
    illumina_cpg_locs_df = pd.read_csv(
        illum_cpg_locs_fn, sep=',', dtype={'CHR': str}, low_memory=False
        )
    illumina_cpg_locs_df = illumina_cpg_locs_df.rename({
        "CHR": "chr", "MAPINFO":"start", "IlmnID": "#id"
        }, axis=1)
    illumina_cpg_locs_df = illumina_cpg_locs_df[['#id','chr', 'start', 'Strand']]
    return illumina_cpg_locs_df


def read_icgc_data() -> tuple:
    print("reading in data")
    # get icgc data
    icgc_data_dir = os.path.join(THIS_SCRIPT_DIR, "../data/icgc")
    dependency_f_dir = os.path.join(THIS_SCRIPT_DIR, "../dependency_files")
    illumina_cpg_locs_df, icgc_mut_df, icgc_methyl_df, icgc_methyl_df_t, icgc_meta_df, icgc_dataset_names_list = main(
        illum_cpg_locs_fn = os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv.gz"),
        out_dir = '',
        methyl_dir = os.path.join(icgc_data_dir, 'qnorm_withinDset_3DS_dropped'),
        mut_fn = os.path.join(icgc_data_dir, "icgc_mut_df.csv.gz"),
        meta_fn = os.path.join(icgc_data_dir, "icgc_meta.csv"),
        is_icgc=True
        )
    icgc_mut_w_age_df, icgc_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(icgc_mut_df, icgc_meta_df, icgc_methyl_df_t)
    return icgc_mut_w_age_df, illumina_cpg_locs_df, icgc_methyl_age_df_t, "", ""

def read_normal_tcga_data(qnorm_methylation = True) -> tuple:
    print("reading in normal data")
    dependency_f_dir = os.path.join(THIS_SCRIPT_DIR, "../dependency_files")
    data_dir = os.path.join(THIS_SCRIPT_DIR, "../data/tcga")
    # read metadata, changing bits to match
    normal_meta_df = pd.read_csv(os.path.join(data_dir, "solid_tissue_normal_meta.tsv"), sep='\t')
    normal_meta_df['sample'] = normal_meta_df['case_submitter_id'].str[:-4]
    normal_meta_df.set_index('sample', inplace=True)
    normal_meta_df.drop(columns = ['case_submitter_id'], inplace=True)
    normal_meta_df['gender'] = normal_meta_df['gender'].map({'male':'MALE', 'female':'FEMALE'})
    normal_meta_df['dataset'] = normal_meta_df['dataset'].str[5:]
    
    # read cancer meta
    cancer_meta_fn = os.path.join(data_dir, "PANCAN_meta.tsv")
    cancer_meta_df, _ = get_metadata(cancer_meta_fn, is_icgc=False)
    cancer_meta_df['tissue'] = 'cancer'
    # combine
    all_meta_df = pd.concat([normal_meta_df, cancer_meta_df], axis=0)
    all_meta_df.reset_index(inplace=True, drop=False)
    all_meta_df.drop_duplicates(subset = 'sample', inplace=True)
    all_meta_df.set_index('sample', inplace=True)
    

    # read mutations and callable basepairs
    all_mut_dfs = []
    callable_bp_dfs = []
    for dataset in ['BRCA', 'PRAD', 'LUAD', 'KIRP', 'HNSC', 'COAD', 'THCA', 'LUSC', 'LIHC', 'STAD', 'UCEC', 'BLCA','KIRC']:#
        mut_df = pd.read_parquet(os.path.join(f"../data/tcga/solid_tissue_vs_blood_mutations/{dataset}", f"{dataset}_solid_tissue_normal_mutations.parquet"))
        all_mut_dfs.append(mut_df)
        callable_bp_df = pd.read_parquet(os.path.join(f"../data/tcga/solid_tissue_vs_blood_mutations/{dataset}", f"{dataset}_solid_tissue_normal_callableBP.parquet"))
        callable_bp_dfs.append(callable_bp_df)
    # create mutation df
    all_mut_df = pd.concat(all_mut_dfs, axis=0)
    all_mut_df["mutation"] = all_mut_df["reference"] + '>' + all_mut_df["alt"]
    all_mut_df = all_mut_df[all_mut_df["mutation"].isin(VALID_MUTATIONS)]
    all_mut_df['sample'] = all_mut_df['sample'].str[:12]
    all_mut_df['chr'] = all_mut_df['chr'].str.replace('chr', '')
    mut_w_age_df = all_mut_df.merge(
        all_meta_df, left_on = 'sample', right_index=True, how='left'
        )
    
    mut_w_age_df.rename(columns={'st_VAF': 'DNA_VAF','sample': 'case_submitter_id' }, inplace=True)
    mut_w_age_df.reset_index(inplace=True, drop=True)
    
    # create callable bp df
    all_callable_bp_df = pd.concat(callable_bp_dfs, axis=0)
    # read methylation
    if qnorm_methylation:
        methyl_fn = os.path.join(data_dir, 'solid_tissue_normal_methylation_noNan_dropped3SD_qnormed.parquet')
    else:
        methyl_fn = os.path.join(data_dir, 'solid_tissue_normal_methylation_noNan.parquet')
    all_methyl_df = pd.read_parquet(methyl_fn)
    all_methyl_df_t = all_methyl_df.T
    all_methyl_df_t.reset_index(inplace=True, drop=False)
    all_methyl_df_t['index'] = all_methyl_df_t['index'].str[:12]
    all_methyl_df_t.set_index('index', inplace=True)
    
    # read illum
    illumina_cpg_locs_df = get_illum_locs(os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv.gz"))
    # merge meta data with transposed methylation
    methyl_age_df_t = all_methyl_df_t.merge(all_meta_df, left_index=True, right_index=True, how='left')
    methyl_age_df_t.dropna(axis=0, inplace=True)
    return mut_w_age_df, illumina_cpg_locs_df, methyl_age_df_t, all_meta_df, all_callable_bp_df
 

def read_tcga_data(qnorm_methylation = True) -> tuple:
    print("reading in data")
    # read in data
    dependency_f_dir = "../dependency_files"
    data_dir = "../data/tcga"
    if qnorm_methylation:
        methyl_dir = os.path.join(data_dir, 'dropped3SD_qnormed_methylation'),
    else:
        methyl_dir = os.path.join(data_dir, 'processed_methylation_noDropNaN'),
    
    illumina_cpg_locs_df, all_mut_df, _, all_methyl_df_t, all_meta_df, _ = main(
        illum_cpg_locs_fn = os.path.join(dependency_f_dir, "illumina_cpg_450k_locations.csv.gz"),
        out_dir = '',
        methyl_dir = methyl_dir,
        mut_fn = os.path.join(data_dir, "PANCAN_mut.tsv.gz"),
        meta_fn = os.path.join(data_dir, "PANCAN_meta.tsv")
        )
    # add ages to all_methyl_df_t
    all_mut_w_age_df, all_methyl_age_df_t = utils.add_ages_to_mut_and_methyl(
        all_mut_df, all_meta_df, all_methyl_df_t
        )
    if not qnorm_methylation:
        # do imputation
        def fill_nan2(df):
            for col in df.columns[df.isnull().any(axis=0)]:
                df[col].fillna(df[col].mean(),inplace=True)
            return df
        # do mean imputation by column
        all_methyl_age_df_t_filled = fill_nan2(all_methyl_age_df_t.iloc[:, 3:])
        all_methyl_age_df_t = pd.concat(
            [all_methyl_age_df_t.iloc[:, :3], all_methyl_age_df_t_filled],
            axis=1)
        
    return all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, all_meta_df, ""

def main(
    illum_cpg_locs_fn,
    out_dir, 
    methyl_dir,
    mut_fn, 
    meta_fn, 
    is_icgc = False
    ):
    # make output directories
    #os.makedirs(out_dir, exist_ok=True)
    # read in illumina cpg locations
    illumina_cpg_locs_df = get_illum_locs(illum_cpg_locs_fn)
    # read in mutations, methylation, and metadata
    all_mut_df = get_mutations(mut_fn, is_icgc)
    all_meta_df, dataset_names_list = get_metadata(meta_fn, is_icgc)
    # add dataset column to all_mut_df
    all_mut_df = all_mut_df.join(all_meta_df, on='sample', how='inner')
    all_mut_df = all_mut_df.drop(columns = ['age_at_index'])
    print("Got mutations and metadata, reading methylation", flush=True)
    all_methyl_df = get_methylation(methyl_dir, is_icgc)
    print("Got methylation, transposing", flush=True)
    # also create transposed methylation
    all_methyl_df_t = transpose_methylation(all_methyl_df)
    print("Done", flush=True)
    return illumina_cpg_locs_df, all_mut_df, all_methyl_df, all_methyl_df_t, all_meta_df, dataset_names_list

"""
processing save 
illumina_cpg_locs_df2 = illumina_cpg_locs_df = pd.read_csv(
        "../dependency_files/illumina_cpg_450k_locations.csv", sep=',', dtype={'CHR': str}, low_memory=False
        )
illumina_cpg_locs_df2 = illumina_cpg_locs_df2.rename({
        "CHR": "chr", "MAPINFO":"start", "IlmnID": "#id"
        }, axis=1)
refgene_lists = illumina_cpg_locs_df2['UCSC_RefGene_Group'].str.split(';')
# get unique values
unique_vals = refgene_lists.explode().unique()
# drop na from numpy array
unique_vals = unique_vals[~pd.isna(unique_vals)]
# Create one-hot-encoded DataFrame
illumina_cpg_locs_df2['UCSC_RefGene_Group'] = illumina_cpg_locs_df2['UCSC_RefGene_Group'].astype(str)

one_hot_encoded = illumina_cpg_locs_df2['UCSC_RefGene_Group'].apply(
    lambda x: {val: 1 for val in x.split(';') if val in unique_vals}
    ).apply(pd.Series).fillna(0)
one_hot_encoded_refgene_df = one_hot_encoded
one_hot_encoded_cpg_island = pd.get_dummies(illumina_cpg_locs_df2['Relation_to_UCSC_CpG_Island'])
one_hot_encoded_cpg_enhancer = pd.get_dummies(illumina_cpg_locs_df2['Enhancer'])
one_hot_encoded_cpg_enhancer.columns = ["Enhancer"]
one_hot_encoded_cpg_reg_feat_group = pd.get_dummies(illumina_cpg_locs_df2['Regulatory_Feature_Group'])
one_hot_encoded_dhs = pd.get_dummies(illumina_cpg_locs_df2['DHS'])
one_hot_encoded_dhs.columns = ["DHS"]
# combine 
illumina_cpg_locs_df2 = pd.concat([illumina_cpg_locs_df2[['#id','chr', 'start', 'Strand']], one_hot_encoded_refgene_df, one_hot_encoded_cpg_island, one_hot_encoded_cpg_enhancer, one_hot_encoded_cpg_reg_feat_group, one_hot_encoded_dhs], axis=1)
illumina_cpg_locs_df2.to_csv("../dependency_files/illumina_cpg_450k_locations_one_hot_encoded.csv", index=False)
"""