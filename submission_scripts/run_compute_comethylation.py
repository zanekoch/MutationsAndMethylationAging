import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import compute_comethylation, get_data
import os
import pandas as pd
import sys
import dask.dataframe as dd
import argparse


def do_comethylation_disturbance_analysis(
    out_dir: str,
    all_mut_w_age_df: pd.DataFrame,
    illumina_cpg_locs_df: pd.DataFrame,
    all_methyl_age_df_t: pd.DataFrame,
    corr_dir: str,
    start_num_mut_to_process: int,
    end_num_mut_to_process: int,
    linkage_method: str = 'dist',
    age_bin_size: int = 10,
    max_dist: int = 100000,
    num_correl_sites: int = 1000,
    num_background_events: int = 100,
    matched_sample_num: int = 20,
    mut_collision_dist: int = 1000,
    all_comparison_site_df: pd.DataFrame = None
    ):
    # create mut_scan object
    mut_scan = compute_comethylation.mutationScan(
        all_mut_w_age_df, illumina_cpg_locs_df, 
        all_methyl_age_df_t, corr_dir = corr_dir,
        age_bin_size = age_bin_size, max_dist = max_dist,
        num_correl_sites = num_correl_sites, num_background_events = num_background_events,
        matched_sample_num = matched_sample_num, mut_collision_dist = mut_collision_dist
        )
    # run comethylation disturbance analysis
    comparison_sites_df_test, all_metrics_df_test = mut_scan.look_for_disturbances(
        start_num_mut_to_process = start_num_mut_to_process,
        end_num_mut_to_process = end_num_mut_to_process,
        linkage_method = linkage_method, 
        out_dir = out_dir, 
        corr_direction = 'pos',
        comparison_sites_df = all_comparison_site_df
        )

    return comparison_sites_df_test, all_metrics_df_test


def main():
    # argparse
    parser = argparse.ArgumentParser(description='Comethylation disturbance analysis')
    parser.add_argument('--out_dir', type=str, help='output directory')
    parser.add_argument('--consortium', type=str, help='TCGA or ICGC')
    # add a flag for each argument in do_comethylation_disturbance_analysis
    parser.add_argument('--corr_dir', type=str, help='path to correlation directory')
    parser.add_argument('--start_num_mut_to_process', type=int, help='start number of mutations to process')
    parser.add_argument('--end_num_mut_to_process', type=int, help='end number of mutations to process')
    parser.add_argument('--linkage_method', type=str, help='linkage method')
    parser.add_argument('--age_bin_size', type=int, help='age bin size')
    parser.add_argument('--max_dist', type=int, help='max distance')
    parser.add_argument('--num_correl_sites', type=int, help='number of correl sites')
    parser.add_argument('--num_background_events', type=int, help='number of background events')
    parser.add_argument('--matched_sample_num', type=int, help='number of matched samples')
    parser.add_argument('--mut_collision_dist', type=int, help='mutation collision distance')
    parser.add_argument(
        '--all_comparison_site_dir', type=str,
        help='path to directory of comparison sites', default=None
        )
    
    # parse
    args = parser.parse_args()
    out_dir = args.out_dir
    consortium = args.consortium
    corr_dir = args.corr_dir
    start_num_mut_to_process = args.start_num_mut_to_process
    end_num_mut_to_process = args.end_num_mut_to_process
    linkage_method = args.linkage_method
    age_bin_size = args.age_bin_size
    max_dist = args.max_dist
    num_correl_sites = args.num_correl_sites
    num_background_events = args.num_background_events
    matched_sample_num = args.matched_sample_num
    mut_collision_dist = args.mut_collision_dist
    all_comparison_site_dir = args.all_comparison_site_dir
    
    # make output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # get data
    if consortium == "ICGC":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, _, _ = get_data.read_icgc_data()
    elif consortium == "TCGA":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, _, _ = get_data.read_tcga_data()
    elif consortium == "TCGA-normal":
        all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, _, _ = get_data.read_normal_tcga_data(qnorm_methylation=True)
        all_mut_w_age_df.query("st_coverage >= 50 and bld_coverage >= 50", inplace=True)
    else:
        raise ValueError("consortium must be TCGA or ICGC")
    if all_comparison_site_dir is not None:
        all_comparison_site_dd = dd.read_parquet(all_comparison_site_dir)
        all_comparison_site_df = all_comparison_site_dd.compute()
    else:
        all_comparison_site_df = None
        
    print(
        f"Running comethylation disturbance analysis with {start_num_mut_to_process} to {end_num_mut_to_process} mutations, {linkage_method} linkage method and writing to {out_dir}", flush=True
        )
    
    # run comethylation disturbance analysis
    comparison_sites_df_test, all_metrics_df_test = do_comethylation_disturbance_analysis(
        out_dir, all_mut_w_age_df, illumina_cpg_locs_df, all_methyl_age_df_t, corr_dir, 
        start_num_mut_to_process, end_num_mut_to_process, linkage_method, age_bin_size, max_dist, 
        num_correl_sites, num_background_events, matched_sample_num, mut_collision_dist, all_comparison_site_df
        )
        
    
if __name__ == '__main__':
    main()