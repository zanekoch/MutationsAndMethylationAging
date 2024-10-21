import sys
sys.path.append('/cellar/users/zkoch/methylation_and_mutation/source_files')
import compute_comethylation
import os
import sys
import glob
import dask.dataframe as dd
import argparse

def get_mean_metrics_by_dist(
    metrics_path_index: int,
    all_metrics_paths: list,
    absolute_distances: list,
    out_dir: str
    ):
    # initialize analyzeComethylation object
    analyze_comethylation = compute_comethylation.analyzeComethylation()
    # read in one metrics file
    print("reading in", all_metrics_paths[metrics_path_index], flush=True)
    one_metrics_dd = dd.read_parquet(all_metrics_paths[metrics_path_index])
    one_metrics_df = one_metrics_dd.compute()
    print("Getting mean metrics", flush=True)
    # get mean metrics for distances 
    one_mean_metrics = analyze_comethylation.get_mean_metrics_by_dist(
        one_metrics_df, 
        absolute_distances = absolute_distances
    )   
    print("done with mean methyl", all_metrics_paths[metrics_path_index], flush=True)
    # get muts number to base name on
    muts_num = all_metrics_paths[metrics_path_index].split("/")[-1].split("_")[2]
    # write to out_dir
    one_mean_metrics.to_parquet(os.path.join(out_dir, f"mean_metrics_by_dist_{muts_num}.parquet"))
    print(f'Wrote to {os.path.join(out_dir, f"mean_metrics_by_dist_{muts_num}.parquet")}', flush=True)
    
def main():
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='output directory')
    parser.add_argument('--comp_site_type', type=str, help='comp_site_type', required=True)
    parser.add_argument(
        '--metrics_path_index', type=int,
        help='index of path to metrics file', required=True
        )
    parser.add_argument(
        '--all_metrics_glob_path', type=str,
        help='glob path to all_metrics files', required=True
        )
    # parse
    args = parser.parse_args()
    # get arguments
    out_dir = args.out_dir
    comp_site_type = args.comp_site_type
    metrics_path_index = args.metrics_path_index
    all_metrics_glob_path = args.all_metrics_glob_path
    # get paths
    all_metrics_paths = glob.glob(all_metrics_glob_path)
    # set absolute distances
    if comp_site_type == 'dist':
        absolute_distances = [100, 500, 1000, 5000, 10000, 50000, 100000]
    elif comp_site_type == 'corr':
        absolute_distances = [5, 10, 50, 100]
    else:
        raise ValueError("comp_site_type must be 'dist' or 'corr'")
    # sort paths in place
    all_metrics_paths.sort(
        key=lambda x: 
            int(x.split('/')[-1].split('Muts')[0].split('all_metrics_')[1].split('-')[0])
        )
    # get mean metrics by dist
    get_mean_metrics_by_dist(
        metrics_path_index=metrics_path_index,
        all_metrics_paths=all_metrics_paths,
        absolute_distances=absolute_distances,
        out_dir=out_dir
        )
    
if __name__ == '__main__':
    main()