# download data from figshare
curl -L -o data.zip "https://figshare.com/ndownloader/files/49904031"
# unzip
unzip data.zip
# move data to the MutationsAndMethylationAging directory, such that MutationsAndMethylationAging/data exists if it does not already 

# gunzip the cpg location file, which had to be compressed to commit to github
gunzip ../dependency_files/illumina_cpg_450k_locations.csv.gz