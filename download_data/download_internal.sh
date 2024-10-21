# download data from figshare
curl -L -o data.zip "https://figshare.com/ndownloader/files/49904031"
# unzip
unzip data.zip
# move data to the MutationsAndMethylationAging directory, such that MutationsAndMethylationAging/data exists
mv data ..
