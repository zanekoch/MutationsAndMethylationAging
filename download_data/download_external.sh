# get sample metadata  
wget https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp .
mv Survival_SupplementalTable_S1_20171025_xena_sp PANCAN_meta.tsv

# get illumina probemap
wget https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/probeMap%2FilluminaMethyl450_hg19_GPL16304_TCGAlegacy .

# get mutatations
wget https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/mc3.v0.2.8.PUBLIC.xena.gz .
mv mc3.v0.2.8.PUBLIC.xena.gz PANCAN_mut.tsv.gz

# get methylation
wget https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz .
mv jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz PANCAN_methyl.tsv.gz

# get reference genome hg19
wget https://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz

# got cgi from ucsc too
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/cpgIslandExt.txt.gz

# get GC content
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/gc5Base/hg19.gc5Base.txt.gz

# get gene annotation hg19
http://genome.ucsc.edu/cgi-bin/hgTables?hgsid=1609098573_NpuW1mQ66OVaW4mH7z0yuEv9NEAb&clade=mammal&org=Human&db=hg19&hgta_group=genes&hgta_track=knownGene&hgta_table=0&hgta_regionType=genome&position=chr16%3A56%2C623%2C013-56%2C692%2C077&hgta_outputType=primaryTable&hgta_outFileName=CpG_islands_hg19.bed
then sftp

# ucsc known gene to refseq name conversion
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/knownToRefSeq.txt.gz