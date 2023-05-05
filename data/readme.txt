This folder contains the TCGA RNA-seq data used in test_rna_tcga.ipynb as well
as the tools needed to download and extract data from the Gene Expression Omnibus
via the ARCHS4 platform.

See report for more attribution and explanation beyond the details below.

* The cancer genome atlas (TCGA)
    * These were collected via the UCI machine learning repository
        * https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq
    * Both input data and labels can be downloaded and placed in TCGA-PANCAN-HiSeq-801x20531/

* Gene expression omnibus (GEO)
    * To download and extract the data that was used to run test_rna_geo.ipynb...
        * Download 'human_matrix_v1.11.h5' from ARCHS4
            * https://maayanlab.cloud/archs4/download.html 
        * Place h5 data in the same folder as cancer_sample_selector.R
        * Run cancer_sample_selector.R to save relevant samples
            * Gene expression levels are in cancer_expression_matrix.tsv
    * Labels are in cancer_expression_labels.tsv
        * Extracted from GEO metadata in the included notebook
