# Data-Science-MSc-Individual-project
SOURCE CODE PYTHON FILES ARE STORED HERE

----------------------------- PAIRS -----------------------------------

DLSH.py  ## This is the main source code for songs PAIRS
         ## to compute Jaccard similarity,
         ## MinHash similarity, LSH banding technique.
         ## It computes also metrics for MinHash similarity vs Jaccard similarity
         ## It computes metrics & TP, TN, FP, FN for LSH 
         ## Plots S-curve for probability of being a candidate pairs vs similarity

LSH_SIM_MH.py  ## shorter version of DLSH.py it does not compute plots, Jaccard similarity, csv tables)
               ## It takes MinHash similarity as the only measure of similarity available and no metrics vs Jaccard similarity are computed
               ## the code focuses on Minhash signatures and LSH candidate pairs
               ## computing the MinHash similarity for only those pairs which are candidate pairs (obtained from LSH banding technique)
               ###### AIM: this is important to have an idea of the time and memory of similarity search by LSH via Minhashing ########


WJS_JS.py  #### source code for the computation of Jaccard similarity and weighted Jaccard similarity for songs PAIRS
           #### It computes also the superimpposed histograms of Jaccard and weighted Jaccard similarity
           #### It computes the statistics for Jaccard and weighted Jaccard similarity


----------------------------- TRIPLETS ------------------------------------------------------------------------------------------------------------

TLSH.py    ## This is the main source code for TRIPLETS
           ## to compute Jaccard similarity,
           ## MinHash similarity, LSH banding technique.
           ## It computes also metrics for MinHash similarity vs Jaccard similarity
           ## It computes metrics & TP, TN, FP, FN for LSH 
           ## Plots S-curve for probability of being a candidate triplets vs similarity
           ## It computes simple Jaccard similarity and weighted Jaccard similarity for triplets of songs

------------------------------- SERVER -------------------------------------------------------------------------------------------------------------

runSparkMongo.sh   #### this file was used to run the python files above in the City's server
                   #### runSparkMongo.sh source code was provided by Dr.J. Pauwels 

------------------------------ SOFTWARES ------------------------------------------------------------------------------------------------------------

PuTTy was used as user interface between my PC and the City server to submit the experiments to
run on the server.

FileZilla was used as user interface between my PC and the City server to transfer files and output
and erro file outputs from the experiments.

OpenVPNConnect was used to connect to the server remotely.

----------------------------- Python Packages -----------------------------------------------------------------------------------------------------

Pyplot and python pandas.
Pyspark for Big Data.
Python Notebooks.

----------------------------  GITHUB --------------------------------------------------------------------------------------------------------------

All the ppython files are stored in a repository on GITHUB.

DLSH.py, LSH_SIM_MH.py, WJS_JS.py, TLSH.py, runSparkMongo.sh have the connector to MongoDB credentials removed as personal information of Dr. Johan Pauwels

Python Notebooks for other plots are also stored in the GITHUB repository.
====================================================================================================================================================



