----------------------------- SONG PAIRS -------------------------------------------------------------------------------------------------------------

DLSH.py 

This is the main python file for similarity songs pairs it computes Jaccard similarity, MinHash similarity, LSH banding technique. It computes also metrics for MinHash similarity vs Jaccard similarity and  TP, TN, FP, FN for LSH. Plots S-curve for probability of being a candidate pairs vs similarity

LSH_SIM_MH.py 
AIM: measure time and memory of similarity search by LSH via Minhashing
It is a shorter version of DLSH.py as it does not compute plots, Jaccard similarity, metrics and csv files. It computes Minhash signatures, LSH candidate pairs and MinHash similarity for those pairs which are candidate pairs. 


WJS_JS.py  
source code for the computation of Jaccard similarity and weighted Jaccard similarity for songs PAIRS.
It computes also the superimposed histograms of Jaccard and weighted Jaccard similarity.
It computes the statistics for Jaccard and weighted Jaccard similarity.


----------------------------- SONG TRIPLETS ------------------------------------------------------------------------------------------------------------------------------
TLSH.py   
This is the main source code for TRIPLETS, computes Jaccard similarity, MinHash similarity, LSH banding technique, metrics for MinHash similarity vs Jaccard similarity. It computes TP, TN, FP, FN for LSH, S-curve for probability of being a candidate triplets vs MinHash similarity. It computes simple Jaccard similarity and weighted Jaccard similarity for triplets of songs

------------------------------- SERVER -----------------------------------------------------------------------------------------------------------------------------------

runSparkMongo.sh   
file was used to run the python/Pyspark files (above) in the City's server
runSparkMongo.sh source code was provided by Dr J. Pauwels 

------------------------------ SOFTWARES ----------------------------------------------------------------------------------------------------------------------------------
PuTTy was used as user interface between my PC and the City server to submit the experiments.

FileZilla was used as user interface between my PC and the City server to transfer files, code generated png, csv and output text documents.

OpenVPNConnect was used to connect to the server remotely.

----------------------------- Python Packages -------------------------------------------------------------------------------------------------------------------------------
Pyplot and python pandas.
Pyspark for Big Data.
Python Notebooks.

=======================================================================

