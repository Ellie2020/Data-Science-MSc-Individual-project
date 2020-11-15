#importing main libraries 
import sys
import socket
import os
import time
import psutil
import random
from random import randrange
import builtins
import functools
import operator
import itertools 

import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
#from pyspark.sql.functions import sum as Fsum
from pyspark.sql import Row
from sparkaid import flatten

import pyspark.sql.functions as F
from pyspark.sql.functions import array, udf, col, lit, mean, min, max, concat, desc, row_number, monotonically_increasing_id
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, VectorUDT, _convert_to_vector
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id

from pyspark.sql.window import Window
import scipy.sparse

from pyspark.sql.types import ArrayType, StringType, DoubleType, IntegerType, FloatType
import pyspark.sql.functions as f
import pyspark.sql.types as T
import pyspark

import numpy as np
import pandas as pd
from matplotlib import pyplot

import builtins

##### input variables ####
songs=1000
num_minhashfuncs=600   ## enter number of MinHash functions here
num_bands=100          ## enter number of LSH bands here
num_LSHfuncs=3         ## enter number of LSH functions here
num_rows=(num_minhashfuncs/num_bands) ## given the inputs above get the number of rows per LSH band
# input parameters for MinHash and LSH function here

p=338563   ### large prime number for MinHash function
m=20117    ### large prime number for MinHash function
m1=20117   ### large prime number for LSH function

###########################################################
## Connector to the MongoDB database owned by Dr Johan Pauwels. 
## The connector was provided by Dr Johan Pauwels
sc = pyspark.SparkContext.getOrCreate()
spark = pyspark.sql.SparkSession.builder \
    .config("xxxxxxxxxxxxxxxxxxxxxxxxx", xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)\
    .getOrCreate()

####################### similarity threshold ################

def SIM_threshold(number_bands, number_minhashfunctions):
    '''Given number of bands and length of MinHash signature
	   it returns the optimal similarity threshold t
	'''
    num_rows=(number_minhashfunctions/number_bands)   ## rows per band
    return np.round((1/number_bands)**(1/num_rows), 3)

num_rows=int(num_minhashfuncs/num_bands)

#set your threshold
#threshold=0.478    #comment out if you want to set a sim threshold that is different from the optimal one

threshold=SIM_threshold(num_bands, num_minhashfuncs)  #comment out to use optimum theshold
print(f"LSH SIMILARITY THRESHOLD {threshold} for {num_bands} bands and {num_rows} rows per band and {songs} songs")

###########################################################################################

df = spark.read.format("mongo").option('database', 'jamendo').option('collection', 'chords').load()
sample_300_df=df.select(["_id","chordRatio"]).limit(songs)

myFunc = f.udf(lambda array_to_list: [int(0) if e is None else int(1) for e in array_to_list], T.ArrayType(T.IntegerType()))
sample_300_df2=sample_300_df.withColumn('chordRatioMinHash', myFunc('chordRatio'))  
df0=sample_300_df2.select(["_id", "chordRatio", "chordRatioMinHash"])

from pyspark.sql import Row
from pyspark.sql.functions import col
from sparkaid import flatten
df0_flat=flatten(df0)
columns_list1=df0_flat.columns[1:-1]
array_df=df0_flat.select('_id', 'chordRatioMinHash',array(columns_list1).alias('chordRatioJS'))

#fill NaNs with zeros in the array column
df2_flat=df0_flat.na.fill(float(0))
columns_list2=df2_flat.columns[1:-1]
array_df2=df2_flat.select('_id', 'chordRatioMinHash',array(columns_list2).alias('chordRatioJS_no_Nulls'))

###
to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
data = array_df2.select('_id', 'chordRatioMinHash', "chordRatioJS_no_Nulls", to_vector("chordRatioJS_no_Nulls").alias("chordRatioWJS"))
data.show(1, truncate=False)

import scipy.sparse
from pyspark.ml.linalg import Vectors, _convert_to_vector, VectorUDT
from pyspark.sql.functions import udf, col
## from dense to sparse array
def dense_to_sparse(vector):
    return _convert_to_vector(scipy.sparse.csc_matrix(vector.toArray()).T)

to_sparse = udf(dense_to_sparse, VectorUDT())
data_sparse=data.withColumn("sparseChordRatioJS", to_sparse(col("chordRatioWJS")))
#data_sparse2=data_sparse.select('_id', 'chordRatio_for_minHash', 'sparseChordRatioJS')

indices_udf = udf(lambda vector: vector.indices.tolist(), ArrayType(IntegerType()))
values_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
data_sparse3=data_sparse.withColumn('indicesJS', indices_udf(F.col('sparseChordRatioJS')))\
.withColumn('values', values_udf(F.col('sparseChordRatioJS')))

## renaming columns for the following steps
simil_DF=data_sparse3.select('_id', 'chordRatioMinHash', 'chordRatioJS_no_Nulls','indicesJS')
simil_DF = simil_DF.select(col("_id").alias("_id"), 
                           col("chordRatioMinHash").alias("chordRatioMinHash"),
                           col("chordRatioJS_no_Nulls").alias("chordRatioWeightJS"),
                           col("indicesJS").alias("indicesJS"))

minHashDF1=simil_DF.select('_id', 'chordRatioMinHash', "chordRatioWeightJS", "indicesJS"). \
withColumnRenamed('_id','id1'). \
withColumnRenamed('chordRatioMinHash', 'chordRatioMinHash1'). \
withColumnRenamed("chordRatioWeightJS","chordRatioWeightJS1"). \
withColumnRenamed("indicesJS", "indicesJS1")

### From Pyspark DataFrame with chordRatioMinHash to Matrix 2Darray 
data1=np.array(minHashDF1.select('chordRatioMinHash1').collect())
#print('rows: '+str(len(data1)), 'columns: '+str(len(data1[0])))
flattened = np.array([val.tolist() for sublist in data1 for val in sublist])

### MinHash function generator 
def make_random_hash_fn():
    a =randrange(1, p-1, 2) # odd random numbers
    b =randrange(0, p-1)    #all numbers even and odd
    if a!=b:
        return lambda x: ((a * x + b) % p) % m
### N MinHash functions generator
def generate_N_hashFunctions(N):
    hashfuncs=[make_random_hash_fn() for i in range(N)]
    return hashfuncs

### minhash the matrix  
### Adapted from ref https://www.bogotobogo.com/Algorithms/minHash_Jaccard_Similarity_Locality_sensitive_hashing_LSH.php
def minhash1(data, hashfuncs):
    '''
    Returns the minhash signature matrix for the songs using a number of 
    hash functions randomly generated
    '''
    '''songs x hashfuncs matrix with values=infinite to start with'''
    sigmat1= [[sys.maxsize for x in range(len(hashfuncs))] for x in range(len(data))]  
    
    for c in range(len(data[0])):
        hashvalue = list(map(lambda x: x(c), hashfuncs))
        
        for r in range(len(data)):
            if data[r][c] == 0:
                continue
            for i in range(len(hashfuncs)):
                if sigmat1[r][i] > hashvalue[i]:
                    sigmat1[r][i] = hashvalue[i]

    return sigmat1

### To measure time and memory comment off
#import time
#import os
#import psutil
#tt0=time.time()

hash_fnc_list1=generate_N_hashFunctions(num_minhashfuncs)
sign_matrix_rf=minhash1(flattened , hash_fnc_list1)
#print('rows: '+str(len(sign_matrix_rf)), 'columns: '+str(len(sign_matrix_rf[0])))

signatures_df=spark.createDataFrame(sign_matrix_rf)
signatures_df.columns

#from minhash signature matrix to pyspark dataframe
def columns_to_array_column(N):    
    '''N MH functions'''
    return array([col("_"+str(N)) for N in range(1, N+1)])
    
a=columns_to_array_column(num_minhashfuncs)
signatures_df2 = signatures_df.withColumn("signature1", a)
signatures_df3=signatures_df2.select('signature1')

# joining the minhash dataframe with the mean one
df1 = minHashDF1. \
withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)

df2 = signatures_df3. \
withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)

df3=df1.join(df2,on='id',how='left').orderBy('id',ascending=True)  #this is correct way to do it
df3=df3.drop('id')

### To measure time and memory comment off
#df3.collect()
#process = psutil.Process(os.getpid())
# memory=process.memory_info().rss # in bytes
#time_MH=time.time()-tt0
#print("Total time to create Min Hash signatures: "+str(time.time()-tt0) + " secs;", "Memory: "+str(memory*0.000001) +" Mb")
#print("Total time to create Min Hash signatures: "+str(time.time()-tt0) + " secs;")

### Locality Sensitive Hashing
df4=df3.select('id1', 'signature1')

### Get the buckets for LSH
def get_LHS_buckets(list1, number_bands, number_hashfuncs, P, M):
    #random.seed(1234) 
    #items_per_band=int(number_hashfuncs/number_bands)
    MYRDN=random.Random(13)
    items_per_band=int(len(list1)/number_bands)
    coeff=[[MYRDN.randrange(1, P-1, 2) for _ in range(items_per_band)] for _ in range(number_hashfuncs)]  ###OK
    list_of_tuples=list(zip(*([iter(list1)] * items_per_band)))
    polynomial_results=[[builtins.sum(x*c for x, c in zip(t, cf))%m] for t in list_of_tuples for cf in coeff] 
    return polynomial_results

### To measure time and memory comment off
#import time
#import os
#import psutil
#tt0=time.time()

### LSH UDF
get_LHS_buckets_udf = udf(lambda x, y, w, z1, z2: get_LHS_buckets(x, y, w, z1, z2), ArrayType(ArrayType(IntegerType())))
# these are the parameters for get_LHS_buckets(list1, number_bands, number_hashfuncs, P, M) 
df4=df3. \
withColumn('LSH1', get_LHS_buckets_udf("signature1", lit(num_bands), lit(num_LSHfuncs), lit(p), lit(m1)))

### To measure time and memory comment off

#df4.collect()
# process = psutil.Process(os.getpid())
# memory=process.memory_info().rss # in bytes
#time_LSH_sign=time.time()-tt0
#print("Total time to create LSH signatures: "+str(time.time()-tt0) + " secs;", "Memory: "+str(memory*0.000001) +" Mb")
#print("Total time to create LSH signatures: "+str(time.time()-tt0) + " secs;")

LSH_DF1=df4

LSH_DF2=df4.select('*'). \
withColumnRenamed('id1','id2'). \
withColumnRenamed('chordRatioMinHash1','chordRatioMinHash2'). \
withColumnRenamed('chordRatioWeightJS1','chordRatioWeightJS2'). \
withColumnRenamed('indicesJS1', 'indicesJS2'). \
withColumnRenamed('signature1', 'signature2'). \
withColumnRenamed("LSH1","LSH2")

# crossjoin without identities and doubles with swapped ids
spark.conf.set("spark.sql.crossJoin.enabled", True)

### Comment off to cache and reuse the DataFrame
#crossjoin_minHashDF=minHashDF1.join(minHashDF2,minHashDF1.id1>minHashDF2.id2).cache()
crossjoin_LSH_DF=LSH_DF1.join(LSH_DF2,LSH_DF1.id1>LSH_DF2.id2). \
select('id1', 'id2', 'chordRatioMinHash1','chordRatioMinHash2',
       'chordRatioWeightJS1','chordRatioWeightJS2',
       'indicesJS1', 'indicesJS2', 'LSH1', 'LSH2', 
       'signature1', 'signature2')
	   
# comment off to print number of columns and rows of the Pyspark DataFrame
#print((crossjoin_LSH_DF.count(), len(crossjoin_LSH_DF.columns)))

### LSH candidate pairs
#minhash similarity function
import builtins
import time
from pyspark.sql.functions import size

##If the signature values of two documents agree in all rows of at least one band
##then these two documents are likely to be similar 

import builtins
def LHSSimilarpairs(l1,l2,n):
    ''' The function compare the bands of two MinHash signature and return the 
        number of buckets for a pair of songs
	'''
    L3=[1 if x==y else 0 for s1, s2 in zip(l1,l2) for x, y in zip(s1,s2)]
    L4=[builtins.sum(L3[i:i+n]) for i in range(0, len(L3),n)]
    LSH=len([x for x in L4 if x>=n])
    return LSH
    


#minhash similarity udf
num_rows=int(num_minhashfuncs/num_bands)
LHSSimilarity_udf = udf(lambda x, y, z: LHSSimilarpairs(x, y, z), IntegerType())

df5=crossjoin_LSH_DF. \
withColumn('SHARED_BUCKETS', LHSSimilarity_udf('LSH1', 'LSH2', lit(num_rows)))

## MinHash similarity
import builtins
def minHashSimilarity(m1, m2):
    '''The function takes two MinHash signatures m1 and m2 and returns 
	   the approximate MinHash similarity
	'''
    size=len(m1)
    if size != len(m2): raise Exception("Signature lengths do not match")
    if size == 0: raise Exception("Signature length is zero")
    if size == len(m2):
        try:
            count=builtins.sum(1 for x, y in zip(m1, m2) if x==y)   #ok in one liner
            SIM=count/size

        except ZeroDivisionError:
            pass

        #print('elapsed'.format(elapsed))
        return SIM

#minhash similarity udf
minHashSimilarity_udf = udf(lambda x, y: minHashSimilarity(x, y), FloatType())

def jaccard_similarity_handling_zero_error(list1, list2):
    ''' This functions takes two arrays list1 and list2 
	    and returns the simple Jaccard similarity '''
    intersection = len(set.intersection(set(list1), set(list2)))
    union = len(set.union(set(list1), set(list2)))
    try:
        sim = intersection/union
    except ZeroDivisionError:
        #return 0.0
        pass
    return sim

#Jaccard similarity UDF function
jaccard_HZE_udf=udf(lambda x, y: jaccard_similarity_handling_zero_error(x, y), FloatType())

#updating table df6 to df7 with the results
df11=df5.withColumn('MH_SIM', minHashSimilarity_udf('signature1', 'signature2')). \
     withColumn('JS', jaccard_HZE_udf('indicesJS1', 'indicesJS2')). \
	 select('id1', 'id2', 'SHARED_BUCKETS', 'JS', 'MH_SIM').cache()

##### Metrics for LSH with respect to SIM(S1,S2) for a given similarity threshold t

FN_DF=df11.filter((df11.SHARED_BUCKETS==0) & (df11.MH_SIM>=threshold))
FP_DF=df11.filter((df11.SHARED_BUCKETS!=0) & (df11.MH_SIM<threshold))
TP_DF=df11.filter((df11.SHARED_BUCKETS!=0) & (df11.MH_SIM>=threshold))
TN_DF=df11.filter((df11.SHARED_BUCKETS==0) & (df11.MH_SIM<threshold))


FN=df11.filter((df11.SHARED_BUCKETS==0) & (df11.MH_SIM>=threshold)).count()
FP=df11.filter((df11.SHARED_BUCKETS!=0) & (df11.MH_SIM<threshold)).count()
TP=df11.filter((df11.SHARED_BUCKETS!=0) & (df11.MH_SIM>=threshold)).count()
TN=df11.filter((df11.SHARED_BUCKETS==0) & (df11.MH_SIM<threshold)).count()

## Metrics for SIM(S1,S2) with respect to ground truth J(S1,S2) for a given similarity threshold t

MH_TP=df11.filter((df11.JS>=threshold) & (df11.MH_SIM>=threshold)).count()
MH_TN=df11.filter((df11.JS<threshold) & (df11.MH_SIM<threshold)).count()
MH_FP=df11.filter((df11.JS<threshold) & (df11.MH_SIM>=threshold)).count()
MH_FN=df11.filter((df11.JS>=threshold) & (df11.MH_SIM<threshold)).count()


print(f"retrieved {songs} songs; {num_minhashfuncs} minhash functions; {num_bands} LSH bands; {num_LSHfuncs} LSH function")


def ACCURACY(t_pos, t_neg, f_pos, f_neg):
   ''' function that takes FP, FN, TP and TN
       and returns accuracy. 
	   It handles ZeroDivisionError
   '''
    try:
        return (t_pos+t_neg)/(t_pos+f_pos+t_neg+f_neg)
    except ZeroDivisionError:
        pass

lsh_accuracy=ACCURACY(TP,TN,FP,FN)
mh_accuracy=ACCURACY(MH_TP, MH_TN, MH_FP, MH_FN)

def PRECISION(t_pos, f_pos):
    ''' function that takes FP, TP
       and returns precision. 
	   It handles ZeroDivisionError
    '''
    try:
        return t_pos/(t_pos+f_pos)
    except ZeroDivisionError:
        pass

lsh_precision=PRECISION(TP, FP)
mh_precision=PRECISION(MH_TP, MH_FP)

def RECALL(t_pos, f_neg):
    ''' function that takes FP, FN
       and returns recall. 
	   It handles ZeroDivisionError
    '''
    try:
        return t_pos/(t_pos+f_neg)
    except ZeroDivisionError:
        pass

lsh_recall=RECALL(TP, FN)
mh_recall=RECALL(MH_TP, MH_FN)

def F1SCORE(recl,precs):
    ''' function that takes recall and precision
       and returns F1-SCORE. 
	   It handles ZeroDivisionError
	   & type None
    '''
    if (recl!=None and precs!=None):
        try:
            return 2*recl*precs/(recl+precs)
        except ZeroDivisionError:
            pass

lsh_F1=F1SCORE(lsh_recall, lsh_precision)
mh_F1=F1SCORE(mh_recall, mh_precision)

###saving metrics in a csv file
### 1. create a dictionary to store results and
metrics={'#songs': songs,'#MHfuncs': num_minhashfuncs,'#LSHfuncs':num_LSHfuncs,
'#bands':num_bands,'mMH':m, 'mLSH':m1,'pMH':p,
'TP':TP, 'TN': TN,'FP':FP, 'FN': FN,'LSH_Accuracy':lsh_accuracy,'LSH_Precision':lsh_precision,
'LSH_Recall':lsh_recall,'LSH_F1':lsh_F1,
'MH_Accuracy': mh_accuracy,'MH_Precision':mh_precision,
'MH_Recall':mh_recall, 'MH_F1':mh_F1,
'threshold':threshold}

### 2.from dictionary to Pandas DataFrame and then metrics exported as csv file
metrics_df = pd.DataFrame([metrics], columns=metrics.keys())
metrics_df.to_csv('./slurm-logs/metrics_LSHfun'+ str(num_LSHfuncs)+
                  '_TP'+str(TP)+
                  '_MHfuncs'+str(num_minhashfuncs)+'_bands'+str(num_bands)+
                  '_threshold'+str(threshold)+'_songs'+str(songs)+'_m'+str(m)+'_p'+str(p)+
				  '_F1'+str(lsh_F1)+
                  '.csv', encoding='utf-8', index=False)
				  
##exporting TP, FP, TN and FN in csv files
TP_DF_csv=TP_DF.toPandas().to_csv('./slurm-logs/TP_LSH'+ str(num_LSHfuncs)+
                  '_TP'+str(TP)+
                  '_MHfuncs'+str(num_minhashfuncs)+'_bands'+str(num_bands)+
                  '_threshold'+str(threshold)+'_songs'+str(songs)+str(m)+
                  '.csv', encoding='utf-8',  index=False)
				  
TN_DF_csv=TN_DF.toPandas().to_csv('./slurm-logs/TN_LSH'+ str(num_LSHfuncs)+
                  '_TP'+str(TN)+
                  '_MHfuncs'+str(num_minhashfuncs)+'_bands'+str(num_bands)+
                  '_threshold'+str(threshold)+'_songs'+str(songs)+str(m)+
                  '.csv', encoding='utf-8',  index=False)
				  
FN_DF_csv=FN_DF.toPandas().to_csv('./slurm-logs/FN_LSH'+ str(num_LSHfuncs)+
                  '_FN'+str(FN)+
                  '_MHfuncs'+str(num_minhashfuncs)+'_bands'+str(num_bands)+
                  '_threshold'+str(threshold)+'_songs'+str(songs)+str(m)+
                  '.csv', encoding='utf-8', index=False)
				  
FP_DF_csv=FP_DF.toPandas().to_csv('./slurm-logs/FP_LSH'+ str(num_LSHfuncs)+
                  '_TP'+str(FP)+
                  '_MHfuncs'+str(num_minhashfuncs)+'_bands'+str(num_bands)+
                  '_threshold'+str(threshold)+'_songs'+str(songs)+str(m)+
                  '.csv', encoding='utf-8',  index=False)

######## S-curve ######
#probability of sharing a bucket
def sharing_buckets_probability(s, r, b):
    '''The function takes similarity values, number of rows and number of bands.
	   It return the probability of  a pair of songs to be candidate pairs
	'''
    probability=1-np.power((1-np.power(s,r)), b)
    return float(probability)

### sharing_buckets_probability UDF 
sharing_buckets_probability_udf=udf(lambda x,y,z: sharing_buckets_probability(x, y, z), FloatType())

### adding a column with the probability values to the Pyspark DataFrame (i.e. df11)
df_prob=df11. \
withColumn('Probability', sharing_buckets_probability_udf('MH_SIM', lit(num_rows),lit(num_bands)))

### from PysparFrame to pandas DataFrame to csv file
df_prob_pandas=df_prob.toPandas().to_csv('./slurm-logs/Probability'+ str(num_LSHfuncs)+
                  '_MHfuncs'+str(num_minhashfuncs)+'_bands'+str(num_bands)+
                  '_threshold'+str(threshold)+'_songs'+str(songs)+str(m)+
                  '.csv', encoding='utf-8',  index=False)
####### Results Visualization  ######
########### S-curve plot ############
df_prob_pandas=df_prob.toPandas()
print(f"{len(df_prob_pandas)} rows")
import matplotlib
from matplotlib.pyplot import figure
figure(figsize=(7,4))
x=df_prob_pandas.MH_SIM
y=df_prob_pandas.Probability

#label on the plot
pyplot.plot(x, y, 'o', color='blue',  label='bands:'+str(num_bands)+'; songs:'+str(songs)+
            '; rows:'+str(int(num_rows))+ '; MHfuncs:'+str(num_minhashfuncs)+
            '; LSHfuncs:'+str(num_LSHfuncs))

pyplot.plot(x, y, 'o', color='blue',  label='b:'+str(num_bands)+' MH:'+str(num_minhashfuncs))
            
pyplot.axvline(threshold, color='r', linestyle='-')
pyplot.legend(loc='upper left')
pyplot.legend(loc=(0.7, 1.05), ncol=2)
pyplot.xlabel("Similarity")
pyplot.ylabel("Probability of being candidate pair")
matplotlib.pyplot.savefig('./slurm-logs/probability_bin_sharing_num_rows'+str(int(num_rows))+'_songs'+str(songs)+
                          '_bands'+str(num_bands)+'_MHfuncs'+str(num_minhashfuncs)+
                          '_LSHfuncs'+str(num_LSHfuncs)+'m'+str(m)+'.png')

pyplot.show()

############# Scatter plots ############
import numpy as np
from matplotlib import pyplot

df11_plot=df11.select('id1', 'id2', 'MH_SIM', 'SHARED_BUCKETS', 'JS')
df11_pandas=df11_plot.toPandas()
z1=df11_pandas.MH_SIM
z2=df11_pandas.SHARED_BUCKETS
z3=df11_pandas.JS

f = pyplot.figure(figsize=(13,5))

pyplot.subplot(1,3,1)

pyplot.plot(z2, z1, 'o', color='r', label='Shared bins vs MHS')
pyplot.legend(loc='upper right')
pyplot.xlabel("LSH: Number of shared bins")
pyplot.ylabel("MH similarity")
pyplot.axhline(y=threshold, color='b', linestyle='-')

pyplot.subplot(1,3,2)

pyplot.plot(z2, z3, 'o', color='g',  label='Shared bins vs JS')
pyplot.legend(loc='upper right')
pyplot.xlabel("LSH: Number of shared bins")
pyplot.ylabel("JS similarity")

pyplot.axhline(y=threshold, color='r', linestyle='-')

############# histograms JS and MHS ############
pyplot.subplot(1,3,3)
bins=np.linspace(0,1,100)
pyplot.hist(z3,bins,alpha=0.5,color='g',label='JS')
pyplot.hist(z1,bins,alpha=0.5,color='r',label='MH_SIM')
pyplot.legend(loc='upper right')
pyplot.xlabel('similarity')
pyplot.ylabel('count')

pyplot.tight_layout()

### saving the plot as png image
matplotlib.pyplot.savefig('./slurm-logs/scatterplot_hists_with_rows'+str(int(num_rows))+'_songs'+str(songs)+
                          '_bands'+str(num_bands)+'_MHfuncs'+str(num_minhashfuncs)+
                          '_LSHfuncs'+str(num_LSHfuncs)+'m'+str(m)+'.png')
pyplot.show()

############# histograms JS and MHS
print('TASKS DONE AND DUSTED')
spark.stop()
