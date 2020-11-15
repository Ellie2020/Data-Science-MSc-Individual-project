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
songs=1000             ## enter number of songs to compare
num_minhashfuncs=600   ## enter number of MinHash functions here
num_bands=100          ## enter number of LSH bands here
num_LSHfuncs=3         ## enter number of LSH functions here
num_rows=(num_minhashfuncs/num_bands) ## given the inputs above get the number of rows per LSH band
# input parameters for MinHash and LSH function here

p=338563   ### large prime number for MinHash function
m=20117    ### large prime number for MinHash function
m1=20117   ### large prime number for LSH function

########################################
## Connector to the MongoDB database owned by Dr. Johan Pauwels. 
## The connector was provided by Dr. Johan Pauwels
sc = pyspark.SparkContext.getOrCreate()
spark = pyspark.sql.SparkSession.builder \
    .config("xxxxxxxxxxxxxxxxxxxxxxxxx", xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)\
    .getOrCreate()

####################### similarity theshold ###############################################

def SIM_threshold(number_bands, number_minhashfunctions):
    '''Given number of bands and lenght of MinHash signature
	   it returns the optimal similarity thresold t
	'''
    num_rows=(number_minhashfunctions/number_bands)   ## rows per band
    return np.round((1/number_bands)**(1/num_rows), 3)

num_rows=int(num_minhashfuncs/num_bands)

#set your threshold
#threshold=0.478    #comment out if you want to set a similair threshold that is different from the optimal one

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

to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
data = array_df2.select('_id', 'chordRatioMinHash', "chordRatioJS_no_Nulls", to_vector("chordRatioJS_no_Nulls").alias("chordRatioWJS"))


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
                           col("chordRatioMinHash").alias("chordRatioMinHash"))

minHashDF1=simil_DF.select('_id', 'chordRatioMinHash'). \
withColumnRenamed('_id','id1'). \
withColumnRenamed('chordRatioMinHash', 'chordRatioMinHash1')

data1=np.array(minHashDF1.select('chordRatioMinHash1').collect())
#print('rows: '+str(len(data1)), 'columns: '+str(len(data1[0])))
flattened = np.array([val.tolist() for sublist in data1 for val in sublist])

def make_random_hash_fn():
    a =randrange(1, p-1, 2) #odd numbers
    b =randrange(0, p-1)  #all numbers even and odd
    if a!=b:
        return lambda x: (a * x + b) % m 

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

hash_fnc_list1=generate_N_hashFunctions(num_minhashfuncs)
sign_matrix_rf=minhash1(flattened , hash_fnc_list1)

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

##LSH
#df4=df3.select('id1', 'signature1')
df3=df3.drop('id')

#  NB items_per_band=int(N_hashfuncs/N_bands) needs to be defined before the get_LHS_buckets
# m is a large prime number 1.3 x hashtable

def get_LHS_buckets(list1, number_bands, number_hashfuncs, P, M):
    #random.seed(1234) 
    #items_per_band=int(number_hashfuncs/number_bands)
    MYRDN=random.Random(13)
    items_per_band=int(len(list1)/number_bands)
    coeff=[[MYRDN.randrange(1, P-1, 2) for _ in range(items_per_band)] for _ in range(number_hashfuncs)]  ###OK
    list_of_tuples=list(zip(*([iter(list1)] * items_per_band)))
    polynomial_results=[[builtins.sum(x*c for x, c in zip(t, cf))%m] for t in list_of_tuples for cf in coeff] 
    return polynomial_results


##LSH UDF
get_LHS_buckets_udf = udf(lambda x, y, w, z1, z2: get_LHS_buckets(x, y, w, z1, z2), ArrayType(ArrayType(IntegerType())))
# these are the parameters for get_LHS_buckets(list1, number_bands, number_hashfuncs, P, M) 
df4=df3. \
withColumn('LSH1', get_LHS_buckets_udf("signature1", lit(num_bands), lit(num_LSHfuncs), lit(p), lit(m1)))

LSH_DF1=df4

LSH_DF2=df4.select('*'). \
withColumnRenamed('signature1','signature2'). \
withColumnRenamed('id1','id2'). \
withColumnRenamed("LSH1","LSH2")

# crossjoin without doubles and swapped ids
spark.conf.set("spark.sql.crossJoin.enabled", True)
crossjoin_LSH_DF=LSH_DF1.join(LSH_DF2,LSH_DF1.id1>LSH_DF2.id2)
#select('id1', 'id2','LSH1', 'LSH2')

### LSH candidate pairs
#minhash similarity function
import builtins
import time
from pyspark.sql.functions import size

import builtins
def LHSSimilarpairs(l1,l2,n):
    L3=[1 if x==y else 0 for s1, s2 in zip(l1,l2) for x, y in zip(s1,s2)]
    L4=[builtins.sum(L3[i:i+n]) for i in range(0, len(L3),n)]
    LSH=len([x for x in L4 if x>=n])
    return LSH

#minhash similarity udf
num_rows=int(num_minhashfuncs/num_bands)
LHSSimilarity_udf = udf(lambda x, y, z: LHSSimilarpairs(x, y, z), IntegerType())

df5=crossjoin_LSH_DF. \
withColumn('SHARED_BUCKETS', LHSSimilarity_udf('LSH1', 'LSH2', lit(num_rows)))

#num_candidates=df5.filter(df5.SHARED_BUCKETS>0).count()
df6=df5.filter(df5.SHARED_BUCKETS>0)
df6.show()


## computing the MinHash similarity for only the candidate pairs obtained from LSH banding technique
import builtins
def minHashSimilarity(m1, m2):
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

import time
import os
import psutil
tt0=time.time()

df7=df6.withColumn('MH_SIM', minHashSimilarity_udf('signature1', 'signature2')).collect()

process = psutil.Process(os.getpid())
memory=process.memory_info().rss # in bytes
time_LSH_sign=time.time()-tt0
print("Total time to calculate MinHash SIM after LSH: "+str(time.time()-tt0) + " secs;", "Memory: "+str(memory*0.000001) +" Mb")
print("Total time to calculate MinHash SIM after LSH: "+str(time.time()-tt0) + " secs;")
			  
print('TASKS DONE AND DUSTED')
spark.stop()
