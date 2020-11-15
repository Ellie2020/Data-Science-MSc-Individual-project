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
songs=10000 ### enter number of songs to compare (pairwise)

########################################
## MongoDB database is owned by Dr Johan Pauwels. 
## The connector was provided by Dr Johan Pauwels
sc = pyspark.SparkContext.getOrCreate()
spark = pyspark.sql.SparkSession.builder \
    .config("xxxxxxxxxxxxxxxxxxxxxxxxx", xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)\
    .getOrCreate()

###########################################################################################

df0 = spark.read.format("mongo").option('database', 'jamendo').option('collection', 'chords').load()
sample_300_df=df0.select(["_id","chordRatio"]).limit(songs)

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
simil_DF=data_sparse3.select('_id', 'chordRatioJS_no_Nulls','indicesJS')
simil_DF = simil_DF.select(col("_id").alias("_id"), 
                           col("chordRatioJS_no_Nulls").alias("chordRatioWeightJS"),
                           col("indicesJS").alias("indicesJS"))

SIMDF1=simil_DF.select('_id', "chordRatioWeightJS", "indicesJS"). \
withColumnRenamed('_id','id1'). \
withColumnRenamed("chordRatioWeightJS","chordRatioWeightJS1"). \
withColumnRenamed("indicesJS", "indicesJS1")


SIMDF2=simil_DF.select('_id', "chordRatioWeightJS", "indicesJS"). \
withColumnRenamed('_id','id2'). \
withColumnRenamed("chordRatioWeightJS","chordRatioWeightJS2"). \
withColumnRenamed("indicesJS", "indicesJS2")

# joining SIMDF1 and SIMDF2
spark.conf.set("spark.sql.crossJoin.enabled", True)
crossjoinDF=SIMDF1.join(SIMDF2,SIMDF1.id1>SIMDF2.id2)
crossjoinDF.show()

#Jaccard similarity
def jaccard_similarity_handling_zero_error(list1, list2):
	''' This functions takes two arrays list1 and list2 
	and returns the simple Jaccard similarity '''
    intersection = len(set.intersection(set(list1), set(list2)))
    union = len(set.union(set(list1), set(list2)))
    try:
        sim = intersection/union
    except ZeroDivisionError:
        pass
    return sim

#Jaccard similarity UDF 
jaccard_HZE_udf=udf(lambda x, y: jaccard_similarity_handling_zero_error(x, y), FloatType())

df=crossjoinDF.withColumn('JS', jaccard_HZE_udf('indicesJS1', 'indicesJS2'))  ##comment out for checking just JS

import builtins
def weighted_JS(l1, l2):
	''' This functions takes two arrays list1 and list2 
	and returns the weighted Jaccard similarity '''

    A=builtins.sum([builtins.min(x, y) for x, y in zip(l1, l2)])
    zip_list=zip(l1, l2)
    B=builtins.sum([builtins.max(x, y) for x, y in zip(l1, l2)])
    try:
        WJS=A/B
    except ZeroDivisionError:
        #return 0.0
        pass
    return WJS     


weighted_JS_UDF= F.udf(lambda x, y: weighted_JS(x, y) , DoubleType())

df=crossjoinDF.withColumn('WJS', weighted_JS_UDF('chordRatioWeightJS1', 'chordRatioWeightJS2'))  ##comment out for checking jusr WJS

#################  Visualization  ##################
############### histograms JS and WJS ############
df_pandas=df.toPandas()
z1=df_pandas.JS
z2=df_pandas.WJS

import matplotlib
from matplotlib import pyplot
from matplotlib.pyplot import figure


bins=np.linspace(0,1,100)
pyplot.hist(z1,bins,alpha=0.5,color='g',label='JS')
pyplot.hist(z2,bins,alpha=0.5,color='r',label='WEIGHTED_SIM')
pyplot.legend(loc='upper right')
pyplot.xlabel('similarity')
pyplot.ylabel('count')

matplotlib.pyplot.savefig('./slurm-logs/hists_JS_WJS_songs'+str(songs)+'.png')
pyplot.show()

df.describe('JS', 'WJS').show()



df.describe('JS').show() ##comment our if checking statistics for JS

df.describe('WJS').show() ##comment out if checking statistics for WJS

print('TASKS DONE AND DUSTED')
spark.stop()
