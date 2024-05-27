#!/usr/bin/env python
# coding: utf-8

# ## Environment setup

# In[1]:


import glob, os
from functools import partial
import importlib

import tensorflow as tf


# ### Test data set preparation

# In[2]:


import io_functions as io
importlib.reload(io)

from architecture_definitions import *

batchSize = 4096
nEpochs=1

testDataDir = "/scratch_ssd/akalinow/ProgrammingProjects/MachineLearning/OMTF/data/18_12_2020/" 
testFileNames = glob.glob(testDataDir+'OMTFHits_pats0x0003_newerSample_files_1_100_chunk_0.tfrecord.gzip')

dataset = io.get_Classic_NN_dataset(batchSize, nEpochs, testFileNames, isTrain=False)


# ### Load selected model version

# In[3]:


import utility_functions as utils
importlib.reload(utils)

trainingSet = utils.getLatestModelPath(pattern="classic")
print("Using training set:",trainingSet)

job_dir = "results/"+trainingSet
os.makedirs(job_dir, exist_ok=True)

plot_dir = job_dir + "/figures"
os.makedirs(plot_dir, exist_ok=True)

checkpoint_path = "training/"+trainingSet

model = tf.keras.models.load_model(checkpoint_path)
model.summary()


# ### Run the model and put the result into Pandas DataFrame

# In[4]:


import utility_functions as utils
importlib.reload(utils)

import plotting_functions as plf
importlib.reload(plf)

modelAnswerPostProc = partial(utils.classicNNAnswerPostProc, output_type=output_type)

df = utils.df

for aBatch in dataset.take(10): 
    df = utils.fillPandasDataset(aBatch, df, model, modelAnswerPostProc)  
    
ptToPtCalibNN, xedges = plf.ptRecalibration(df, plot_dir, oneOverPt, "NN_pt")
df["NN_pt_recalib"] = plf.ptRecalibrated(df["NN_pt"], ptToPtCalibNN, xedges)     


# ### Make the plots

# In[5]:


import plotting_functions as plf
importlib.reload(plf)

import utility_functions as utils
importlib.reload(utils)

fileName = job_dir+"/performance.txt"
utils.calculateLossFunctions(df, fileName)

plf.plotPtGenPtRec(df, plot_dir, oneOverPt)

for ptCut in plf.ptCuts :
    effStr, _, _ = plf.plotTurnOn(df, ptCut=ptCut, qualityCut = plf.qualityCut, plotDir=plot_dir)
    utils.appendStringToFile(effStr, fileName)
    print(effStr)

ratesStr = plf.plotRate(df, plf.qualityCut, plotDir=plot_dir)
print("rates NN", ratesStr)

utils.appendStringToFile(ratesStr, fileName)


# In[ ]:




