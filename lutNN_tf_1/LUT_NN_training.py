#!/usr/bin/env python
# coding: utf-8

# ## Environment setup

# In[1]:


import glob, os, time
from datetime import datetime
import importlib

import numpy as np

import tensorflow as tf


# ## Networks definitions and adaptations

# In[2]:


from architecture_definitions import *

oneOverPt = False 
lut_nn = True
output_type = 0
last_input_is_bias = True

if output_type == 1:
    layer3_neurons = 3
    loss_fn = custom_loss3
else: 
    output_cnt = 1
    layer3_neurons = 1
    loss_fn = 'mae'
        
if not last_input_is_bias:
    networkInputSize =  nLayers
    layer2_lutRangesCnt = 1
    layer2_input_offset = None 
 
dir_postfix = get_lut_nn_dir_postfix() 
    
print_LUT_NN()  


# ### Training data set preparation

# In[3]:


import io_functions as io
importlib.reload(io)

batchSize = 4096
nEpochs = 1

trainDataDir = "/scratch_ssd/akalinow/ProgrammingProjects/MachineLearning/OMTF/data/18_12_2020/"   
trainFileNames = glob.glob(trainDataDir+'OMTFHits_pats0x0003_oldSample_files_*_chunk_0.tfrecord.gzip')

dataset = io.get_LUT_NN_dataset(batchSize, nEpochs, trainFileNames, 
                                nRefLayers=nRefLayers,
                                layer1_lut_size=layer1_lut_size,
                                layer2_lut_size=layer2_lut_size,
                                layer2_lutRangesCnt=layer2_lutRangesCnt,
                                last_input_is_bias=last_input_is_bias,
                                rangeFactor=rangeFactor,
                                isTrain=True)


# ### Model definition

# In[4]:


import model_functions as models
importlib.reload(models)

model = models.get_LUT_NN(last_input_is_bias=last_input_is_bias, loss_fn=loss_fn)
model.summary()


# ### The training loop

# In[5]:


current_time = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
print("Training start. Current Time =", current_time)

nEpochs = 10

log_dir = "logs/fit/" + current_time + dir_postfix
job_dir = "training/" + current_time + dir_postfix

checkpoint_path = job_dir + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq = 5085)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=(10, 20))

model.save_weights(checkpoint_path.format(epoch=0))
   
model.fit(dataset, epochs=nEpochs, shuffle=True,
            callbacks=[tensorboard_callback, cp_callback]
            )
model.save(job_dir, save_format='tf')

current_time = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
print("Training end. Current Time =", current_time)


# In[ ]:




