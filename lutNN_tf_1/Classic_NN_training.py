#!/usr/bin/env python
# coding: utf-8

# ## Environment setup

# In[1]:


import glob, os, time
from datetime import datetime
from functools import partial
import importlib

import numpy as np

import tensorflow as tf


# ## Networks definitions and adaptations

# In[2]:


from architecture_definitions import *
 
dir_postfix = get_classic_nn_dir_postfix() 
    
print_Classic_NN()  


# ### Training data set preparation

# In[4]:


import io_functions as io
importlib.reload(io)

batchSize = 4096
nEpochs = 1

trainDataDir = "/scratch_ssd/akalinow/ProgrammingProjects/MachineLearning/OMTF/data/18_12_2020/"   
trainFileNames = glob.glob(trainDataDir+'OMTFHits_pats0x0003_oldSample_files_*_chunk_0.tfrecord.gzip')

dataset = io.get_Classic_NN_dataset(batchSize, nEpochs, trainFileNames, isTrain=True)


# ### Model definition

# In[5]:


import model_functions as models
importlib.reload(models)

import io_functions as io
importlib.reload(io)

networkInputSize = 2 * np.sum(io.getFeaturesMask()) + 1
loss_fn = 'mae'

model = models.get_Classic_NN(networkInputSize=networkInputSize, loss_fn=loss_fn)
model.summary()


# ### The training loop

# In[8]:


from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


# In[15]:


current_time = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
print("Training start. Current Time =", current_time)

nEpochs = 1

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




