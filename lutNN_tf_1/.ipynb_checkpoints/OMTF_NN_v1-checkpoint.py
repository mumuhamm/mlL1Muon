

import glob

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

import os
import time
import shutil

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
#import scipy
#from termcolor import colored

import pandas as pd

import LutInterLayer
import plotting_functions as plf

inputDataPrefix = ""

train = True
#train = False

train_sample = "newerSample"

oneOverPt = False 

lut_nn = True
#lut_nn = False
def custom_loss3(y_true, y_pred) :
    loss = y_true[ : , 2] * tf.keras.losses.MAE(y_true[ : , 0], y_pred[ : , 0]) + (1 - y_true[ : , 2] ) * tf.keras.losses.MAE(y_true[ : , 1], y_pred[ : , 1]) + tf.keras.losses.MSE(y_true[ : , 2], y_pred[ : , 2])
    return loss

def nn_out_to_pt(nn_out) :
    pts = tf.where(nn_out[ : , 2] > 0.80, nn_out[ : , 0], nn_out[ : , 1])
    #print("pts", pts)
    return pts
# output_type = 0 - pt 
# output_type = 1 - pt if pt > 10 GeV, pt if pt < 10 GeV, pt > 10 GeV,
output_type = 0


    

output_cnt = 1
loss_fn = 'mae'
if output_type == 1:
    output_cnt = 3
    loss_fn = custom_loss3
    
run_eagerly = False
#run_eagerly = True # only when true, the lut plots are made

nEpochs = 140 #80 #20 # 15

last_input_is_bias = True
#last_input_is_bias = False

nRefLayers = 8
nLayers = 18
nPDFBins = 2**7
minProbability = 0.001
minPlog = np.log(minProbability)
nPdfValBits = 6
refLayers = [0, 7, 2, 6, 16, 4, 10, 11]

# parameters of the classic NN
dense_layer1_size = 16 * 8
dense_layer2_size = 8 * 8
dense_layer3_size = 8 * 6
dense_layer4_size = output_cnt

# parameters of the LUT NN
input_I = 10
input_F = 4
networkInputSize = nLayers;
if last_input_is_bias :
    networkInputSize = networkInputSize +1

layer1_lut_size = 1 << input_I
layer1_neurons = 16 
layer1_lut_I = 3
layer1_lut_F = 10

layer1_output_I = 4
# 4 bits are for the count of the noHit layers which goes to the input of the layer2, other 4 for layer1_output_I
layer2_input_I = layer1_output_I + 4
layer2_lut_size = 1 << layer2_input_I
layer2_neurons = 8  #9 if the charge output is used
layer2_lut_I = 5
layer2_lut_F = 10

layer2_lutRangesCnt = 1
layer2_input_offset = None 
if last_input_is_bias :
    layer2_lutRangesCnt = 16
    layer2_input_offset = layer2_lut_size / layer2_lutRangesCnt / 2

layer3_input_I = 5
layer3_lut_size = 1 << layer3_input_I
layer3_neurons = output_cnt
layer3_lut_I = 6
layer3_lut_F = 10

# for LUT NN incpput conversion
rangeFactor = np.full(
  shape = nLayers,
  fill_value = 2 * 2,
  dtype = int
)

rangeFactor[1] = 8 *2
rangeFactor[3] = 4 *2 *2
rangeFactor[5] = 4 *2
rangeFactor[9] = 1 *2

if lut_nn:
    print("layer1_lut_size", layer1_lut_size)
    print("layer2_lut_size", layer2_lut_size)
    print("layer3_lut_size", layer3_lut_size)
    print("last_input_is_bias", last_input_is_bias)
    print("layer2_lutRangesCnt", layer2_lutRangesCnt)
    print("layer2_input_offset", layer2_input_offset)

#trainDataDir = "C:/Users/kbunk/Development/omtf_data/"
trainDataDir = "/home/kbunkow/cms_data/OMTF_data_2020/18_12_2020/"
   
testDataDir = trainDataDir 

inputDataPrefix =  "/scratch_ssd/akalinow/"
trainDataDir = inputDataPrefix+"/ProgrammingProjects/MachineLearning/OMTF/data/18_12_2020/"   
testDataDir = inputDataPrefix+"/ProgrammingProjects/MachineLearning/OMTF/data/18_12_2020/" 

testFileNames = glob.glob(trainDataDir+'OMTFHits_pats0x0003_newerSample_files_1_100_chunk_0.tfrecord.gzip')
trainFileNames = glob.glob(trainDataDir+'OMTFHits_pats0x0003_oldSample_files_*_chunk_0.tfrecord.gzip')
#trainFileNames = glob.glob(trainDataDir+'OMTFHits_pats0x0003_oldSample_files_1_10_chunk_0.tfrecord.gzip')
#testFileNames =  trainFileNames #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

testFileNames = glob.glob(trainDataDir+'OMTFHits_pats0x0003_newerSample_files_1_100_chunk_0.tfrecord.gzip')
#testFileNames = glob.glob(trainDataDir+'OMTFHits_pats0x0003_oldSample_files_1_10_chunk_0.tfrecord.gzip') #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Data manipulation functions

columns = np.array(['muonPt', 'muonEta', 'muonPhi', 'muonCharge', 'omtfPt', 'omtfEta',
       'omtfPhi', 'omtfCharge', 'omtfScore', 'omtfQuality', 'omtfRefLayer',
       'omtfProcessor', 'omtfFiredLayers', 'phiDist_0', 'phiDist_1',
       'phiDist_2', 'phiDist_3', 'phiDist_4', 'phiDist_5', 'phiDist_6',
       'phiDist_7', 'phiDist_8', 'phiDist_9', 'phiDist_10', 'phiDist_11',
       'phiDist_12', 'phiDist_13', 'phiDist_14', 'phiDist_15', 'phiDist_16',
       'phiDist_17'])

def getFeaturesMask():
    featuresMask = np.full_like(columns, False, dtype=bool)
    for iLayer in range(0, 18): 
        featureLabel = "phiDist_{}".format(iLayer)
        featuresMask += (columns==featureLabel)
    #featuresMask += columns=="omtfFiredLayers"
    #featuresMask += columns=="omtfRefLayer"
    #featuresMask += columns=="omtfPt"
    #featuresMask = columns=="omtfQuality"
    return featuresMask

def getFeature(name, dataRow):
    columnIndex = np.where(columns == name)[0][0]  
    return dataRow[:,columnIndex]

def parse_tensor(tensor):
    return tf.io.parse_tensor(tensor, out_type=tf.float32)


#here the LUT size of 1024 is asumed  
def makeLutNNFeatures(layerHits, omtfRefLayerFeature) :
    lutSize  = layer1_lut_size #1024
    maxHitVal = lutSize / nRefLayers / 2 #should be 64
    print("makeLutNNFeatures maxHitVal", maxHitVal, "lutSize", lutSize)
    
    #here in principle would be better if maxHitVal-1, i.e. 63, then the addres 1023 would be not used in the natural way 
    #offset = (omtfRefLayerFeature<<7) + maxHitVal; 
    offset = (omtfRefLayerFeature * (1<<7)) + maxHitVal; 
    print("makeLutNNFeatures offset", offset) 
        
    nnInputs = tf.divide(layerHits, rangeFactor)
    nnInputs = tf.clip_by_value(nnInputs, clip_value_min = -(maxHitVal-1), clip_value_max = maxHitVal-1)
    nnInputs = nnInputs + offset 
    nnInputs = tf.where(layerHits < 9999, nnInputs, lutSize -1) #noHitVal is 1023
    
    if last_input_is_bias :
        bias = layer2_lut_size / layer2_lutRangesCnt
        firedLayersBias = tf.reduce_sum( tf.where(nnInputs == lutSize -1, bias, 0), axis=1, keepdims=True)
        print("makeLutNNFeatures bias", bias)
        print("makeLutNNFeatures nnInputs", nnInputs)
        print("makeLutNNFeatures firedLayersBias", firedLayersBias)
        
        nnInputs = tf.concat([nnInputs, firedLayersBias], 1)
               
    return nnInputs
 
def modifyFeatures(dataRow, batchSize, isTrain=False):
    columnsMask = getFeaturesMask()
    print("columnsMask", columnsMask)
    # only selected columns form dataRow are used as features
    features = tf.boolean_mask(dataRow, columnsMask, axis=1)
    print("features:", features) 

    
    omtfRefLayerMask = columns=="omtfRefLayer"
    print("omtfRefLayerMask:", omtfRefLayerMask) 
    omtfRefLayerFeature = tf.boolean_mask(dataRow, omtfRefLayerMask, axis=1)
    
    if lut_nn :
        features = makeLutNNFeatures(features, omtfRefLayerFeature)
        if last_input_is_bias :
            features.set_shape([batchSize, np.count_nonzero(columnsMask) +1])
        else :
            features.set_shape([batchSize, np.count_nonzero(columnsMask)])    
            
    else :    
        dummyValue = 0.
        #dummyValue = 0
        firedLayers = tf.where(features<9999, 0., 16.) 
        features = tf.where(features<9999, features, dummyValue)
     
        features = tf.concat([features, firedLayers, omtfRefLayerFeature], 1)

        #features = features / 256.
        features.set_shape([batchSize, 2 * np.count_nonzero(columnsMask) +1])
    
    #features = tf.one_hot(tf.cast(features+128, dtype=tf.int32), depth=256)
    
    columnIndex = np.where(columns == "muonCharge")[0][0]  
    chargeLabels = (dataRow[:,columnIndex]+1)/2 
    chargeLabels.set_shape([batchSize,])
    
    columnIndex = np.where(columns == "muonPt")[0][0]
    ptLabels = dataRow[:,columnIndex]
    print("ptLabels line 227:", ptLabels)  
    ptLabels.set_shape([batchSize])
    
    if oneOverPt :
        trainWeight = 1.0 + tf.where(ptLabels > 10.0, 0.2 * (ptLabels - 10.0), 0.0) 
        ptLabels =  tf.divide(1., ptLabels)
    else :
        #trainWeight = 1.0 + tf.where(ptLabels<5.0, 1.0 * (5.0-ptLabels), 0.0)
        #trainWeight = 1.0 + tf.where(ptLabels<7.0, 2. * (7.0-ptLabels), 0.0)
        #trainWeight = 1.0 + tf.where(ptLabels<10.0, 1. * (10.0-ptLabels), 0.0) * tf.where(ptLabels<5.0, 1.0 * (5.0-ptLabels), 1.0)
        if train_sample == "newerSample" :
            trainWeight = 1.0 + tf.where(ptLabels < 15., 1.04 * (15.-ptLabels), 0.0) * tf.where(ptLabels < 7.0, 3. * (7.0-ptLabels), 0.0) * tf.where(ptLabels < 5.0, 1.1 * (5.0-ptLabels), 1.0) #should be the same as in the omtfRegression_v58 and 59
        else:
            trainWeight = 1.0 + tf.where(ptLabels < 7.0, 3. * (7.0-ptLabels), 0.0) * tf.where(ptLabels < 5.0, 1.1 * (5.0-ptLabels), 1.0) #should be the same as in the omtfRegression_v58 and 59
     
    print("ptLabels", ptLabels)
    if isTrain:
        if output_type == 1 :
            #ptLabels.set_shape([batchSize, 1])
            ptLabels = tf.expand_dims(ptLabels, 1)
            #ptLabels = tf.where(ptLabels > 10.0, tf.concat([ptLabels, tf.zeros_like(ptLabels), tf.ones_like(ptLabels)], 1), tf.concat([tf.zeros_like(ptLabels), ptLabels, tf.zeros_like(ptLabels)], 1))
            #print("tf.where(ptLabels > 10.0, ptLabels, 0.)\n" , tf.where(ptLabels > 10.0, ptLabels, 0.))
            #print("tf.where(ptLabels > 10.0, tf.ones_like(ptLabels), tf.zeros_like(ptLabels))", tf.where(ptLabels > 10.0, tf.ones_like(ptLabels), tf.zeros_like(ptLabels)))
            ptLabels = tf.concat([tf.where(ptLabels > 10.0, ptLabels, 0.), tf.where(ptLabels > 10.0, tf.zeros_like(ptLabels), ptLabels), tf.where(ptLabels > 10.0, tf.ones_like(ptLabels), tf.zeros_like(ptLabels))], 1)
            print("ptLabels line 233:", ptLabels)        
        
        #ptLabels = pT2Label(ptLabels)
        #return (features, (ptLabels, chargeLabels), trainWeight)
        return (features, (ptLabels,), trainWeight)
        #return (features, (ptLabels,))
    else:
        columnIndex = np.where(columns == "omtfPt")[0][0]  
        omtfPt = dataRow[:,columnIndex]
        columnIndex = np.where(columns == "omtfQuality")[0][0]  
        omtfQuality = dataRow[:,columnIndex]
        omtfPt = tf.where(omtfQuality>=12, omtfPt, 0) 
        omtfPt.set_shape([batchSize,])
        
        columnIndex = np.where(columns == "omtfCharge")[0][0]  
        omtfCharge = dataRow[:,columnIndex]
        omtfCharge = (omtfCharge+1)/2 
        omtfCharge.set_shape([batchSize,])
        
        columnIndex = np.where(columns == "muonEta")[0][0]  
        etaLabels = dataRow[:,columnIndex]
        etaLabels.set_shape([batchSize,])
        
        return (features, (ptLabels, chargeLabels, etaLabels, trainWeight), omtfPt, omtfCharge, omtfQuality)
    return dataRow
    
            
def loadDataset(fileNames, isTrain, nEpochs=1, batchSize=1):   
    print("loadDataset fileNames", fileNames)
    raw_dataset = tf.data.TFRecordDataset(fileNames, compression_type="GZIP", buffer_size=2**23)
    dataset = raw_dataset.map(parse_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("loadDataset: dataset", dataset)
    dataset = dataset.batch(batchSize, drop_remainder=True)
    #Split data into [features, labels] and modify features
    dataset = dataset.map(lambda x: modifyFeatures(x, batchSize, isTrain),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
   # tf.random.shuffle(dataset)
    #dataset = dataset.cache('/scratch/akalinow/data_cache/')
    return dataset

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(1E-10)
    tf.print("Execution time:", time.perf_counter() - start_time)

# Utility functions
# Functions used for making a ROOT-like Pandas DataFrame for making the performance plots.

def finalModelAnswer(predictions, cumulativePosteriorCut):
    variance = tf.math.reduce_variance(predictions[0], axis=1)
    
    x = np.arange(0,predictions[0].shape[1])
    x = np.broadcast_to(x, predictions[0].shape)
    mean = np.average(a=x,weights=predictions[0], axis=1)
    mean2 = np.average(a=x*x,weights=predictions[0], axis=1)
    sigma = np.sqrt(mean2-mean**2) 

    pt = tf.cumsum(predictions[0], axis=1)>cumulativePosteriorCut
    pt = tf.argmax(pt, axis=1)
    probability = np.amax(predictions[0], axis=1)
    charge = tf.cast(predictions[1]>0.5, dtype=tf.float32)
    charge =  tf.reshape(charge, (-1))
    return label2Pt(pt), charge, probability, sigma

def fillPandasDataset(aBatch, df, cumulativePosteriorCut):    
    features = aBatch[0]
    labels = aBatch[1]
    omtfPredictions = aBatch[2:4]
    probability = model.predict(features, use_multiprocessing=True)
    value = finalModelAnswer(probability, cumulativePosteriorCut)
    batch_df = pd.DataFrame(data={"genPt":labels[0], "genCharge":labels[1], "genEta":labels[2],
                                    "OMTF_pt":omtfPredictions[0], 
                                    "OMTF_charge":omtfPredictions[1], 
                                    "NN_pt": value[0],
                                    "NN_charge":value[1],
                                    "NN_prob":value[2],
                                    "NN_sigma":value[3],
                                    })
    return df.append(batch_df, ignore_index=True)

def fillPandasDatasetRegression(aBatch, df):    
    features = aBatch[0]
    labels = aBatch[1]
    omtfPredictions = aBatch[2:5]
    nnPrediction = model.predict(features, use_multiprocessing=True)
    #print("fillPandasDatasetRegression nnPrediction 341\n", nnPrediction)
    if output_type == 1 :
        nnPrediction = nn_out_to_pt(nnPrediction) 
        #nnPrediction = nnPrediction.numpy()
        #print("fillPandasDatasetRegression nnPrediction 345\n", nnPrediction)
    #else :    
    #    nnPrediction = nnPrediction.reshape([nnPrediction.__len__()])  #watch out when charge is added
        
    nnPrediction = tf.reshape(nnPrediction, (-1))
        
    #print("fillPandasDatasetRegression nnPrediction 348\n", nnPrediction)
    #print("fillPandasDatasetRegression")
    #print("features", features, "features.__len__", features.__len__())
    #print("labels", labels, "labels.__len__()", labels.__len__())
    #print("labels[0].__len__()", labels[0].__len__())
    #print("labels[1].__len__()", labels[1].__len__())
    #print("labels[2].__len__()", labels[2].__len__())
    #print("omtfPredictions", omtfPredictions)
    #print("nnPrediction", nnPrediction, "nnPrediction.__len__()", nnPrediction.__len__())
    #print("nnPrediction[0].__len__()", nnPrediction[0].__len__())
    batch_df = pd.DataFrame(data={"genPt":labels[0], "genCharge":labels[1], "genEta":labels[2],
                                    "OMTF_pt":omtfPredictions[0], 
                                    "OMTF_charge":omtfPredictions[1], 
                                    "OMTF_qual":omtfPredictions[2],
                                    #"NN_pt": nnPrediction[0],
                                    "NN_pt": nnPrediction.numpy(),
                                    #"NN_charge":value[1],
                                    #"NN_prob":value[2],
                                    #"NN_sigma":value[3],
                                    "weight":labels[3]
                                    })
    return df.append(batch_df, ignore_index=True)


# Data reading test

#import plotting_functions as plf
#importlib.reload(plf)




print("testFileNames", testFileNames)
print("trainFileNames", trainFileNames)

train_dataset = loadDataset(trainFileNames, isTrain=True, nEpochs=1, batchSize=10)
#benchmark(train_dataset)

#dataset = dataset.map(lambda x,y,z: tf.stack([x,y[1],y[1], z], axis=0))
#print(dataset)

#dataset = dataset.map(tf.io.serialize_tensor)

print("\ntrain_dataset:\n", train_dataset)

#printing fist event
#for element in train_dataset.skip(0).take(10):
for aBatch in train_dataset.as_numpy_iterator(): #TODO change to 
    #count = count +1
    print ("aBatch\n", aBatch)
    element =  aBatch
    #train data format: (features, (ptLabels, chargeLabels), trainWeight)
    #print("Hits in iLayer 0:\n", element[0][0][0])
    #print("Hits in iLayer 1:\n", element[0][0][1])
    
    #print("Hits in iLayers:\n", element[0][0])
    print("Hits in OMTF Layers:\n", element[0])
    print("ptLabels:\n", element[1])
    #print("chargeLabelss:\n", element[1][1])
    print("trainWeigh:\n", element[2])
    #plf.plotEvent(element, label2Pt)
    break


eventNum = 0

trainBatchSize = 2*4096

evnNum = 0
i = 0

# for element in train_dataset: 
#     pts[i] = 1./element[1][0]
#     i = i + 1
#     evnNum = evnNum + 1 
#     if i == trainBatchSize :
#         target_values_hist = tf.histogram_fixed_width(pts, value_range = (0, 200), nbins=100)
#         print("target_values_hist", target_values_hist)
#         i = 0

# for batch in train_dataset.skip(0).take(10) :
#     print('batch\n', batch)
#     print('batch[:][1][0]\n', batch[:][1][0])
#     target_values_hist = tf.histogram_fixed_width(1./batch[:][1][0], value_range = (0, 200), nbins=100)
#     print("target_values_hist", target_values_hist)
#     break
#
# print("trainBatchSize event count ", evnNum)      
#
# exit() 

class HistogramTargetValues(tf.keras.Model) :
    batch_target_values_hists = []
    def train_step(self, data):
        x, y, z = data
        #keys = list(logs.keys())
        #print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
        print("\n train_step, data", data)
        #target_values_hist = tf.map_fn(fn = lambda d : tf.histogram_fixed_width(d, value_range = (0, 200), nbins=100),
        #                         elems = batch[1],
        #                         fn_output_signature = tf.TensorSpec(shape=[None], dtype=tf.int32))
        
        target_values_hist = tf.histogram_fixed_width(tf.divide(1., y), value_range = (0, 200), nbins=100)
        
        print("target_values_hist", target_values_hist)
        self.batch_target_values_hists.append(target_values_hist)
        return super().train_step(data)


class TargetValuesHistogram(tf.keras.metrics.Metric):
    def __init__(self, name='targetValuesHistogram', **kwargs):
        super(TargetValuesHistogram, self).__init__(name=name, **kwargs)
        self.batch_target_values_hists = self.add_weight(name='tp', initializer='zeros', shape=(100, 100))
        #batch_target_values_hists = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        print("\non_train_batch_begin, batch", y_true)

        target_values_hist = tf.histogram_fixed_width(y_true, value_range = (0, 200), nbins=100)
        
        print("target_values_hist", target_values_hist)
        self.batch_target_values_hists.add(target_values_hist)

    def result(self):
        return self.batch_target_values_hists

dir_postfix =""

if lut_nn :
    dir_postfix = "_lut_nn_" + str(layer1_neurons) + "_" + str(layer2_neurons) + "_" + str(layer3_neurons) + "_" + train_sample
else :
    dir_postfix = "_classic_" + str(dense_layer1_size) + "_" + str(dense_layer2_size) + "_" + str(dense_layer3_size)+ "_" + str(dense_layer4_size) + "_" + train_sample
    
    

job_dir = "training/" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + dir_postfix

if train :
    model = keras.Sequential()
    
    num_inputs = np.sum(getFeaturesMask())
    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + dir_postfix
        
    if lut_nn :
        seed = 1234
        num_inputs = networkInputSize
        
        if run_eagerly :
            hist_writer = tf.summary.create_file_writer(log_dir + "/lutnn_input_hist")
            write_lut_hist = True
        else :
            hist_writer = None    
            write_lut_hist = False
        
        #initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = layer2_lut_size/64., seed = seed)
        initializer = LutInterLayer.LutInitializerLinear(maxLutVal = 1<<(layer1_lut_I-1), initSlopeMin = 0.01, initSlopeMax = 0.1/8, lutRangesCnt = nRefLayers)
        layer1 = LutInterLayer.LutInterLayer("layer1", lut_size = layer1_lut_size, num_inputs = num_inputs, num_outputs = layer1_neurons, input_offset= 0, initializer = initializer, hist_writer = hist_writer, write_lut_hist=write_lut_hist, last_input_is_bias = last_input_is_bias)
        
        #initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = 1, seed = seed)
        initializer = LutInterLayer.LutInitializerLinear(maxLutVal = 1<<(layer2_lut_I-1), initSlopeMin = 0.01, initSlopeMax = 0.1/8, lutRangesCnt = layer2_lutRangesCnt)
        layer2 = LutInterLayer.LutInterLayer("layer2", lut_size = layer2_lut_size, num_inputs = layer1_neurons, num_outputs = layer2_neurons, input_offset = layer2_input_offset, initializer = initializer, hist_writer = hist_writer, write_lut_hist=write_lut_hist)
        
        #initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = 1, seed = seed)
        initializer = LutInterLayer.LutInitializerLinear(maxLutVal = 1<<(layer3_lut_I-1), initSlopeMin = 0.01, initSlopeMax = 0.1, lutRangesCnt = 1)
        layer3 = LutInterLayer.LutInterLayer("layer3", lut_size = layer3_lut_size, num_inputs = layer2_neurons, num_outputs = 1, initializer = initializer, hist_writer = hist_writer, write_lut_hist=write_lut_hist)

              
        print("building model")
        model.add(tf.keras.Input(shape = [num_inputs], name="delta_Phi"))
        model.add(layer1)
        model.add(layer2)
        model.add(layer3)

           
        learning_rate=0.03
    else :
        print("building model")
        num_inputs = 2 * np.sum(getFeaturesMask()) + 1
        model.add(tf.keras.Input(shape = [num_inputs], name="delta_Phi") )
        model.add(tf.keras.layers.Dense(dense_layer1_size, activation='relu', name="pt_layer_1") )
        model.add(tf.keras.layers.Dense(dense_layer2_size, activation='relu', name="pt_layer_2") )
        model.add(tf.keras.layers.Dense(dense_layer3_size, activation='relu', name="pt_layer_3") )
        model.add(tf.keras.layers.Dense(dense_layer4_size)) #, activation='sigmoid'
        
        learning_rate=0.002
         
    train_dataset = loadDataset(trainFileNames, isTrain=True, nEpochs=1, batchSize=2*4096) #2*4096
    train_dataset.shuffle(buffer_size=1024*1024, reshuffle_each_iteration = True)
    print("train_dataset", train_dataset)
    
    #model(train_dataset)
    
    initial_learning_rate = learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1600/4,
    decay_rate=0.95,
    staircase=False)
    
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0., )
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule) #learning_rate
    #optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    #targetValuesHistogram = TargetValuesHistogram()
    #model.compile(optimizer=optimizer, loss='mse', run_eagerly=run_eagerly) #metrics=[targetValuesHistogram]
    model.compile(optimizer = optimizer, loss = loss_fn, run_eagerly=run_eagerly)
                      
    model.summary() 
    
   
    checkpoint_path = job_dir + "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq = 5085)
    
    #model_with_hist = HistogramTargetValues(model.input, model.output)
    #model_with_hist.compile(optimizer=model.optimizer, loss=model.loss, run_eagerly=True)
    
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=(10, 20))
    
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    #file_writer.set_as_default()
    
    class LossToTBoardCallback(keras.callbacks.Callback):
        def __init__(self):
            self.iteration = 0
            
        def on_train_batch_end(self, batch, logs=None):
            #print( "Up to batch {}, the average loss is {:7.5f}.".format(batch, logs["loss"]))
            with file_writer.as_default() :
                tf.summary.scalar('batch_loss', logs["loss"], step = self.iteration )  
            self.iteration = self.iteration +1      
        
    model.save_weights(checkpoint_path.format(epoch=0))

    #model_with_hist.fit(train_dataset, epochs=nEpochs, shuffle=False, batch_size = 1024 *2)
    model.fit(train_dataset, epochs=nEpochs, shuffle=True,
              callbacks=[tensorboard_callback, cp_callback, LossToTBoardCallback() ]
              )
    
    #print("targetValuesHistogram.batch_target_values_hists.size", targetValuesHistogram.batch_target_values_hists.__len__())
    #for hist in targetValuesHistogram.batch_target_values_hists :
    #    print (hist, summarize=-1) 
    
    # Save the whole model
    path = job_dir + "/{epoch:04d}_" + datetime.now().strftime("%Y_%b_%d-%H_%M")
    model.save(path.format(epoch=nEpochs), save_format='tf')
    
    current_time = datetime.now().strftime("%Y %b %d  %H:%M:%S")
    print("Training end. Current Time =", current_time)
else :
    if lut_nn :
        num_inputs = networkInputSize
        
        model = keras.Sequential()
        layer1 = LutInterLayer.LutInterLayer("layer1", lut_size = layer1_lut_size, num_inputs = num_inputs, num_outputs = layer1_neurons, input_offset= 0, write_lut_hist=True, last_input_is_bias = last_input_is_bias)
        layer2 = LutInterLayer.LutInterLayer("layer2", lut_size = layer2_lut_size, num_inputs = layer1_neurons, num_outputs = layer2_neurons, input_offset = layer2_input_offset, write_lut_hist=True)
        #layer3 = LutInterLayer.LutInterLayer("layer3", lut_size = layer3_lut_size, num_inputs = layer2_neurons, num_outputs = 1)
        layer3 = LutInterLayer.LutInterLayer("layer3", lut_size = layer3_lut_size, num_inputs = layer2_neurons, num_outputs = 1, write_lut_hist=True)

        print("building model")
        model.add(tf.keras.Input(shape = [num_inputs], name="delta_Phi"))
        model.add(layer1)
        model.add(layer2)
        model.add(layer3)
        
        #job_dir = "training/2022_12_20-11_09_57"
        #job_dir = "training/2022_12_20-00_36_27"
        #job_dir = "training/2022_12_29-08_25_50_lut_nn_16_8_1"
        #job_dir = "training/2022_12_29-10_33_17_lut_nn_32_16_1"
        #job_dir = "training/2022_12_31-14_36_14_lut_nn_16_8_1"
        #job_dir = "training/2022_12_29-10_33_17_lut_nn_32_16_1"
        #job_dir = "training/2022_12_29-08_25_50_lut_nn_16_8_1"
        job_dir = "training/2023_01_02-00_38_32_lut_nn_16_8_1"
        #job_dir = "training/2023_01_02-15_54_50_lut_nn_16_8_1"
        
        #job_dir = "training/2023_01_07-09_14_28_lut_nn_32_16_1//"
        #job_dir = "training/2023_01_07-19_04_30_lut_nn_32_16_1/"
        #job_dir = "training/2023_01_07-21_20_43_lut_nn_32_16_1/"
        
        #job_dir = "training/2023_01_08-01_03_33_lut_nn_16_8_1/"
                
        filepath = job_dir + "/cp-0012.ckpt"
        model.load_weights(filepath)
        
        model.compile(loss = loss_fn)
    else:    
        #job_dir = "training/2022_12_19-21_56_22"
        
        #job_dir = "training/2023_01_05-23_04_09_classic_192_96_1"
        #model = "0012_2023_Jan_06-00_46" 
        
        #job_dir = "training/2023_01_03-08_18_00_classic_256_128_1"
        #model = "0012_2023_Jan_03-10_03"       
        
        #job_dir = "training/2023_01_03-14_56_23_classic_256_128_1"
        #model = "0012_2023_Jan_03-16_41"
        
        #job_dir = "training/2023_01_06-13_14_24_classic_256_128_128_1"
        #model = "0012_2023_Jan_06-15_04"
        
        #job_dir = "training/2023_01_10-14_58_48_classic_256_128_128_1"
        #model = "0015_2023_Jan_10-17_19"

        job_dir = "training/2023_01_30-13_34_03_classic_256_128_96_3"
        #model = "0005_2023_Jan_30-13_14"

        #filepath = job_dir + "/" + model
        #model = tf.keras.models.load_model(filepath)
        
        model = keras.Sequential()
        num_inputs = 2 * np.sum(getFeaturesMask()) + 1
        model.add(tf.keras.Input(shape = [num_inputs], name="delta_Phi") )
        model.add(tf.keras.layers.Dense(dense_layer1_size, activation='relu', name="pt_layer_1") )
        model.add(tf.keras.layers.Dense(dense_layer2_size, activation='relu', name="pt_layer_2") )
        model.add(tf.keras.layers.Dense(dense_layer3_size, activation='relu', name="pt_layer_3") )
        model.add(tf.keras.layers.Dense(dense_layer4_size)) #, activation='sigmoid'
        
        filepath = job_dir + "/cp-0005.ckpt"
        model.load_weights(filepath)
        
        model.compile(loss = loss_fn)
        
    #model.load_weights(filepath, by_name)
    print("model loaded from", job_dir)
    model.summary()
    
for layer in model.layers:
    print(layer)
    if isinstance(layer, LutInterLayer.LutInterLayer):
        print("layer.write_lut_hist", layer.write_lut_hist)
        layer.write_lut_hist = True
        print("layer.write_lut_hist", layer.write_lut_hist)
    

if False :
    test_dataset = loadDataset(testFileNames, isTrain=True, nEpochs=1, batchSize= 32*1024) #32*

    model.evaluate(test_dataset, batch_size = 1024 * 2,
              #callbacks=[tensorboard_callback, cp_callback, LossToTBoardCallback() ]
              )
    
    exit()

test_dataset = loadDataset(testFileNames, isTrain=False, nEpochs=1, batchSize= 32*1024) #32*
    
df = pd.DataFrame(columns=["genPt", "genCharge", "genEta",
                            "OMTF_pt", "OMTF_charge",
                            "NN_pt" #,"NN_charge","NN_prob"
                            ])

print("test_dataset", test_dataset)

count = 0
for aBatch in test_dataset.as_numpy_iterator(): #TODO change to 
    df = fillPandasDatasetRegression(aBatch, df)
    count = count +1
    #if count >= 10 :
     #   break


plotDir = job_dir + "/figures"
if not train :
    plotDir = job_dir + "/figures_1"
    if not os.path.exists(plotDir):
        os.mkdir(plotDir, )
else :    
    os.mkdir(plotDir)

#doing te plot_luts takes long time, so it is not worth to reapeat it
#remove if if needed     
if train :    
    for layer in model.layers:
        if isinstance(layer, LutInterLayer.LutInterLayer):
            layer.plot_luts(plotDir + "/")

print("df.size", df.size)

print("df[genPt].size", df["genPt"].size)
print("genPt\n", df["genPt"])
print("NN_pt\n", df["NN_pt"])

plt.style.use('_mpl-gallery')
plt.rcParams['axes.labelsize'] = 6 
plt.rcParams['ytick.labelsize'] = 6 
plt.rcParams['xtick.labelsize'] = 6 
# plot



if oneOverPt :
    #if in modifyFeatures ptLabels =  tf.divide(1., ptLabels)
    df["genPt"] = 1./df["genPt"]
    df["NN_pt"] =  1./df["NN_pt"]


losses_text_file = open(job_dir + "/losses.txt", "w")

lossFunction = tf.keras.losses.MeanAbsoluteError()
loss = lossFunction(df["genPt"], df["NN_pt"]).numpy()
print("MeanAbsoluteError NN", loss)
print("MeanAbsoluteError NN", loss, file=losses_text_file)

loss = lossFunction(df["genPt"], df["OMTF_pt"]).numpy()
print("MeanAbsoluteError OMTF", loss)

lossFunction = tf.keras.losses.MeanSquaredError()
loss = lossFunction(df["genPt"], df["NN_pt"]).numpy()
print("MeanSquaredError NN", loss)
print("MeanSquaredError NN", loss, file=losses_text_file)

loss = lossFunction(df["genPt"], df["OMTF_pt"]).numpy()
print("MeanSquaredError OMTF", loss)


plf.plotPtGenPtRec(df, plotDir, oneOverPt)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! pt recalibration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ptToPtCalibNN, xedges = plf.ptRecalibration(df, plotDir, oneOverPt, "NN_pt")
df["NN_pt"] = plf.ptRecalibrated(df["NN_pt"], ptToPtCalibNN, xedges) 


lossFunction = tf.keras.losses.MeanAbsoluteError()
loss = lossFunction(df["genPt"], df["NN_pt"]).numpy()
print("ptRecalibrated\nMeanAbsoluteError NN", loss)
print("ptRecalibrated\nMeanAbsoluteError NN", loss, file=losses_text_file)

lossFunction = tf.keras.losses.MeanSquaredError()
loss = lossFunction(df["genPt"], df["NN_pt"]).numpy()
print("MeanSquaredError NN", loss)
print("MeanSquaredError NN", loss, file=losses_text_file)


#ptToPtCalibNN, xedges = plf.ptRecalibration(df, plotDir, oneOverPt, "OMTF_pt")
#df["OMTF_pt"] = plf.ptRecalibrated(df["OMTF_pt"], ptToPtCalibNN, xedges)

df1 = df

ptCuts = (10, 15, 20, 25, 30, 40)
qualityCut =  12

for ptCut in ptCuts :
    eff = plf.plotTurnOn(df1, ptCut=ptCut, qualityCut = qualityCut, plotDir=plotDir)
    print( eff[0])
    print( eff[0], file=losses_text_file)

rates = plf.plotRate(df1, qualityCut, plotDir=plotDir)

plt.show()
plt.subplots_adjust(hspace=0.4, wspace=0.17, left = 0.035, bottom = 0.045, top = 0.95)

print("rates NN", rates)
print("rates NN", rates, file=losses_text_file)
losses_text_file.close()

if train :
    shutil.copy2('OMTF_NN_v1.py', job_dir)
    shutil.copy2('out.txt', job_dir)

