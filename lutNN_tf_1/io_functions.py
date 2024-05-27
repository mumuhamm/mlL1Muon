import numpy as np
import time
from functools import partial

import tensorflow as tf

import architecture_definitions 
###################################################
###################################################
columns_v1 = np.array(['muonPt', 'muonEta', 'muonPhi', 'muonCharge', 'omtfPt', 'omtfEta',
       'omtfPhi', 'omtfCharge', 'omtfScore', 'omtfQuality', 'omtfRefLayer',
       'omtfProcessor', 'omtfFiredLayers', 'phiDist_0', 'phiDist_1',
       'phiDist_2', 'phiDist_3', 'phiDist_4', 'phiDist_5', 'phiDist_6',
       'phiDist_7', 'phiDist_8', 'phiDist_9', 'phiDist_10', 'phiDist_11',
       'phiDist_12', 'phiDist_13', 'phiDist_14', 'phiDist_15', 'phiDist_16',
       'phiDist_17'])
###################################################
###################################################
columns_v2 = np.array(['eventNum', 'muonEvent', 'muonPt', 'muonEta', 'muonPhi', 'muonCharge',
       'muonDxy', 'muonRho', 'omtfPt', 'omtfEta', 'omtfPhi', 'omtfCharge',
       'omtfHwEta', 'omtfProcessor', 'omtfScore', 'omtfQuality',
       'omtfRefLayer', 'omtfRefHitNum', 'omtfFiredLayers', 
       'phiDist_0', 'phiDist_1', 'phiDist_2', 'phiDist_3', 'phiDist_4',
       'phiDist_5', 'phiDist_6', 'phiDist_7', 'phiDist_8', 'phiDist_9',
       'phiDist_10', 'phiDist_11', 'phiDist_12', 'phiDist_13', 'phiDist_14',
       'phiDist_15', 'phiDist_16', 'phiDist_17'])
###################################################
columns = columns_v2
###################################################
featuresMaskBase = np.full_like(columns, False, dtype=bool)
for iLayer in range(0, architecture_definitions.nLayers): 
        featureLabel = "phiDist_{}".format(iLayer)
        featuresMaskBase += (columns==featureLabel)
###################################################
def getFeaturesMask():
    featuresMask = featuresMaskBase
    #featuresMask += columns=="omtfFiredLayers"
    #featuresMask += columns=="omtfRefLayer"
    #featuresMask += columns=="omtfPt"
    #featuresMask = columns=="omtfQuality"
    return featuresMask
###################################################
###################################################
def getFeature(name, dataRow):
    columnIndex = np.where(columns == name)[0][0]  
    return dataRow[:,columnIndex]
###################################################
###################################################
def parse_tensor(tensor):
    x = tf.io.parse_tensor(tensor, out_type=tf.float32)
    x = tf.cast(x, tf.float16)
    return x
###################################################
###################################################
def weightsFuncDummy(labels, batchSize):
    return tf.constant(value=1.0, shape=(batchSize,), dtype=tf.float16)
###################################################
###################################################
def makeLutNNFeatures(dataRow, batchSize, 
                      nRefLayers,
                      layer1_lut_size, layer2_lut_size, layer2_lutRangesCnt,
                      last_input_is_bias, rangeFactor):
    
    columnsMask = getFeaturesMask()
    layerHits = tf.boolean_mask(dataRow, columnsMask, axis=1)
    
    omtfRefLayerMask = columns=="omtfRefLayer"
    omtfRefLayerFeature = tf.boolean_mask(dataRow, omtfRefLayerMask, axis=1)
    
    lutSize  = layer1_lut_size 
    assert lutSize==1024, "layer1_lut_size should be 1024" 
    maxHitVal = lutSize / nRefLayers / 2 
    noHitVal = lutSize - 1
    assert maxHitVal==64, "maxHitVal should be 64"
    
    #here in principle would be better if maxHitVal-1, i.e. 63 was used, then the addres 1023 would be not used in the natural way 
    offset = (omtfRefLayerFeature * (1<<7)) + maxHitVal; 
        
    nnInputs = tf.divide(layerHits, rangeFactor)
    nnInputs = tf.clip_by_value(nnInputs, clip_value_min = -(maxHitVal-1), clip_value_max = maxHitVal-1)
    nnInputs = nnInputs + offset 
    nnInputs = tf.where(layerHits < 9999, nnInputs, noHitVal) 
    nnInputs.set_shape([batchSize, np.count_nonzero(columnsMask)])   
    
    if last_input_is_bias :
        bias = layer2_lut_size / layer2_lutRangesCnt
        firedLayersBias = tf.reduce_sum(tf.where(nnInputs == lutSize -1, bias, 0), axis=1, keepdims=True)    
        firedLayersBias = tf.cast(firedLayersBias, tf.float16)
        nnInputs = tf.concat([nnInputs, firedLayersBias], 1)
        nnInputs.set_shape([batchSize, np.count_nonzero(columnsMask) +1])
               
    return nnInputs
###################################################
###################################################
def makeNNFeatures(dataRow, batchSize):
    
    columnsMask = getFeaturesMask()
    layerHits= tf.boolean_mask(dataRow, columnsMask, axis=1)
    
    omtfRefLayerMask = columns=="omtfRefLayer"
    omtfRefLayerFeature = tf.boolean_mask(dataRow, omtfRefLayerMask, axis=1)
    
    dummyValue = tf.constant(0.0, tf.float16)
    firedLayers = tf.where(layerHits<9999, 0., 16.) 
    firedLayers = tf.cast(firedLayers, tf.float16)
    layerHits = tf.where(layerHits<9999, layerHits, dummyValue)
    nnInputs = tf.concat([layerHits, firedLayers, omtfRefLayerFeature], 1)
    nnInputs.set_shape([batchSize, 2 * np.count_nonzero(columnsMask) +1])
    
    return nnInputs
###################################################
###################################################
def makeLabels(dataRow, batchSize, isTrain):
    
    columnIndex = np.where(columns == "muonCharge")[0][0]  
    chargeLabels = (dataRow[:,columnIndex]+1)/2 
    chargeLabels.set_shape([batchSize,])
    
    columnIndex = np.where(columns == "muonPt")[0][0]
    ptLabels = dataRow[:,columnIndex]
    ptLabels.set_shape([batchSize])
    
    if isTrain:
        labels = (ptLabels,)
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
        labels = (ptLabels, chargeLabels, etaLabels, omtfPt, omtfCharge, omtfQuality)
        
    return labels       
###################################################
###################################################
def datasetMap(dataRow, batchSize, featuresMapFunc, labelsMapFunc, weightsFunc, isTrain=False):
    
    features = featuresMapFunc(dataRow, batchSize)
    labels = labelsMapFunc(dataRow, batchSize, isTrain)
    weights = weightsFunc(labels, batchSize)
    
    return (features, labels, weights)
###################################################
###################################################  
def loadDataset(fileNames, mapFunc, nEpochs=1, batchSize=1):   
    print("Reading data from files:", *fileNames, sep="\n")
    raw_dataset = tf.data.TFRecordDataset(fileNames, compression_type="GZIP", buffer_size=2**23)
    dataset = raw_dataset.map(parse_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchSize, drop_remainder=True)
    dataset = dataset.map(mapFunc,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
###################################################
###################################################
def dumpOneEvent(dataset):
    
    for aBatch in dataset.as_numpy_iterator(): 
        features,labels,weights = aBatch
        print("features.shape:",features.shape)
        print("len(labels)",len(labels))
        print("labels[0].shape:",labels[0].shape)
        print("weights.shape:",weights.shape)
        print("Hits in OMTF Layers:\n", features[0])
        print("ptLabels:\n", labels[0][0])
        print("weights:\n", weights[0])
        break
###################################################
###################################################
def reading_benchmark(dataset, nEpochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(nEpochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(1E-10)
    tf.print("Execution time:", time.perf_counter() - start_time)
###################################################
###################################################
def get_Classic_NN_dataset(batchSize, nEpochs, trainFileNames, isTrain):

    featuresMapFunc = makeNNFeatures
    labelsMapFunc = makeLabels
    weightsFunc = weightsFuncDummy

    datasetMapFunc = partial(datasetMap, batchSize=batchSize, 
                                           featuresMapFunc=featuresMapFunc, 
                                           labelsMapFunc=labelsMapFunc, 
                                           weightsFunc=weightsFunc, 
                                           isTrain=isTrain)

    dataset = loadDataset(trainFileNames, datasetMapFunc, nEpochs=nEpochs, batchSize=batchSize)
    return dataset
###################################################
###################################################
def get_LUT_NN_dataset(batchSize, nEpochs, trainFileNames, isTrain,
                        nRefLayers,
                        layer1_lut_size,
                        layer2_lut_size,
                        layer2_lutRangesCnt,
                        last_input_is_bias,
                        rangeFactor):

    featuresMapFunc = partial(makeLutNNFeatures, 
                              nRefLayers=nRefLayers,
                              layer1_lut_size=layer1_lut_size,
                              layer2_lut_size=layer2_lut_size,
                              layer2_lutRangesCnt=layer2_lutRangesCnt,
                              last_input_is_bias=last_input_is_bias, 
                              rangeFactor=rangeFactor)
    
    labelsMapFunc = makeLabels
    weightsFunc = weightsFuncDummy

    datasetMapFunc = partial(datasetMap, batchSize=batchSize, 
                                           featuresMapFunc=featuresMapFunc, 
                                           labelsMapFunc=labelsMapFunc, 
                                           weightsFunc=weightsFunc, 
                                           isTrain=isTrain)

    dataset = loadDataset(trainFileNames, datasetMapFunc, nEpochs=nEpochs, batchSize=batchSize)
    return dataset
###################################################
###################################################