import os, glob

import tensorflow as tf
import pandas as pd
import numpy as np
###################################################
###################################################
df = pd.DataFrame(columns=["GEN_pt", "GEN_charge", "GEN_eta",
                            "OMTF_pt", "OMTF_charge", "OMTF_quality"
                            "NN_pt", "NN_charge","NN_prob"
                            ])
###################################################
###################################################
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
###################################################
###################################################
def classicNNAnswerPostProc(modelAnswer, output_type):
     
    result = modelAnswer    
    result = np.reshape(result, (-1))    
    dummyColumn = np.full_like(result, 0)  
    return np.column_stack((result, dummyColumn, dummyColumn, dummyColumn))
###################################################
###################################################
def lutNNAnswerPostProc(modelAnswer, output_type):
    
    if output_type == 1 :
        result = tf.where(modelAnswer[ : , 2] > 0.80, modelAnswer[ : , 0], modelAnswer[ : , 1])
    else:
        result = modelAnswer
        
    result = np.reshape(result, (-1))    
    dummyColumn = np.full_like(result, 0)  
    return np.column_stack((result, dummyColumn, dummyColumn, dummyColumn))
###################################################
###################################################
def fillPandasDataset(aBatch, df, model, modelAnswerPostProc):    
    features = aBatch[0]
    labels = aBatch[1]
    weights = aBatch[2]
    omtfPredictions = labels[3:6]
    modelAnswer = model(features)
    modelAnswer = modelAnswerPostProc(modelAnswer)
    
    batch_df = pd.DataFrame(data={"GEN_pt":labels[0], 
                                  "GEN_charge":labels[1], 
                                  "GEN_eta":labels[2],
                                  "OMTF_pt":omtfPredictions[0], 
                                  "OMTF_charge":omtfPredictions[1],
                                  "OMTF_quality":omtfPredictions[2],
                                  "NN_pt": modelAnswer[:,0],
                                  "NN_charge":modelAnswer[:,1],
                                  "NN_prob":modelAnswer[:,2],
                                  "NN_sigma":modelAnswer[:,3],
                                  "weight": weights
                                    })
    return pd.concat((df, batch_df), ignore_index=True).astype('float32')
###################################################
###################################################
def calculateLossFunctions(df, fileName):
    
    losses_text_file = open(fileName, "w")

    lossFunction = tf.keras.losses.MeanAbsoluteError()
    loss = lossFunction(df["GEN_pt"], df["NN_pt"]).numpy()
    print("MeanAbsoluteError NN", loss)
    print("MeanAbsoluteError NN", loss, file=losses_text_file)
    
    loss = lossFunction(df["GEN_pt"], df["NN_pt_recalib"]).numpy()
    print("MeanAbsoluteError NN recalib.", loss)
    print("MeanAbsoluteError NN recalib.", loss, file=losses_text_file)

    loss = lossFunction(df["GEN_pt"], df["OMTF_pt"]).numpy()
    print("MeanAbsoluteError OMTF", loss)

    lossFunction = tf.keras.losses.MeanSquaredError()
    loss = lossFunction(df["GEN_pt"], df["NN_pt"]).numpy()
    print("MeanSquaredError NN", loss)
    print("MeanSquaredError NN", loss, file=losses_text_file)
    
    loss = lossFunction(df["GEN_pt"], df["NN_pt_recalib"]).numpy()
    print("MeanSquaredError NN recalib", loss)
    print("MeanSquaredError NN recalib", loss, file=losses_text_file)

    loss = lossFunction(df["GEN_pt"], df["OMTF_pt"]).numpy()
    print("MeanSquaredError OMTF", loss)
    losses_text_file.close()

###################################################
###################################################
def appendStringToFile(aString, fileName):
    text_file = open(fileName, "a")
    print(aString, file=text_file)
    text_file.close()
###################################################
###################################################
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
###################################################
###################################################
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
###################################################
###################################################
class LossToTBoardCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.iteration = 0
            
        def on_train_batch_end(self, batch, logs=None):
            #print( "Up to batch {}, the average loss is {:7.5f}.".format(batch, logs["loss"]))
            with file_writer.as_default() :
                tf.summary.scalar('batch_loss', logs["loss"], step = self.iteration )  
            self.iteration = self.iteration +1 
###################################################
###################################################
def pT2Label(tensor):
    tensor = tf.searchsorted(ptBins, tensor, side='left')
    return tensor
###################################################
###################################################    
def label2Pt(tensor):  
    return tf.where(ptBins.numpy()[tensor]<9999, ptBins.numpy()[tensor], [200])
###################################################
###################################################
def getLatestModelPath(trainingPath="training/", pattern="lut"):
 
    directories = glob.glob(trainingPath+"/*"+pattern+"*")
    directories.sort(key=os.path.getmtime)
    return directories[-1][len(trainingPath):]
###################################################
###################################################