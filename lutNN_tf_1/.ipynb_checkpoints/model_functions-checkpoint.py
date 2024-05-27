import tensorflow as tf
import LutInterLayer
from architecture_definitions import *

###################################################
###################################################
def get_LUT_NN(last_input_is_bias, loss_fn):
    hist_writer = None    
    write_lut_hist = False

    initializer = LutInterLayer.LutInitializerLinear(maxLutVal = 1<<(layer1_lut_I-1), 
                                                     initSlopeMin = 0.01, initSlopeMax = 0.1/8, 
                                                     lutRangesCnt = nRefLayers)
    layer1 = LutInterLayer.LutInterLayer("layer1", lut_size = layer1_lut_size, num_inputs = networkInputSize, num_outputs = layer1_neurons, 
                                         input_offset= 0, initializer = initializer, 
                                         hist_writer = hist_writer, write_lut_hist=write_lut_hist, 
                                         last_input_is_bias = last_input_is_bias)

    initializer = LutInterLayer.LutInitializerLinear(maxLutVal = 1<<(layer2_lut_I-1), 
                                                     initSlopeMin = 0.01, initSlopeMax = 0.1/8, 
                                                     lutRangesCnt = layer2_lutRangesCnt)
    layer2 = LutInterLayer.LutInterLayer("layer2", lut_size = layer2_lut_size, num_inputs = layer1_neurons, num_outputs = layer2_neurons, 
                                         input_offset = layer2_input_offset, initializer = initializer, hist_writer = hist_writer, 
                                         write_lut_hist=write_lut_hist)

    initializer = LutInterLayer.LutInitializerLinear(maxLutVal = 1<<(layer3_lut_I-1), 
                                                     initSlopeMin = 0.01, initSlopeMax = 0.1, 
                                                     lutRangesCnt = 1)
    layer3 = LutInterLayer.LutInterLayer("layer3", lut_size = layer3_lut_size, num_inputs = layer2_neurons, num_outputs = 1, 
                                         initializer = initializer, hist_writer = hist_writer, write_lut_hist=write_lut_hist)

    model = tf.keras.Sequential()     
    model.add(tf.keras.Input(shape = [networkInputSize], name="delta_phi"))
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)

    learning_rate=0.03
    initial_learning_rate = learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                decay_steps=1600/4,
                decay_rate=0.95,
                staircase=False)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule) 
    model.compile(optimizer = optimizer, loss = loss_fn)                  
    return model
###################################################
###################################################
def get_Classic_NN(networkInputSize, loss_fn):
    
    model = tf.keras.Sequential()  
    model.add(tf.keras.Input(shape = [networkInputSize], name="delta_phi") )
    model.add(tf.keras.layers.Dense(dense_layer1_size, activation='relu', name="pt_layer_1") )
    model.add(tf.keras.layers.Dense(dense_layer2_size, activation='relu', name="pt_layer_2") )
    model.add(tf.keras.layers.Dense(dense_layer3_size, activation='relu', name="pt_layer_3") )
    model.add(tf.keras.layers.Dense(dense_layer4_size, activation='linear', dtype=tf.float32)) 
    
    learning_rate=0.03
    initial_learning_rate = learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                decay_steps=1600/4,
                decay_rate=0.95,
                staircase=False)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule) 
    model.compile(optimizer = optimizer, loss = loss_fn)                  
    return model
###################################################
###################################################    