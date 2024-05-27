import tensorflow as tf
from tensorflow import keras
import numpy as np
#from keras import backend
import math
import matplotlib.pyplot as plt

import LutInterLayer

#import numpy as np

#tf.random.set_seed(1234) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
#rng = np.random.RandomState(2021)
rng = np.random.RandomState()


#@tf.custom_gradient
def get_out_values(x, luts_float, entries_hist = None, training=None) :
    x = x + lut_size/2.
    x = tf.clip_by_value(x, clip_value_min = 0, clip_value_max = lut_size - 1.00001)
    
    #addresses = tf.reshape(x, shape = [self.num_outputs, self.num_address_bits] ) this should be done in layer
    #print("\get_out_values: x:", x)
    x = tf.transpose(x)
    #print("\get_out_values: transpose(x):", x)
    
    if training == False :
        #entries_hist_ = tf.histogram_fixed_width_bins(x[0], value_range = (0, float(luts_float.shape[2]) ), nbins=luts_float.shape[2])
        entries_hist = tf.map_fn(fn = lambda d : tf.histogram_fixed_width(d, value_range = (0, float(luts_float.shape[2])), nbins=luts_float.shape[2]),
                                 elems = x,
                                 fn_output_signature = tf.TensorSpec(shape=[None], dtype=tf.int32))
        print("entries_hist", entries_hist)
        
    #watch out: cast for negative values rounds up, not down, 
    #but the x should be positive here, otherwise it will not address the LUTs correctly
    x0 = tf.cast(x, tf.int32) 
    
    x1 = x0 + 1;
    
    #print("luts_float.shape()[0]", luts_float.shape[0])
    # for iOut in range(0, luts_float.shape[0], 1) :     
    #     lutVals = tf.gather(params = luts_float[iOut], indices=x, axis=1, batch_dims=1) 
    #     print("lutVals", lutVals)
    #
    #     lutSum = tf.reduce_sum(lutVals, axis=0)
    #     print("lutSum", lutSum)
    
    def getOutValue(lut):
        lutVals0 = tf.gather(params = lut, indices=x0, axis=1, batch_dims=1) 
        #print("lutVals", lutVals)
     
        lutVals1 = tf.gather(params = lut, indices=x1, axis=1, batch_dims=1) 
        
        interVals = (lutVals1 - lutVals0) * tf.math.floormod(x, 1) + lutVals0
     
        lutSum = tf.reduce_sum(interVals, axis=0)
        #print("lutSum", lutSum)
        return lutSum
     
    outValues = tf.map_fn(fn = getOutValue,
          elems = luts_float,
          fn_output_signature = tf.TensorSpec(shape=[None], dtype=tf.float32))

    outValues = tf.transpose(outValues)
    #print("outValues", outValues)
    
    return outValues

###################################################################################################

###################################################################################################

#test

# luts = tf.Variable([[[1, 2, 3],
#                      [4, 5, 6]], 
#                     [[11, 12, 13],
#                      [14, 15, 16]]])
#
# print ("luts", luts)
# print ("luts.shape", luts.shape)
#
# print ("luts[0]", luts[0])
# print ("luts[1]", luts[1])
#
# print("tf.version.VERSION", tf.version.VERSION)
#
# inputs = tf.Variable([[0, 1],
#                       [1, 2],
#                       [2, 1],
#                       [1, 0]]) 
#
# print ("inputs", inputs)
#
# get_out_values(inputs, luts)

#exit()


layer0_lut_size = 16 
num_inputs = 2 
layer0_num_outputs = 10 

layer2_lut_size = 20

eventCnt = 10000

input_data = rng.uniform(size=(eventCnt, num_inputs), low= -layer0_lut_size/2., high = (layer0_lut_size - 0.001)/2. -1)

y = tf.Variable(initial_value = tf.zeros(shape = [eventCnt, 1], dtype = tf.float32 ))

for i in range(0, eventCnt) :
    val = 0.
    for j in range(0, num_inputs):
        val += input_data[i, j] * input_data[i, j] / 16.
    y[i].assign(math.sin(val) )
    #print("y[", i, "] ", y[i])
    
y = tf.constant(y)

#print("\ninput_data", input_data)

#optimizer = tf.keras.optimizers.SGD(learning_rate=1., momentum=0., )
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.002)

#layer0_result = layer0(input = input_data)
#layer1_result = layer1(input = layer0_result)

lut_nn = False

model = keras.Sequential()

if lut_nn :
    initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = layer2_lut_size/4., seed = 1234)
    layer0 = LutInterLayer.LutInterLayer("layer0", lut_size = layer0_lut_size, num_inputs = num_inputs, num_outputs = layer0_num_outputs, initializer = initializer)
    
    initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = 1, seed = 1234)
    layer1 = LutInterLayer.LutInterLayer("layer1", lut_size = layer2_lut_size, num_inputs = layer0_num_outputs, num_outputs = 1, initializer = initializer)
    
    print("building model")
    model.add(tf.keras.Input(shape = [num_inputs]))
    model.add(layer0)
    model.add(layer1)

else :
    print("building model")
    model.add(tf.keras.Input(shape = [num_inputs]))
    model.add(tf.keras.layers.Dense(100, activation='sigmoid') )
    model.add(tf.keras.layers.Dense(100, activation='sigmoid') )
    model.add(tf.keras.layers.Dense(1)) #, activation='sigmoid'
     
   
model(input_data)

model.compile(optimizer=optimizer, loss='mse')

model.summary() 

model.fit(input_data, y, epochs=800, shuffle=True, batch_size = 500)

print("model.evaluate")
model.evaluate(input_data,  y, verbose=2)

#print("layer0.luts_float", layer0.luts_float)
#print("layer1.luts_float", layer1.luts_float)

output_array = model(input_data)
print("inputData.shape", input_data.shape)
print("output_array.shape", output_array.shape)

#print("inputData\n", inputData)
#print("output_array\n", output_array)
#print("y", y) 

#for i in range(0, eventCnt) :
#    print(y[i], output_array[i], math.fabs(y[i] - output_array[i]))

plt.style.use('_mpl-gallery')
plt.rcParams['axes.labelsize'] = 6 
plt.rcParams['ytick.labelsize'] = 6 
plt.rcParams['xtick.labelsize'] = 6 
# plot
fig1, ax1 = plt.subplots(2, 2)

ax1[0, 0].scatter(y, output_array, linewidth=1.0, s=2)


#ax1[0, 1].scatter(input_data[: , 0], y, linewidth=1.0, s=9)

#ax1[1, 1].scatter(input_data[: , 0], output_array, linewidth=1.0, s=9)

ax1[0, 1].scatter(input_data[: , 0], input_data[: , 1], c = y, linewidth=1.0, s=4)

ax1[1, 1].scatter(input_data[: , 0], input_data[: , 1], c = output_array, linewidth=1.0, s=4)

#ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#       ylim=(0, 8), yticks=np.arange(1, 8))

def plot_luts(lut_layer) :
    fig2, ax2 = plt.subplots(8, 4, sharex=True) #(y,x)
    
    lut_xlables =  list(range(lut_layer.luts_float.shape[2]))
    
    #print("lut_layer.entries_hist", lut_layer.entries_hist)
    #luts_float[num_outputs][num_inputs][lut_size]
    for i_out in range(0, min(ax2.shape[1], lut_layer.luts_float.shape[0])) :
        for i_in in range(0, min(ax2.shape[0], lut_layer.luts_float.shape[1])) :
            ax2[i_in, i_out].plot(lut_xlables, lut_layer.luts_float[i_out][i_in], linewidth=1.0)
            
            ax_entries = ax2[i_in, i_out].twinx() 
            ax_entries.scatter(lut_xlables, lut_layer.entries_hist[i_in], linewidth=1.0, s=9, color='tab:red')
            ax2[i_in, i_out].set_title('lut_layer.luts_float[' + str(i_out) + '][' + str(i_in) + ']', fontdict={'fontsize': 6})

if lut_nn :
    plot_luts(layer0)
    plot_luts(layer1)

#plt.margins(0.2)
plt.subplots_adjust(hspace=0.4, wspace=0.17, left = 0.035, bottom = 0.045, top = 0.95)
plt.show()



