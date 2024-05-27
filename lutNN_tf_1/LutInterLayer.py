import tensorflow as tf
from tensorflow import keras
import numpy as np
#from keras import backend
import math
import matplotlib.pyplot as plt

import io

#import numpy as np


# class LutClip(keras.constraints.Constraint):
#     '''Clips the weights incident to each hidden unit to be inside a range
#     '''
#     def __init__(self, maxVal, minVal):
#         self.max = maxVal
#         self.min = minVal
#
#     def __call__(self, p):
#         return keras.backend.clip(p, self.min, self.max)
#
#     def get_config(self):
#         return {'name': self.__class__.__name__,
#                 'c': self.c}

###################################################################################################

@tf.keras.utils.register_keras_serializable()        
class LutInterLayer(keras.layers.Layer):
    def __init__(self, name, 
                lut_size, 
                num_inputs, 
                num_outputs, 
                initializer = None,
                input_offset = None,
                hist_writer = None,
                write_lut_hist = False,
                min_lut_val = 0, max_lut_val = 1, out_val_offset = 0.5, 
                last_input_is_bias = False,
                **kwargs):
        super(LutInterLayer, self).__init__(name = name, **kwargs)
        self.lut_size = lut_size
        self.num_inputs = num_inputs
        if last_input_is_bias :
            self.num_inputs = num_inputs -1
        self.num_outputs = num_outputs
        self.min_lut_val = min_lut_val
        self.max_lut_val = max_lut_val
        self.out_val_offset = out_val_offset
        self.last_input_is_bias = last_input_is_bias
        
        if input_offset == None :
            self.input_offset = lut_size/2.
        else :    
            self.input_offset = input_offset

        self.train_counter = 0
     
        self.luts_float = self.add_weight(name + ".luts_float",
                                  shape=[self.num_inputs, 
                                         lut_size,
                                         self.num_outputs ], 
                                  dtype  = tf.float32,
                                  initializer = initializer,
                                  trainable = True,
                                  #constraint = LutClip(self.min_out_val, self.max_out_val)
                                  )
        
        self.entries_hist = tf.Variable(initial_value = tf.zeros(shape = [ self.num_inputs, lut_size ], 
                                                                 dtype = tf.int32) , trainable = False,  name = "entries_hist", ) 
        
        self.hist_writer = hist_writer
        
        #hist writing works only in eager mode
        self.write_lut_hist = write_lut_hist

        #super(LutInterLayer, self).__init__(**kwargs)
        print("constructing LutInterLayer ", name, "lut_size", lut_size, "num_inputs", num_inputs, "num_outputs", num_outputs, "input_offset", input_offset)
        print("write_lut_hist", self.write_lut_hist, "hist_writer", self.hist_writer)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "lut_size": self.lut_size,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "min_lut_val": self.min_lut_val,
            "max_lut_val": self.max_lut_val,
            "out_val_offset": self.out_val_offset,
            "write_lut_hist": self.write_lut_hist,
        })
        return config

        
    def build(self, input_shape):
        #print(self.name, "\nLutInterLayer.build: luts_float:", self.luts_float)
        print(self.name, "\nLutInterLayer.build: luts_float:", self.luts_float.name, "shape", self.luts_float.shape )

    def call(self, inputs, training=None):
        #print('\nLutInterLayer.call: ', self.name, 'inputs[0]') 
        #tf.print(inputs[0], summarize=-1) 

        #print('\nLutInterLayer.call:', self.name)
        
        #print('inputs', inputs)
        
        if self.last_input_is_bias :
            #x = tf.slice(inputs, begin=[0, 0], size=[inputs.shape[0], self.num_inputs-1])
            x = inputs[:, :-1] 
            #bias = tf.slice(inputs, begin=[0, self.num_inputs-1], size=[inputs.shape[0], 1])
            bias = inputs[:, -1:] 
        else :
            x = inputs
            #bias = 0
        
        
        x = x + self.input_offset
        x = tf.clip_by_value(x, clip_value_min = 0, clip_value_max = self.lut_size - 1.0001)
        
        #print("x:", x)
        #transpose is needed becasue gather uses the first dimension of address tensor od something like batch dim
        #i.e. it trhreads it as a set of addresses for a given LUT
        x = tf.transpose(x)
        #print("transpose(x):", x)
    
        if self.write_lut_hist :
            #entries_hist_ = tf.histogram_fixed_width_bins(x[0], value_range = (0, float(luts_float.shape[2]) ), nbins=luts_float.shape[2])
            #in principle it should be  value_range = (0, float(self.lut_size-1)), nbins = self.lut_size-1, but it does not matter rather
            entries = tf.map_fn(fn = lambda d : tf.histogram_fixed_width(d, value_range = (0, float(self.lut_size)), nbins = self.lut_size),
                                     elems = x,
                                     fn_output_signature = tf.TensorSpec(shape=[None], dtype=tf.int32))
            
            #tf.print(self.name, "entries", entries, summarize=128)
            if training == True : 
                self.entries_hist.assign(entries) 
            else :  
                self.entries_hist.assign_add(entries) # when doing these histograms during model evaluation, summing all batches  
            #print(self.name, "entries_hist", self.entries_hist)
         
            if training == True and self.train_counter % 10 == 0 and self.hist_writer != None :
                #print(self.name + '_entries_in_0', "self.train_counter", self.train_counter)
                #print("x[0]", x[0])
                with self.hist_writer.as_default(step = self.train_counter) :
                    #tf.summary.histogram(self.name + '_entries_input_0', x[0]) #, family = self.name
                    tf.summary.image(self.name + "luts", plot_to_image(self.plot_luts()))
  
            self.train_counter += 1        
            
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
            #print("lut", lut)
            lutVals0 = tf.gather(params = lut, indices=x0, axis=1, batch_dims=1) 
            #print("lutVals0", lutVals0)
         
            lutVals1 = tf.gather(params = lut, indices=x1, axis=1, batch_dims=1) 
            #print("lutVals1", lutVals1)
            
            #fractionalPart = tf.expand_dims(x - tf.cast(x0, tf.float32), axis=-1)
            #print("fractionalPart", fractionalPart)
            #interValues = (lutVals1 - lutVals0) * tf.math.floormod(x, 1) + lutVals0
            interValues = (lutVals1 - lutVals0) * tf.expand_dims(x - tf.cast(x0, tf.float32), axis=-1) + lutVals0
            #interValues = (lutVals1 - lutVals0) * (x - tf.cast(x0, tf.float32)) + lutVals0
         
         
            #print("interValues", interValues)
            lutSum = tf.reduce_sum(interValues, axis=0)
            #print("lutSum", lutSum)
            return lutSum
         
        #outValues = tf.map_fn(fn = getOutValue,
        #      elems = self.luts_float,
        #      fn_output_signature = tf.TensorSpec(shape=[None], dtype=tf.float32))
    
        outValues = getOutValue(self.luts_float)
        #outValues = tf.transpose(outValues)
        #print("outValues", outValues)
           
        if self.last_input_is_bias :
            outValues = outValues + bias   
           
        return outValues

    # this ploting works only in eager mode, i.e. model.compile(optimizer=optimizer, loss='mse', run_eagerly=True) 
    def plot_luts(self, path="") :
        print(self.name, "plot_luts")
                                 #(y,x)                        # x y    
        fig, ax = plt.subplots(8*4, 4, sharex=True, figsize=(18, 12 *4)) 
        
        for x in ax:
            for y in x:
                y.tick_params(axis='y', colors='blue', labelsize=8)
        
        lut_xlables =  list(range(self.lut_size))
        
        #print("lut_layer.entries_hist", lut_layer.entries_hist)
        #luts_float[num_outputs][num_inputs][lut_size]
        
        i_in_offset = 0
        if self.num_inputs > 500 :
            i_in_offset = 200
        for i_out in range(0, min(ax.shape[1], self.num_outputs)) :
            for i_in in range(0, min(ax.shape[0], self.num_inputs)) :
                lut = tf.slice(self.luts_float, begin=[i_in, 0, i_out], size=[1, self.lut_size, 1])
                lut = tf.reshape(lut, shape=[self.lut_size])
                #print("lut", lut)
                ax[i_in, i_out].plot(lut_xlables, lut, linewidth=1.0)
                ax[i_in, i_out].tick_params(axis='y', colors='blue', labelsize=8)
                
                ax_entries = ax[i_in, i_out].twinx() 
                #ax_entries.scatter(lut_xlables, lut_layer.entries_hist[i_in], linewidth=1.0, s=9, color='tab:red')
                ax_entries.step(lut_xlables, self.entries_hist[i_in + i_in_offset], linewidth=1.0, color='tab:red', where='post')
                ax[i_in, i_out].spines['right'].set_color('red') 
                ax_entries.spines['right'].set_color('red') 
                ax_entries.set_yscale("log")
                ax_entries.tick_params(axis='y', colors='red', labelsize=8)
                ax[i_in, i_out].set_title(self.name + '.lut_out' + str(i_out) + '_in_' + str(i_in), fontdict={'fontsize': 6})
        
        plt.savefig(path + self.name + "_luts.png", bbox_inches="tight")
        print("LutInterLayer.plot_luts(): creating", path + self.name + "_luts.png")
        return fig        

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class LutInitializerLinear(tf.keras.initializers.Initializer):

    def __init__(self, maxLutVal, initSlopeMin, initSlopeMax, lutRangesCnt):
        self.maxLutVal = maxLutVal
        self.initSlopeMin = initSlopeMin
        self.initSlopeMax = initSlopeMax
        self.lutRangesCnt = lutRangesCnt

    def __call__(self, shape, dtype=None) :
        rng = np.random.default_rng() #2021
        
        luts_float = np.ndarray(shape=shape)
        lutSize = luts_float.shape[1]
        num_inputs = luts_float.shape[0]
        num_outputs = luts_float.shape[2]
        
        #luts_float[i_in][iAddr][i_out]
        for i_out in range(0, num_outputs) :
            for i_in in range(0, num_inputs) :
                b1 = rng.uniform(low = self.initSlopeMin, high = self.initSlopeMax)
                s1 = rng.integers(0, 2)
                
                if s1 > 0 :
                    b1 *= -1;

                b0 = -b1 * lutSize / self.lutRangesCnt / 2.;
                offset = b0;
                if offset * b1 > 0 :
                    offset = -offset
                
                val = offset;
                lutRange = lutSize / self.lutRangesCnt;
                for iAddr  in range(0, lutSize) :
                    if iAddr % lutRange == 0 :
                        val = offset
                        
                    luts_float[i_in][iAddr][i_out] = val;
                    if iAddr == lutSize -1 :
                        luts_float[i_in][iAddr][i_out] = 0 #TODO f last_input_is_bias and the last addres is for the no_hit then it is better to have 0 here 
                        
                    if val > self.maxLutVal :
                        luts_float[i_in][iAddr][i_out] = self.maxLutVal;
                    elif val < -self.maxLutVal :
                        luts_float[i_in][iAddr][i_out] = -self.maxLutVal;

                    val += b1;

        return tf.convert_to_tensor(luts_float, dtype=dtype)
                

    def get_config(self):  # To support serialization
        return {
            'maxLutVal' : self.maxLutVal, 
            'initSlopeMin' : self.initSlopeMin,
            'initSlopeMax' : self.initSlopeMax,
            'lutRangesCnt' : self.lutRangesCnt,
            'last_input_is_bias' : self.last_input_is_bias,
        }
    