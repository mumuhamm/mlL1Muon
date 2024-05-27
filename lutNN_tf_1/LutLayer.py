import tensorflow as tf
from tensorflow import keras

#import numpy as np

tf.random.set_seed(12)

#luts should has binary values, 0 or 1
@tf.custom_gradient
def get_lut_values(x, luts_float, luts_int) :
    #addresses = tf.reshape(x, shape = [self.num_outputs, self.num_address_bits] ) this should be done in layer
    x = tf.cast(x, tf.int32) #<<<<<<<<<<<<<<<<<<<<<
    print("\nget_lut_values: x:", x)
        
    num_address_bits = x.shape[1]
    
    print("get_lut_values.num_address_bits", num_address_bits)
    
    print("tf.range(0, num_address_bits, 1)", tf.range(0, num_address_bits, 1))
    print("2 ** tf.range(0, num_address_bits, 1)", 2 ** tf.range(0, num_address_bits, 1))
    
    addres_kernel = 2 ** tf.range(0, num_address_bits, 1) #, shape = [num_address_bits, 1], dtype = x.dtype)
    addres_kernel =  tf.reshape(addres_kernel, shape = [num_address_bits, 1])
    print("addres_kernel", addres_kernel)
    addresses = tf.matmul(x, addres_kernel, transpose_b=False)
    print('\nget_lut_values addresses:', addresses) 
    
    #y = tf.round(tf.gather(luts_int, addresses, axis = 1, batch_dims=1) )
    y = tf.gather(luts_int, addresses, axis = 1, batch_dims=1)
    print("\nget_lut_values: y - lutValues:", y)

    def grad_dluts(dy):
        #indices i.e. lut addresses (as in the event) to use it for the SparseTensor
        #                     [consecutive numbers from 0 to lut_count, addresses]
        indices = tf.concat( [tf.reshape(tf.range(0, luts_int.shape[0], 1, dtype=tf.int64), addresses.shape), tf.cast(addresses,  dtype=tf.int64)], axis = 1 )
        
        print("\n\ngrad_dluts indices", indices)
        print("values", tf.reshape(dy, (indices.shape[0]) ) )
        
        #the result is a SparseTensor composed with the indices and the dy
        result =  tf.SparseTensor(indices = indices, 
                                  values = tf.reshape(dy, (indices.shape[0]) ), dense_shape = luts_int.shape)
        
        print("grad_dluts result", tf.sparse.to_dense(result) )
        return result
        #todo the sign of depending on the the lut value (0 or 1) must be included, either here or when updating the luts_int 
        
    def grad_dx(dy):
        gradients = []
        ones = tf.ones(shape = addresses.shape, dtype = addresses.dtype)
        print("\n\ngrad_dx num_address_bits", num_address_bits)
        #fliping one by one each bit in the addresses
        for i in range(0, num_address_bits, 1) :
            addresses_updated = tf.bitwise.bitwise_xor(addresses, ones)
            #print ("aaaa", (y != tf.gather(luts_int, addresses_updated, axis = 1, batch_dims=1) ))
            #the gradient is whether the value taken from the luts with the updated addresses changed or not * dy
            #dy is the cost related to changing the lut output value 
            gradients.append( tf.abs(tf.cast((y - tf.gather(luts_int, addresses_updated, axis = 1, batch_dims=1)), dtype=tf.float32)) * dy)
            ones = 2 * ones
               
        print ("grad_dx gradients", gradients)                 
        return tf.stack(gradients)
        
    def grad_empty(dy):
        return dy
        
    def grad(dy):
        print("\n\ngrad <<<<<<<<<<<<<<<<<<<<<<<<,")
        print("dy", dy)
         
        return (grad_dx(dy), grad_dluts(dy), grad_empty(dy) )
       
    #y must be casted to float, otherwise GradientTape.gradient does not work
    return tf.cast(y, dtype=tf.float32), grad 
###################################################################################################

class LutClip(keras.constraints.Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, max, min):
        self.max = max
        self.min = min

    def __call__(self, p):
        return keras.backend.clip(p, self.min, self.max)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

###################################################################################################

class BinaryLutLayer(keras.layers.Layer):
    def __init__(self, name, num_address_bits, num_outputs, min_out_val = 0, max_out_val = 1, out_val_offset = 0.5):
        super(BinaryLutLayer, self).__init__(name = name)
        self.num_address_bits = num_address_bits
        self.num_outputs = num_outputs
        self.min_out_val = min_out_val
        self.max_out_val = max_out_val
        self.out_val_offset = out_val_offset
        self.addres_kernel = tf.constant(2 ** tf.range(0, num_address_bits, 1), shape = [num_address_bits, 1], dtype= tf.int32)
        print("BinaryLutLayer.__init__: addres_kernel", self.addres_kernel)


        initializer = tf.keras.initializers.TruncatedNormal(mean = (self.min_out_val + self.max_out_val)/2. - self.out_val_offset, 
                                                            stddev = (self.max_out_val - self.min_out_val)/4.)
        
        #luts[iNode = iOutput][iAddr]       
        self.luts_float = self.add_weight(name + ".luts_float",
                                  shape=[ self.num_outputs, #TODO rather do it based on the input size, (int(input_shape[-1]) /
                                          2 ** self.num_address_bits ], 
                                  dtype  = tf.float32,
                                  initializer = initializer,
                                  trainable = True,
                                  #constraint = LutClip(self.min_out_val, self.max_out_val)
                                  )
        
        self.luts_int = tf.Variable(initial_value = tf.cast(tf.round(self.luts_float + self.out_val_offset), tf.int8 ), 
                                    trainable = False,  name = "luts_int", )
        
        self.gradient = tf.Variable(initial_value = tf.zeros(shape = self.luts_float.get_shape(), dtype = tf.float32) , 
                                    trainable = False,  name = "gradient", )
        
        self.luts_sign = tf.Variable(initial_value= tf.sign(self.luts_float) , 
                                    trainable = False,  name = "luts_sign", )

    def build(self, input_shape):
        print(self.name, "\nBinaryLutLayer.build: luts_float:", self.luts_float)
        print(self.name, "\nBinaryLutLayer.build: luts_sign:", self.luts_sign)
        print(self.name, "\nBinaryLutLayer.build: luts_int:",   self.luts_int)

    def call(self, inputs):
        print('\nBinaryLutLayer.call: inputs', inputs)
        input_shaped = tf.reshape(inputs, shape = [self.num_outputs, self.num_address_bits] )
        print('\nBinaryLutLayer.call: input_shaped', input_shaped)
           
        return get_lut_values(input_shaped, self.luts_float, self.luts_int)
    
    def sum_gradient(self, gradient_tape, loss ):
        # Calculate gradients with respect to every luts
        print("sum_gradient", self.name, "- loss", loss)
        grad_sparse = gradient_tape.gradient(loss, self.trainable_variables)

        #print("sum_gradient", self.name, "- grad_sparse", grad_sparse)


        for g in grad_sparse:
            print("\ng:", tf.sparse.to_dense(g))
            self.gradient.assign_add(tf.multiply(self.luts_sign, tf.sparse.to_dense(g) ))
        
        #print("\nlayer.trainable_variables", [var for var in self.trainable_variables])
        print("\nsum_gradient", self.name, " - self.gradient ", self.gradient)
        
        
    def apply_gradient(self, optimizer):    
        print ("self.trainable_variables", self.trainable_variables)
        optimizer.apply_gradients(zip([self.gradient], self.trainable_variables), experimental_aggregate_gradients=False)    
        
        self.luts_int.assign(tf.cast(tf.round(self.luts_float + self.out_val_offset), tf.int8) )
        self.gradient.assign(tf.zeros(shape = self.luts_float.get_shape(), dtype = tf.float32) )
        self.luts_sign.assign(tf.sign(self.luts_float) )
        
        print ("self.trainable_variables", self.trainable_variables)
 
###################################################################################################

#test
num_address_bits = 4 
num_outputs = 4
layer0 = BinaryLutLayer("layer0", num_address_bits, num_outputs)

layer1 = BinaryLutLayer("layer1", num_address_bits, 1)

#input = tf.ones([num_address_bits * num_outputs], dtype = tf.int32)
input_data = tf.random.uniform(shape=[num_address_bits * num_outputs], minval=0, maxval=2,  dtype = tf.float32)
y_true = 1

print("\ninput_data", input_data)

optimizer = tf.keras.optimizers.SGD(learning_rate=1., momentum=0., )

#layer0_result = layer0(input = input_data)
#layer1_result = layer1(input = layer0_result)

print("building model")

model = keras.Sequential()
model.add(tf.keras.Input(shape = input_data.shape))
model.add(layer0)
model.add(layer1)

model(input_data)
iterations = 1
#loss = tf.ones(shape = [iterations, 1]#
#loss tf.TensorArray
#print("\nloss", loss)

for i in range(0, iterations, 1) :
    with tf.GradientTape(persistent = True) as tape:
        result = model(input_data)
        loss = tf.losses.mean_squared_error(y_true, result)
        print("model result", result)

    #print('\nlayer1.trainable_variables ', layer1.trainable_variables)

    #print("\nwatched_variables", [var.name for var in tape.watched_variables()])

    for layer in model.layers :
        layer.sum_gradient(tape, loss)
    del tape

for layer in model.layers :
    layer.apply_gradient(optimizer)


#layer1.trainable_variables[0].assign_add(grad[0])

print("\nlayer.trainable_variables", [var for var in layer1.trainable_variables])
