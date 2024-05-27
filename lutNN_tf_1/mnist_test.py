import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras

class BinaryInputLayer(keras.layers.Layer):
    #self.indices
    
    def __init__(self, name, num_outputs, min_out_val = 0, max_out_val = 1):
        super(BinaryInputLayer, self).__init__(name = name)
        self.num_outputs = num_outputs
        self.min_out_val = min_out_val
        self.max_out_val = max_out_val
        
        #self.indices = tf.random.uniform(shape = [num_outputs], minval= 0, maxval = num_inputs, dtype=tf.dtypes.int32 )
        #tf.constant(value, dtype = tf.int32, shape, name = "indices")
        
    def build(self, input_shape):
        print(self.name, "BinaryInputLayer.build: input_shape:", input_shape)
        self.indices = tf.random.uniform(shape = [self.num_outputs], minval= 0, maxval = input_shape[0], dtype=tf.dtypes.int32 )
        print(self.name, "BinaryInputLayer.build: indices:", self.indices)

    def call(self, inputs):
        print('\n BinaryInputLayer.call: input', input)
        x = tf.gather(inputs, self.indices) #, axis = 1, batch_dims=1  
        print('\n BinaryInputLayer.call: input', x)
        return tf.greater_equal(x, tf.constant([64], dtype=tf.uint8) )

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

print ("x_train[0]", x_train[0])

# Add a channels dimension
#x_train = x_train[..., tf.newaxis].astype("float32")
#x_test = x_test[..., tf.newaxis].astype("float32")

print ("x_train[0]", x_train[0])

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

EPOCHS = 1

binaryInputLayer = BinaryInputLayer(name= "binaryInputLayer", num_outputs = 20)

for epoch in range(EPOCHS):
# Reset the metrics at the start of the next epoch
#train_loss.reset_states()
#   train_accuracy.reset_states()
#   test_loss.reset_states()
#   test_accuracy.reset_states()

    for images, labels in train_ds:
        #train_step(images, labels)
        #print ("images", images)
        #print ("labels", labels)
        for i in range(images.shape[0]) :
            print ("images[", i , "]", images[i])
            print (labels[i])
            
            image =  tf.reshape(images[i], [images.shape[1] * images.shape[2]])
            y = binaryInputLayer(image)
            print ("y", y)
            
            break
        break

    break
    #for test_images, test_labels in test_ds:
    #  test_step(test_images, test_labels)

#   print(
#     f'Epoch {epoch + 1}, '
#     f'Loss: {train_loss.result()}, '
#     f'Accuracy: {train_accuracy.result() * 100}, '
#     f'Test Loss: {test_loss.result()}, '
#     f'Test Accuracy: {test_accuracy.result() * 100}'
#   )
