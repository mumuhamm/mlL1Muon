import tensorflow as tf
from tensorflow import keras
import LutLayer

class BinaryLutNetwork(tf.keras.Model):
    def __init__(self):
        super(BinaryLutNetwork, self).__init__(name='')
        
        num_address_bits = 4 
        num_outputs = 4
        self.layer0 = LutLayer.BinaryLutLayer(num_address_bits, num_outputs)

        self.layer1 = LutLayer.BinaryLutLayer(num_address_bits, 1)

    def call(self, input_tensor, training=False):
        x = self.layer0(input_tensor)
        y = self.layer1(x, training=training)

        return y

model = keras.Sequential()
