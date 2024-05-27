import tensorflow as tf
from tensorflow import keras

import datetime

import LutInterLayer
from math import sqrt

#tf.random.set_seed(1234) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,

run_eagerly = False

input_I = 1
input_F = 4

layer1_lut_size = 1 << input_I
layer1_neurons = 32
layer1_lut_I = 3
layer1_lut_F = 10

layer1_output_I = 3

layer2_input_I = layer1_output_I 
layer2_lut_size = 1 << layer2_input_I
layer2_neurons = 10 #8 #* 2 #9 if the charge output is used
layer2_lut_I = 5
layer2_lut_F = 10

layer3_input_I = 4
layer3_lut_size = 1 << layer3_input_I
layer3_neurons = 1
layer3_lut_I = 6
layer3_lut_F = 10

print("layer1_lut_size", layer1_lut_size)
print("layer2_lut_size", layer2_lut_size)
print("layer3_lut_size", layer3_lut_size)
 
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print("x_train", x_train)
#print("y_train", y_train)

#print(x_train[:1])
#print(y_train[:1])

x_train, x_test = x_train / 255.0 * layer1_lut_size, x_test / 255.0 * layer1_lut_size

num_inputs = 28 * 28

model = keras.Sequential()

if run_eagerly :
    hist_writer = tf.summary.create_file_writer(log_dir + "/lutnn_input_hist")
    write_lut_hist = True
else :
    hist_writer = None    
    write_lut_hist = False

print("building model")
#initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = layer2_lut_size/4./sqrt(num_inputs))
initializer = LutInterLayer.LutInitializerLinear(maxLutVal = 1<<(layer1_lut_I-1), initSlopeMin = 0.01, initSlopeMax = 0.1/8, lutRangesCnt = 1)
layer1 = LutInterLayer.LutInterLayer("layer1", lut_size = layer1_lut_size, num_inputs = num_inputs, num_outputs = layer1_neurons, input_offset= 0, initializer = initializer, hist_writer = hist_writer, write_lut_hist=write_lut_hist)

#initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = 1)
initializer = LutInterLayer.LutInitializerLinear(maxLutVal = 1<<(layer2_lut_I-1), initSlopeMin = 0.01, initSlopeMax = 0.1/8, lutRangesCnt = 1)
layer2 = LutInterLayer.LutInterLayer("layer2", lut_size = layer2_lut_size, num_inputs = layer1_neurons, num_outputs = layer2_neurons, initializer = initializer, hist_writer = hist_writer, write_lut_hist=write_lut_hist)

#initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = 1)
initializer = LutInterLayer.LutInitializerLinear(maxLutVal = 1<<(layer3_lut_I-1), initSlopeMin = 0.01, initSlopeMax = 0.1, lutRangesCnt = 1)
layer3 = LutInterLayer.LutInterLayer("layer3", lut_size = layer3_lut_size, num_inputs = layer2_neurons, num_outputs = 1, initializer = initializer, hist_writer = hist_writer, write_lut_hist=write_lut_hist)

print("building model")
#model.add(tf.keras.Input(shape = [num_inputs], name="inputs"))
model.add(tf.keras.layers.Flatten(input_shape=(28, 28), name="inputs"))
#model.add(tf.keras.layers.GaussianNoise(stddev=0.2))
#model.add(tf.keras.layers.Dropout(0.15))
model.add(layer1)
model.add(layer2)
#model.add(layer3)
model.add(tf.keras.layers.Activation('softmax'))

   
learning_rate=0.001

print (model.summary())

print("x_train.shape", x_train.shape)

predictions = model(x_train[:1]).numpy()

initial_learning_rate = learning_rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate,
decay_steps=1600/4,
decay_rate=0.99,
staircase=False)

#optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0., )
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule) #learning_rate
#optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
 
#optimizer='adam'
#print(x_train[0])
 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) #from_logits should be False is softmax layer is used
loss_fn(y_train[:1], predictions).numpy()
 
model.compile(optimizer = optimizer,
              loss = loss_fn,
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="mymodel_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    ),
    
    tf.keras.callbacks.TensorBoard(
    log_dir="tf_logs_mnist/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    histogram_freq=1,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="batch", #batch
)  # How often to write logs (default: once per epoch)

]
 
model.fit(x_train, y_train, epochs=40, batch_size = 64*4, callbacks=callbacks, validation_split=0.2, shuffle=True) #

#model.fit(x_train, y_train, epochs=7, validation_split=0.2, batch_size=1000, verbose=1)



print("model.evaluate, x_train.len ", x_train.__len__() )
 
model.evaluate(x_train,  y_train, verbose=2)

for layer in model.layers:
    #print(layer)
    if isinstance(layer, LutInterLayer.LutInterLayer):
        print("layer.write_lut_hist", layer.write_lut_hist)
        layer.write_lut_hist = True
        print("layer.write_lut_hist", layer.write_lut_hist)

print("model.evaluate, x_test.len ", x_test.__len__() )
model.run_eagerly=True
model.compile(optimizer = optimizer,
              loss = loss_fn,
              metrics=['accuracy'],
              run_eagerly=True)
model.evaluate(x_test,  y_test, verbose=2, batch_size=1000)

for layer in model.layers:
    if isinstance(layer, LutInterLayer.LutInterLayer):
        layer.plot_luts("mnist_plots/")
