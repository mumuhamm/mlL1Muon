import tensorflow as tf
import datetime

#tf.random.set_seed(1234) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
 
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print("x_train", x_train)
#print("y_train", y_train)

#print(x_train[:1])
#print(y_train[:1])

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(32, activation='relu'),
  #tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


print (model.summary())

print("x_train.shape", x_train.shape)

predictions = model(x_train[:1]).numpy()

 
#print(x_train[0])
 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) #from_logits should be False is softmax is used
loss_fn(y_train[:1], predictions).numpy()
 
model.compile(optimizer='adam',
              loss=loss_fn,
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
 
model.fit(x_train, y_train, epochs=40, callbacks=callbacks, validation_split=0.2, shuffle=True) #

#model.fit(x_train, y_train, epochs=7, validation_split=0.2, batch_size=1000, verbose=1)

print("model.evaluate, x_train.len ", x_train.__len__() )
 
model.evaluate(x_train,  y_train, verbose=2)

print("model.evaluate, x_test.len ", x_test.__len__() )
 
model.evaluate(x_test,  y_test, verbose=2)


