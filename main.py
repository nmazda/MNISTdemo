# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import matplotlib.pyplot as plt

#from tensorflow.python.compiler.mlcompute import mlcompute
#tf.compat.v1.disable_eager_execution()
#mlcompute.set_mlc_device(device_name='gpu')
#print("is_apple_mlc_enabled %s" % mlcompute.is_apple_mlc_enabled())
#print("is_tf_compiled_with_apple_mlc %s" % #mlcompute.is_tf_compiled_with_apple_mlc())
#print(f"eagerly? {tf.executing_eagerly()}")
#print(tf.config.list_logical_devices())

# with tf.device("/gpu:0"):

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# Converting the integer data into decimal points [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Show all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('arrucary')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'test_accuracy', 'training_loss', 'test_loss'], loc='upper left')
plt.show()

loss, acc = model.evaluate(x_test, y_test)
print("Accuracy: {:5.2}%".format(100*acc))
