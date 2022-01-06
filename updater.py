# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# load and evaluate a saved model
import tensorflow as tf
from numpy import loadtxt
from keras.models import load_model
import matplotlib.pyplot as plt
 
# load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# load model
model = tf.keras.models.load_model('model.h5')

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x=x_train,y=y_train, epochs=10)

# summarize model.
# model.summary()

# evaluate the model
score = model.evaluate(x_test, y_test, verbose = 0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# image_index = 1
# plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
# pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
# print(pred.argmax())
# plt.show()