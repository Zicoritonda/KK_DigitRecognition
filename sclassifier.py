# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# load and evaluate a saved model
import tensorflow as tf
from numpy import loadtxt
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
 
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
model = tf.keras.models.load_model('adam.h5')

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# summarize model.
model.summary()

# evaluate the model
score = model.evaluate(x_test, y_test, verbose = 0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

tes = cv.imread("Coin/masked/102058.png")
print(tes.shape)
tes = cv.cvtColor(tes, cv.COLOR_BGR2GRAY)
# image_index = 3
# tes = np.array([cv.resize(tes, (28,28))])
tes = cv.resize(tes, (28,28))
cv.imshow("tes",tes)
cv.waitKey()
print(tes)
plt.imshow(tes,cmap='Greys')
pred = model.predict(tes.reshape(1,28,28,1))
print(pred.argmax())
plt.show()