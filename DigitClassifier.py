import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

file = open('images.csv', 'r')
lines = csv.reader(file)
dataset = list(lines)
images = np.array(dataset).astype('uint8')
print(images[0])

# print(len(X))
# # Plot
pixels = np.array(images[5999])
pixels = pixels.reshape((28, 28))

plt.imshow(pixels, cmap='gray')
plt.show()
