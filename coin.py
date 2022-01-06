# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# load and evaluate a saved model
import tensorflow as tf
from numpy import loadtxt
from keras.models import load_model
import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

def nothing(x):
    pass
 
# load dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(y_test)
# Reshaping the array to 4-dims so that it can work with the Keras API
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# input_shape = (28, 28, 1)

# # Making sure that the values are float so that we can get decimal points after division
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# # Normalizing the RGB codes by dividing it to the max RGB value.
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print('Number of images in x_train', x_train.shape[0])
# print('Number of images in x_test', x_test.shape[0])

# load model
model = tf.keras.models.load_model('adadelta.h5')

model.compile(optimizer='adadelta', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#-----------------------------------------------------------
def run(path):
# path = 'D:/Folder/Kuliah/KK  E Smt. 5/DR_FPKKE/nnimg/12.jpg'
    image = cv2.imread(path)
    image = cv2.resize(image,(int(image.shape[1]/3),int(image.shape[0]/3)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,20)

    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=4) 

    # invert = (255-thresh)
    invert = cv2.bitwise_not(thresh)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(invert,connectivity=8)
    # print(labels[2])
    sizes = stats[:, -1]
    indexes_group = np.argsort(-stats[:, cv2.CC_STAT_AREA])

    num = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(1,len(indexes_group)):
        img2 = np.zeros(labels.shape)
        img2[labels == indexes_group[i]] = 255
        
        bbox = cv2.boundingRect(np.uint8(img2))
        bbox = list(bbox)

        img3 = img2[bbox[1]:bbox[3]+bbox[1],bbox[0]:bbox[0]+bbox[2]]

        #reassign to square size
        #Getting the bigger side of the image
        s = max(img3.shape[0:2])
        #Creating a dark square with NUMPY  
        squared_img2 = np.zeros((s,s),np.uint8)
        #Getting the centering position
        ax, ay = (s - img3.shape[1])//2, (s - img3.shape[0])//2
        #Pasting the 'image' in a centering position
        squared_img2[ay:img3.shape[0]+ay,ax:ax+img3.shape[1]] = img3

        tes = cv2.resize(squared_img2, (28,28))
        tes = tes.astype(np.float32)
        pred = model.predict(tes.reshape(1,28,28,1))

        # cv2.imshow(str(i),tes)
        # pred = model.predict(tes.reshape(1,28,28,1))
        # print(pred.argmax())

        # put in list
        num.append([bbox,pred.argmax()])

        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (36,255,12), 2)
        cv2.putText(image,str(pred.argmax()),(bbox[0]+4,bbox[1]+15), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        # cv2.waitKey()

    #mengurutkan angka berdasarkan posisi (x)
    actual_ = actual[int(filename[:-4])]
    print(actual_)
    num = sorted(num,key=lambda l:l[0])
    predicted=[]
    val = ''
    for i in num:
        predicted.append(i[1])
        # pred.append(i[i])
        val += str(i[1])
    # print(predicted)
    # print(val)
    
    npredicted = predicted[:len(actual_)].copy()
    for i in npredicted:
        predi.append(i)
    # predi = predi +  npredicted
    print(npredicted)
    
    akurasi = accuracy_score(actual_, npredicted)
    total_akurasi.append(akurasi)
    print ('Accuracy Score :' + str(akurasi))



    bbox = cv2.boundingRect(np.uint8(invert))
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (36,255,12), 1)
    cv2.putText(image,val,(bbox[0],bbox[1]-5), font, 1, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite('new_'+path[45:],image)

    # cv2.imshow('new_'+path[45:], image)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('invert', invert)

# actual = [[2,0,1,9],
#             [4,4],
#             [1,2,2],
#             [9,5,3,4,7],
#             [7,4,1,6],
#             [1,0,5],
#             [6,8,3],
#             [6],
#             [3],
#             [1,2,3,4,5],
#             [1,2,3,4,5,6,7,8,9],
#             [1,2,3,4,5,6,7,8,9],
#             [1,2,3,4,5,6,7,8,9],
#             [1,2,3,4,5,6,7,8,9],]

actual = [[1,3,5,6],
            [4,8,2,0],
            [2,1,1,2]]

_actual = [1,3,5,6,4,8,2,0,2,1,1,2]

predi = []

total_akurasi = []

Paths = 'D:/Folder/Kuliah/KK  E Smt. 5/DR_FPKKE/nnimg'
image = []
for dirName, subdirList, fileList in os.walk(Paths):
    fileList.sort()
    for filename in fileList:
        path = dirName + "/" + filename
        run(path)

results = confusion_matrix(_actual, predi)
print(results)

print('Rata-rata Akurasi: %.3f%%' % (sum(total_akurasi)/(len(total_akurasi))))

cv2.waitKey()
# cv2.destroyAllWindows()