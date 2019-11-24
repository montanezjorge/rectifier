import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
seednumber = 8
from numpy.random import seed
seed(seednumber)
import tensorflow as tf
#tf.random.set_seed(seednumber)
from affine import Affine
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Input, Concatenate, Dot, Activation, Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Lambda
from tensorflow.keras.regularizers import l2, l1
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import CustomObjectScope, get_custom_objects
from tensorflow.keras.optimizers import Optimizer, Adam
import numpy as np
import random
import sys
import pickle
import warnings
import pytesseract
import cv2
import xml.etree.ElementTree as ET
import image_transformer
from image_transformer import *

def GenerateData():
    systemRandom = random.SystemRandom()

    iter = 0

    for dir in os.listdir("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images"):
        images=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_preprocessed", "ab")
        transforms=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms", "ab")            

        if iter == 5:
            break

        iter = iter + 1

        for r, d, f in os.walk(os.path.join("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images", dir)):
            for file in f:

                fileName = os.path.join(r, file)
                print(fileName)
                theta = systemRandom.uniform(-30.0, 30.0)
                phi = systemRandom.uniform(-30.0, 30.0)
                gamma = systemRandom.uniform(-30.0, 30.0)
                img = ImageTransformer(fileName, shape = (754, 1000))
                im, transform = img.rotate_along_axis(theta, phi, gamma)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                cv2.threshold(im ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                im = im / 255
                transform = np.linalg.inv(transform)
                im.tofile(images)
                transform.tofile(transforms)

        transforms.close()
        images.close() 

def PreprocessData():

    Y_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms", dtype=np.dtype("(9,)f8"))
    scalery = StandardScaler(copy=False)
    Y_train = scalery.fit_transform(Y_train)
    dump(scalery, 'C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\std_scalery.bin', compress=True)
    Y_train.tofile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_preprocessed")  

def ValidateTrainingData():
    Y_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_preprocessed", dtype=np.dtype("(9,)f8"))
    print(Y_train.shape)

    ID = 0

    while True:
        im = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_preprocessed", dtype=np.dtype("(754000,)f8"), count=1, offset=754000*ID*8)
        im = (im.reshape(754, 1000) * 255).astype(np.uint8)
        plt.imshow(im)
        plt.show()
        ID = ID + 1



def Train():
    hidden_layers = [layers.Dense(units=100, 
                                  activation='relu', 
                                  input_shape=[754000], 
                                  kernel_initializer=keras.initializers.glorot_uniform(), 
                                  bias_initializer=keras.initializers.glorot_uniform())]
    for i in range(16):
        hidden_layers.append(layers.Dense(units=100, 
                                          activation='relu',
                                          kernel_initializer=keras.initializers.glorot_uniform(), 
                                          bias_initializer=keras.initializers.glorot_uniform()))
    hidden_layers.append(layers.Dense(units=9, use_bias=False))

    model = keras.Sequential(hidden_layers)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_R_squared', verbose=1, patience=5, mode='max')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_R_squared', min_delta=0.00005, patience=12, verbose=1, mode='max', baseline=None, restore_best_weights=True)
    model.compile(loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                metrics=['accuracy', R_squared])

    def emptyPreprocessing(X, Y, scalerX, scalerY):
        return X, Y

    training_generator = DataGenerator("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_preprocessed", 
                                       "C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_preprocessed", 
                                       0, 59999, 754000, 9, None, None, emptyPreprocessing, 25)
    validation_generator = DataGenerator("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_preprocessed", 
                                       "C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_preprocessed",
                                       60000, 65000, 754000, 9, None, None, emptyPreprocessing, 25)

    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=50, use_multiprocessing=True, workers=0, verbose=1, callbacks=[reduce_lr, early_stopping], shuffle=True)
    model.save("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasmodel.h5")


def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    return tf.subtract(1.0, tf.divide(residual, total))


def ValidateTrainingWithKeras():

    nn = []
    with CustomObjectScope({'r_squared': R_squared}):
        nn = load_model("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasmodel")

    X_test = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_imagese.txt", dtype=np.dtype("(188000,)u1"))
    y_test = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_imagese.txt", dtype=np.dtype("(3,3)f8"))

    scalerx = load('C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\std_scalerx.bin')
    scalery = load('C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\std_scalery.bin')

    X_test = scalerx.transform(X_test)

    list = []

    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images\\imagese"):
        for file in f:
            list.append(os.path.join(r, file))

    #scaler = MinMaxScaler()

    for i in range(0, int(X_test.shape[0] / 2), 2):

        #X1 = scaler.fit_transform(X_test[i].reshape(1, -1))
        #X2 = scaler.fit_transform(X_test[i+1].reshape(1, -1))

        y_predicted1 = nn.predict(X_test[i].reshape(1, -1).astype(np.float32))[0]
        y_predicted2 = nn.predict(X_test[i+1].reshape(1, -1).astype(np.float32))[0]

        y_predicted1 = scalery.inverse_transform(y_predicted1)
        y_predicted2 = scalery.inverse_transform(y_predicted2)

        #scaler.fit_transform(y_test.reshape(y_test.shape[0], 9))

        #y_predicted1 = scaler.inverse_transform(y_predicted1.reshape(1, -1))
        #y_predicted2 = scaler.inverse_transform(y_predicted2.reshape(1, -1))

        fileName = list[i]
        print(fileName)
        img = Image.open(fileName)
        plt.imshow(np.asarray(img), cmap='gray')
        plt.show()  
        T_inv1 = np.linalg.inv(y_test[i])
        img_transformation1 = img.transform((img.size[0], img.size[1]), Image.AFFINE, data=T_inv1.flatten()[:6], resample=Image.BICUBIC)
        plt.imshow(np.asarray(img_transformation1), cmap='gray')
        plt.show()  
        img_result1 = img_transformation1.transform((img_transformation1.size[0], img_transformation1.size[1]), Image.AFFINE, data=y_predicted1.flatten()[:6], resample=Image.BICUBIC)
        plt.imshow(np.asarray(img_result1), cmap='gray')
        plt.show()  
        T_inv2 = np.linalg.inv(y_test[i+1])
        img_transformation2 = img.transform((img.size[0], img.size[1]), Image.AFFINE, data=T_inv2.flatten()[:6], resample=Image.BICUBIC)
        plt.imshow(np.asarray(img_transformation2), cmap='gray')
        plt.show()  
        img_result2 = img_transformation2.transform((img_transformation2.size[0], img_transformation2.size[1]), Image.AFFINE, data=y_predicted2.flatten()[:6], resample=Image.BICUBIC)
        plt.imshow(np.asarray(img_result2), cmap='gray')
        plt.show()

def GenerateSegmentationData():
    frames = open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\frames_full", "ab")
    boxes = open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\boxes_full", "ab")

    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\testDataset"):
        for file in f:
            if file.endswith(".avi"):
                vidcap = cv2.VideoCapture(os.path.join(r, file))
                success,image = vidcap.read()
                while success:
                    resize = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation= cv2.INTER_LANCZOS4)
                    array = np.asarray(resize)
                    array.tofile(frames)  
                    success,image = vidcap.read()
                file = file[:-4] + ".gt.xml"
                tree = ET.parse(os.path.join(r, file))
                root = tree.getroot()
                vector = np.zeros(9)
                for frame in root.iter("frame"):
                    if frame.attrib["rejected"] == "false":
                        vector[0] = 1
                    count = 1
                    for point in frame.iter("point"):
                        vector[count] = float(point.attrib["x"]) / 2 
                        vector[count + 1] = float(point.attrib["y"]) / 2
                        count = count + 2
                    vector.tofile(boxes)

def ShuffleSegementationData():
    X_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\frames", dtype=np.dtype("(518400,)u1"))
    y_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\boxes", dtype=np.dtype("(9,)f8"))

    X_train, y_train = shuffle(X_train, y_train)

    X_train.tofile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\frames_shuffled")
    y_train.tofile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\boxes_shuffled")

def TrainSegmentation():

    hidden_layers = [layers.Dense(units=100, 
                                activation='relu', 
                                input_shape=[518400], 
                                #kernel_regularizer = keras.regularizers.l2(0.001),
                                #bias_regularizer = keras.regularizers.l2(0.001),
                                #activity_regularizer = keras.regularizers.l1(0.001),
                                kernel_initializer=keras.initializers.glorot_uniform(), 
                                bias_initializer=keras.initializers.glorot_uniform())]
    for i in range(10):
        hidden_layers.append(layers.Dense(units=100, 
                                    activation='relu', 
                                    #kernel_regularizer = keras.regularizers.l2(0.001),
                                    #bias_regularizer = keras.regularizers.l2(0.001),
                                    #activity_regularizer = keras.regularizers.l1(0.001),
                                    kernel_initializer=keras.initializers.glorot_uniform(), 
                                    bias_initializer=keras.initializers.glorot_uniform()))
    hidden_layers.append(layers.Dense(units=9, use_bias=False))

    model = keras.Sequential(hidden_layers)
    model.compile(loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                metrics=['accuracy', R_squared])

    X_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\frames_processed_shuffled", dtype=np.dtype("(518400,)f8"))
    y_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\boxes_processed_shuffled", dtype=np.dtype("(9,)f8"))

    reduce_lr = keras.callbacks.ReduceLROnPlateau(verbose=1, patience=5)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=400, batch_size=100, validation_split=0.1, verbose=1, shuffle=True, callbacks=[reduce_lr, early_stopping])
    model.save("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\kerasmodel")

def CNN():

    data = Input(shape=(540, 960, 3), name="data")

    conv1 = Conv2D(96, 11, name="conv1",  
                strides=4, 
                padding='same',
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
                bias_initializer=keras.initializers.Constant(value=0),
                kernel_regularizer=keras.regularizers.l2(1),
                bias_regularizer=keras.regularizers.l2(0))(data)

    relu1 = Activation("relu", name="relu1")(conv1)

    pool1 = MaxPooling2D(pool_size=3, strides=2, name="pool1")(relu1)

    norm1 = BatchNormalization(name="norm1")(pool1)


    normalized1_1 = Lambda(lambda x: x[:, :, :, :48])(norm1)

    normalized1_2 = Lambda(lambda x: x[:, :, :, 48:])(norm1)

    conv2_1 = Conv2D(filters=128,
                     kernel_size=5,
                     activation='relu',
                     strides=2,
                     padding='same',
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
                     bias_initializer=keras.initializers.Constant(value=1),
                     kernel_regularizer=keras.regularizers.l2(1),
                     bias_regularizer=keras.regularizers.l2(0),
                     name='conv2_1')(normalized1_1)

    conv2_2 = Conv2D(filters=128,
                     kernel_size=5,
                     activation='relu',
                     strides=2,
                     padding='same',
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
                     bias_initializer=keras.initializers.Constant(value=1),
                     kernel_regularizer=keras.regularizers.l2(1),
                     bias_regularizer=keras.regularizers.l2(0),
                     name='conv2_2')(normalized1_2)

    conv2 = Concatenate(name='conv_2_merge')([conv2_1, conv2_2])

    relu2 = Activation("relu", name="relu2")(conv1)

    pool2 = MaxPooling2D(pool_size=3, strides=2, name="pool2")(relu2)

    norm2 = BatchNormalization(name="norm2")(pool2)

    conv3 = Conv2D(384, 3, name="conv3",  
            strides=1, 
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
            bias_initializer=keras.initializers.Constant(value=0),
            kernel_regularizer=keras.regularizers.l2(1),
            bias_regularizer=keras.regularizers.l2(0))(norm2)

    relu3 = Activation("relu", name="relu3")(conv3)

    relu3_1 = Lambda(lambda x: x[:, :, :, :192])(relu3)

    relu3_2 = Lambda(lambda x: x[:, :, :, 192:])(relu3)

    conv4_1 = Conv2D(filters=192,
                     kernel_size=3,
                     activation='relu',
                     strides=1,
                     padding='same',
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
                     bias_initializer=keras.initializers.Constant(value=1),
                     kernel_regularizer=keras.regularizers.l2(1),
                     bias_regularizer=keras.regularizers.l2(0),
                     name='conv4_1')(relu3_1)

    conv4_2 = Conv2D(filters=192,
                     kernel_size=3,
                     activation='relu',
                     strides=1,
                     padding='same',
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
                     bias_initializer=keras.initializers.Constant(value=1),
                     kernel_regularizer=keras.regularizers.l2(1),
                     bias_regularizer=keras.regularizers.l2(0),
                     name='conv4_2')(relu3_2)

    conv4 = Concatenate(name='conv_4_merge')([conv4_1, conv4_2])

    relu4 = Activation("relu", name="relu4")(conv4)

    relu4_1 = Lambda(lambda x: x[:, :, :, :128])(relu3)

    relu4_2 = Lambda(lambda x: x[:, :, :, 128:])(relu3)

    conv5_1 = Conv2D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     strides=1,
                     padding='same',
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
                     bias_initializer=keras.initializers.Constant(value=1),
                     kernel_regularizer=keras.regularizers.l2(1),
                     bias_regularizer=keras.regularizers.l2(0),
                     name='conv5_1')(relu3_1)

    conv5_2 = Conv2D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     strides=1,
                     padding='same',
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
                     bias_initializer=keras.initializers.Constant(value=1),
                     kernel_regularizer=keras.regularizers.l2(1),
                     bias_regularizer=keras.regularizers.l2(0),
                     name='conv5_2')(relu3_2)

    conv5 = Concatenate(name='conv_5_merge')([conv5_1, conv5_2])


    relu5 = Activation("relu", name="relu5")(conv5)

    pool5 = MaxPooling2D(pool_size=3, strides=2, name="pool5")(relu5)

    fc6 = Dense(4096,
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.005), 
                     bias_initializer=keras.initializers.Constant(value=1),
                     kernel_regularizer=keras.regularizers.l2(1),
                     bias_regularizer=keras.regularizers.l2(0),
                     name="fc6")(pool5)

    relu6 = Activation("relu", name="relu6")(fc6)

    fc7 = Dense(4096,
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.005), 
                     bias_initializer=keras.initializers.Constant(value=1),
                     kernel_regularizer=keras.regularizers.l2(1),
                     bias_regularizer=keras.regularizers.l2(0),
                     name="fc7")(relu6)

    relu7 = Activation("relu", name="relu7")(fc7)

    fc8 = Flatten()(relu7)
    pred = Dense(units=9, use_bias=False, name="pred")(fc8)

    model = Model(inputs=[data], outputs=[pred])
    model.summary()
    model.compile(loss='mse',
                   optimizer='adam',
                   metrics=['accuracy', R_squared])
    
    X_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\frames_full", dtype=np.dtype("(540, 960, 3)u1"))
    Y_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\boxes_full", dtype=np.dtype("(9,)f8"))

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_R_squared', verbose=1, patience=8, mode='max')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_R_squared', min_delta=0.0002, patience=15, verbose=1, mode='max', baseline=None, restore_best_weights=True)

    model.fit(X_train, Y_train, epochs=80, batch_size=10, validation_split=0.1, verbose=1, shuffle=True, callbacks=[reduce_lr, early_stopping])
    model.save("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\kerasmodel")


def TrainSegmentationWithGenerator():
    model = keras.Sequential()
    model.add(Dense(100, activation='relu', input_shape=[2073600], kernel_initializer=keras.initializers.glorot_uniform(),  bias_initializer=keras.initializers.glorot_uniform()))
    for i in range(32):
        model.add(Dense(100, activation='relu', kernel_initializer=keras.initializers.glorot_uniform(),  bias_initializer=keras.initializers.glorot_uniform()))
    model.add(Dense(9))

    model.compile(loss='mse',
        optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
        metrics=['accuracy', R_squared])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(verbose=1, patience=5)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    def preprocessing(X, Y, scalerX, scalerY):
        scalerX.partial_fit(X)
        scalerY.partial_fit(Y)
        X = scalerX.transform(X)
        Y = scalerY.transform(Y)
        return X, Y

    scalerX = StandardScaler(copy=False)
    scalerY = StandardScaler(copy=False)

    training_generator = DataGenerator("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\frames", 
                                       "C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\boxes", 
                                       0, 20000, 2073600, 9, scalerX, scalerY, preprocessing, 50)
    validation_generator = DataGenerator("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\frames", 
                                       "C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\boxes", 
                                       20000, 24000, 2073600, 9, scalerX, scalerY, preprocessing, 50)

    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=20, use_multiprocessing=True, workers=0, verbose=1, callbacks=[reduce_lr, early_stopping], shuffle=True)

    dump(scalerX, 'C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\std_scalerx_segmentation.bin', compress=True)
    dump(scalerY, 'C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\std_scalery_segmentation.bin', compress=True)

    model.save("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\kerasmodel")

def ValidateSegmentation():
    model = []
    with CustomObjectScope({'R_squared': R_squared}):
        model = load_model("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\kerasmodelBackup")

    #scalerX = load('C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\std_scalerx_segmentation.bin')
    #scalerY = load('C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\std_scalery_segmentation.bin')

    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\testDataset\\background05"):
        for file in f:
            if file.endswith(".avi"):
                p = os.path.join(r, file)
                print(p)
                vidcap = cv2.VideoCapture(p)
                success,image = vidcap.read()
                while success:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    resize = cv2.resize(gray, (int(gray.shape[1]/2), int(gray.shape[0]/2)), interpolation= cv2.INTER_LANCZOS4)
                    array = np.asarray(resize).reshape(1,540,960,1).astype(np.float)
                    test_predictions = model.predict(array)
                    #test_predictions = model.predict(scalerX.transform(array))
                    prediction = test_predictions[0]
                    prediction[1:] = prediction[1:] * 2
                    print(prediction)
                    pts = np.array([[prediction[1], prediction[2]],[prediction[3], prediction[4]],[prediction[5], prediction[6]],[prediction[7], prediction[8]]], np.int32)
                    cv2.polylines(image, [pts], True, (0, 0, 255))
                    cv2.imshow('Window', image)
                    cv2.waitKey(0)
                    success,image = vidcap.read()

    cv2.destroyAllWindows()

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X_src, Y_src, start, end, n_features, n_outputs, scalerX, scalerY, preprocessing, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.X_src = X_src
        self.Y_src = Y_src
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.shuffle = shuffle
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.preprocessing = preprocessing
        self.list_IDs = []
        for i in range(start, end):
            self.list_IDs.append(i)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty([self.batch_size, self.n_features])
        Y = np.empty([self.batch_size, self.n_outputs])

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i] = np.fromfile(self.X_src, dtype=np.dtype("(" + str(self.n_features) + ",)f8"), count=1, offset=self.n_features*ID*8)
            Y[i] = np.fromfile(self.Y_src, dtype=np.dtype("(" + str(self.n_outputs) + ",)f8"), count=1, offset=self.n_outputs*ID*8)

        return self.preprocessing(X, Y, self.scalerX, self.scalerY)

def ExtractIndividualFrames():
    frames = "C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Data\\frames"
    masks = "C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Data\\masks"

    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\testDataset"):
        for file in f:
            if file.endswith(".avi"):
                vidcap = cv2.VideoCapture(os.path.join(r, file))
                success,img = vidcap.read()
                count = 1
                coors = file[:-4] + ".gt.xml"
                tree = ET.parse(os.path.join(r, coors))
                root = tree.getroot()
                directory = os.path.basename(os.path.normpath(r))
                while success:  
                    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation= cv2.INTER_LANCZOS4)
                    cv2.imwrite(os.path.join(frames, directory + file[:-4] + "frame" + str(count) + ".png"), img)
                    pts = np.zeros(8)
                    frame = list(root.iter("frame"))[count-1]
                    index = 0
                    for point in frame.iter("point"):
                        pts[index] = round(float(point.attrib["x"]) / 2) 
                        pts[index + 1] = round(float(point.attrib["y"]) / 2)
                        index = index + 2
                    height,width,depth = img.shape
                    mask_img = np.zeros((height,width), np.uint8)
                    pts = np.array([[pts[0], pts[1]],[pts[2], pts[3]],[pts[4], pts[5]],[pts[6], pts[7]]], np.int32)
                    cv2.polylines(mask_img, [pts], True, 255)
                    masked_data = cv2.bitwise_and(img, img, mask=mask_img)
                    cv2.fillPoly(masked_data, [pts], (255, 255, 255, 255))
                    cv2.imwrite(os.path.join(masks, directory + file[:-4] + "mask" + str(count) + ".png"), masked_data)
                    count = count + 1
                    success,img = vidcap.read()                

import utils
from config import Config
from utils import *
from model import MaskRCNN
from visualize import *
import fileinput

class SegmentationConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "segmentation"
   

    # Train on 1 GPU and 1 images per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 540
    IMAGE_MAX_DIM = 960

class SegmentationDataset(Dataset):
    def load(self, start, end):
        self.add_class("segmentation", 1, "document")

        count = 1
        for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Data\\frames"):
            for file in f:
                if count <= start and count <= end:
                    self.add_image("segmentation", count, os.path.join(r, file))
                    count = count + 1

    def image_reference(self, image_id):

        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = cv2.imread(info["path"].replace("frame", "mask"))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask > 128
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

        return mask, np.ones([1], np.int32)

def TrainMaskRCNN():
    ROOT_DIR = "C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Data"

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    ## Local path to trained weights file
    #COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    ## Download COCO trained weights from Releases if needed
    #if not os.path.exists(COCO_MODEL_PATH):
    #    utils.download_trained_weights(COCO_MODEL_PATH)

    config = SegmentationConfig()    

    # Training dataset
    dataset_train = SegmentationDataset()
    dataset_train.load(501, 24889)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SegmentationDataset()
    dataset_val.load(1, 500)
    dataset_val.prepare()

    model = MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    #model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=15, layers='all')
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    model.keras_model.save_weights(model_path)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def RunMaskRCNN():

    ROOT_DIR = "C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Data"

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    config = SegmentationConfig()

    # Recreate the model in inference mode
    model = MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    #nn = []
    #with CustomObjectScope({'r_squared': R_squared}):
    #nn = load_model("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasmodel.h5", custom_objects={'R_squared': R_squared})

    #scalerx = load('C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\std_scalerx.bin')
    #scalery = load('C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\std_scalery.bin')

    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\sampleDataset\\input_sample\\background00"):
        for file in f:
            if file.endswith("magazine001.avi"):
                p = os.path.join(r, file)
                print(p)
                vidcap = cv2.VideoCapture(p)
                success,image = vidcap.read()
                while success:
                    plt.figure(figsize=(15, 15))
                    plt.subplot(1,2,1)
                    plt.imshow(image)

                    resize = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation= cv2.INTER_LANCZOS4)

                    results = model.detect([resize], verbose=1)

                    r = results[0]

                    #display_instances(resize, r['rois'], r['masks'], r['class_ids'], ['BG', 'document'], r['scores'])

                    roi = resize[r['rois'][0][0]:r['rois'][0][2], r['rois'][0][1]:r['rois'][0][3], :]
                    plt.subplot(1,2,2)
                    plt.imshow(roi)

                    #roi_input= cv2.resize(roi, (754, 1000), interpolation = cv2.INTER_CUBIC)
                    #roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2GRAY)
                    #roi_input = cv2.adaptiveThreshold(roi_input, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    ##plt.subplot(1,4,4)
                    ##plt.imshow(roi_gray, cmap='gray')
                    #plt.subplot(2,2,3)
                    #plt.imshow(roi_input)

                    #X = roi_input.reshape(1, -1) / 255

                    #y_predicted = nn.predict(X.astype(np.float32))[0]
                    #y_predicted = scalery.inverse_transform(y_predicted)
                    ##bsize = 200
                    ##roi = cv2.copyMakeBorder(roi, top=bsize, bottom=bsize, left=bsize, right=bsize, borderType=cv2.BORDER_CONSTANT)
                    #img_result = cv2.warpPerspective(roi, y_predicted.reshape(3, 3), (roi.shape[1] + 100, roi.shape[0] + 100))
                    #plt.subplot(2,2,4)
                    #plt.imshow(img_result)
                    plt.show()  

                    success,image = vidcap.read()

def Test():
    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\sampleDataset\\input_sample\\background00"):
        for file in f:
            if file.endswith("magazine001.avi"):
                p = os.path.join(r, file)
                print(p)
                vidcap = cv2.VideoCapture(p)
                success,image = vidcap.read()
                for i in range(29):
                    success,image = vidcap.read()

                plt.subplot(1,2,1)
                plt.imshow(image)
                img = ImageTransformer(p, None, image)
                img.image = image
                img.height = image.shape[0]
                img.width = image.shape[1]
                im, tranform = img.rotate_along_axis(theta=30)
                plt.subplot(1,2,2)
                plt.imshow(im)
                plt.show()            
                    
if __name__ == '__main__':
    GenerateData()