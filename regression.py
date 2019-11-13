import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
seednumber = 8
from numpy.random import seed
seed(seednumber)
import tensorflow as tf
tf.random.set_seed(seednumber)
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

def GenerateData():
    systemRandom = random.SystemRandom()

    iter = 0
    dir_num = 4

    for dir in os.listdir("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images"):
        transforms=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_all", "ab")
        images=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_all", "ab")
        names=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\imageNames_all", "a")

        if iter == 10:
            break

        iter = iter + 1
            

        for r, d, f in os.walk(os.path.join("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images", dir)):
            for file in f:

                try:
                    fileName = os.path.join(r, file)
                    print(fileName)
                    print(fileName, file=names)
                    img = Image.open(fileName)
                    img = img.resize((376, 500), Image.LANCZOS)
                    x = systemRandom.uniform(0.0, 30.0)
                    y = systemRandom.uniform(-40.0, 40.0)
                    T_shear = np.array(Affine.shear(x, y)).reshape(3, 3)
                    T_shear_inv = np.linalg.inv(T_shear)
                    img_shear_transformation = np.asarray(img.transform((376, 500), Image.AFFINE, data=T_shear_inv.flatten()[:6], resample=Image.BICUBIC))

                    angle = systemRandom.uniform(-40.0, 60.0)
                    T_rotation = np.array(Affine.rotation(angle)).reshape(3, 3)
                    T_rotation_inv = np.linalg.inv(T_rotation)
                    img_rotation_transformation = np.asarray(img.transform((376, 500), Image.AFFINE, data=T_rotation_inv.flatten()[:6], resample=Image.BICUBIC))

                    T_shear.tofile(transforms)
                    img_shear_transformation.tofile(images)

                    T_rotation.tofile(transforms)
                    img_rotation_transformation.tofile(images)

                except Exception as exception:
                    print(exception)

        transforms.close()
        images.close()
        names.close()

def ValidateData():
    transforms = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_imagesa.txt", dtype=np.dtype("(3,3)f8"))
    images = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_imagesa.txt", dtype=np.dtype("(250, 189)u1"))

    for i in range(transforms.shape[0]):
        img = Image.fromarray(images[i])
        plt.imshow(np.asarray(img), cmap='gray')
        plt.show() 
        img_transformed = np.asarray(img.transform((img.size[0], img.size[1]), Image.AFFINE, data=transforms[i].flatten()[:6], resample=Image.BICUBIC))
    
        plt.imshow(np.asarray(img_transformed), cmap='gray')
        plt.show()  

def Test():
    systemRandom = random.SystemRandom()
    fileName = "C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images\\imagesa\\a\\a\\a\\aaa13d00\\514409402_514409407.tif"
    img = Image.open(fileName)    
    plt.imshow(np.asarray(img), cmap='gray')
    plt.show()  

    x = systemRandom.uniform(0.0, 30.0)
    y = systemRandom.uniform(-40.0, 40.0)
    T_shear = np.array(Affine.shear(x, y)).reshape(3, 3)
    T_shear_inv = np.linalg.inv(T_shear)
    img_transformed = np.asarray(img.transform((img.size[0], img.size[1]), Image.AFFINE, data=T_shear_inv.flatten()[:6], resample=Image.BICUBIC))
    plt.imshow(np.asarray(img_transformed), cmap='gray')
    plt.show()  

    img = Image.fromarray(img_transformed)
    img_transformed = np.asarray(img.transform((img.size[0], img.size[1]), Image.AFFINE, data=T_shear.flatten()[:6], resample=Image.BICUBIC))
    plt.imshow(np.asarray(img_transformed), cmap='gray')
    plt.show()  

def Train():
    X = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_imagesc.txt", dtype=np.dtype("(188000,)u1"))
    y = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_imagesc.txt", dtype=np.dtype("(9,)f8"))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

    nn = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100), verbose=True, learning_rate='adaptive', early_stopping=True)

    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        nn.fit(X_train, y_train)

    pickle.dump(nn, open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\nnReplicate.p", "wb"), protocol=4)

def PreprocessData():
    X_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_all", dtype=np.dtype("(188000,)u1"))[:50000]
    Y_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_all", dtype=np.dtype("(9,)f8"))[:50000]

    scalerx = StandardScaler(copy=False)
    scalery = StandardScaler(copy=False)
    X_train = scalerx.fit_transform(X_train)
    Y_train = scalery.fit_transform(Y_train)

    dump(scalerx, 'C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\std_scalerx.bin', compress=True)
    dump(scalery, 'C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\std_scalery.bin', compress=True)

    X_train.tofile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_all_preprocessed")
    Y_train.tofile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_all_preprocessed")


def PreprocessSegmentationData():
    X_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\frames", dtype=np.dtype("(518400,)u1"))
    Y_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\boxes", dtype=np.dtype("(9,)f8"))

    scalerX = StandardScaler(copy=False)
    scalerY = StandardScaler(copy=False)
    X_train = scalerX.fit_transform(X_train)
    Y_train = scalerY.fit_transform(Y_train)

    dump(scalerX, 'C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\std_scalerx_segmentation.bin', compress=True)
    dump(scalerY, 'C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\std_scalery_segmentation.bin', compress=True)

    X_train.tofile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\frames_processed")
    Y_train.tofile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\Images\\boxes_processed")

def TrainWithKeras():
    X_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_all_preprocessed", dtype=np.dtype("(188000,)f8"))
    y_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_all_preprocessed", dtype=np.dtype("(9,)f8"))

    hidden_layers = [layers.Dense(units=100, 
                                  activation='relu', 
                                  input_shape=[X_train.shape[1]], 
                                  #kernel_regularizer = keras.regularizers.l2(0.0001),
                                  kernel_initializer=keras.initializers.glorot_uniform(), 
                                  bias_initializer=keras.initializers.glorot_uniform())]
    for i in range(16):
        hidden_layers.append(layers.Dense(units=100, 
                                          activation='relu', 
                                          #kernel_regularizer = keras.regularizers.l2(0.0001),
                                          kernel_initializer=keras.initializers.glorot_uniform(), 
                                          bias_initializer=keras.initializers.glorot_uniform()))
    hidden_layers.append(layers.Dense(units=9, use_bias=False))

    model = keras.Sequential(hidden_layers)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(verbose=1, patience=5)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    model.compile(loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                metrics=['accuracy', R_squared])

    model.fit(X_train, y_train, epochs=800, batch_size=1000, validation_split=0.1, verbose=1, shuffle=True, callbacks=[reduce_lr, early_stopping])
    model.save("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasmodel")


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

def ValidateTraining():
    #pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\jxmr\\Desktop\\ProjectIII\\wrapper\\tesseract.bat'

    nn = pickle.load( open( "C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\nn3.p", "rb" ) )

    X_test = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_imagese.txt", dtype=np.dtype("(188000,)u1"))
    y_test = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_imagese.txt", dtype=np.dtype("(3,3)f8"))

    list = []

    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images\\imagese"):
        for file in f:
            list.append(os.path.join(r, file))

    for i in range(0, int(X_test.shape[0] / 2), 2):

        y_predicted1 = nn.predict(X_test[i].reshape(1, -1))[0]
        y_predicted2 = nn.predict(X_test[i+1].reshape(1, -1))[0]

        fileName = list[i]
        print(fileName)
        img = Image.open(fileName)
        #file1 = open( "C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\output1.txt", "w" ) 
        #print(pytesseract.image_to_string(img), file=file1)
        #file1.flush()
        #file1.close()
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
        #file2 = open( "C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\output2.txt", "w" ) 
        #print(pytesseract.image_to_string(img_transformation2), file=file2)
        #file2.flush()
        #file2.close()
        plt.imshow(np.asarray(img_transformation2), cmap='gray')
        plt.show()  
        img_result2 = img_transformation2.transform((img_transformation2.size[0], img_transformation2.size[1]), Image.AFFINE, data=y_predicted2.flatten()[:6], resample=Image.BICUBIC)
        #file3 = open( "C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\output3.txt", "w" ) 
        #print(pytesseract.image_to_string(img_result2), file=file3)
        #file3.flush()
        #file3.close()

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
                    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


#def BoundingBoxes():
#    file = r'C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images\\imagesm\\m\\a\\a\\maa00d00\\50307296-7296.tif'
#    im1 = cv2.imread(file,0)
#    im = cv2.imread(file)
#    ret,thresh1 = cv2.threshold(im1,180,278,cv2.THRESH_BINARY)
#    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    for cnt in contours:
#	    x,y,w,h = cv2.boundingRect(cnt)
#	    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
#    #i=0
#    #for cnt in contours:
#	   # x,y,w,h = cv2.boundingRect(cnt)
#	   # #following if statement is to ignore the noises and save the images which are of normal size(character)
#	   # #In order to write more general code, than specifying the dimensions as 100,
#	   # # number of characters should be divided by word dimension
#	   # if w>100 and h>100:
#		  #  #save individual images
#		  #  cv2.imwrite(str(i)+".jpg",thresh1[y:y+h,x:x+w])
#		  #  i=i+1
#    cv2.namedWindow('BindingBox', cv2.WINDOW_NORMAL)
#    cv2.imwrite('C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\BindingBox3.jpg',im)
#    img = Image.open('C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\BindingBox3.jpg')    
#    img.show()

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
            X[i] = np.fromfile(self.X_src, dtype=np.dtype("(" + str(self.n_features) + ",)u1"), count=1, offset=self.n_features*ID)
            Y[i] = np.fromfile(self.Y_src, dtype=np.dtype("(" + str(self.n_outputs) + ",)u1"), count=1, offset=self.n_outputs*ID)

        return self.preprocessing(X, Y, self.scalerX, self.scalerY)

def Replicate():
    nn = pickle.load( open( "C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\nn3.p", "rb" ) )
    weights = nn.coefs_
    intercepts = nn.intercepts_
    print (len(weights))
    print(weights[0].shape)
    print(weights[1].shape)

    print (len(intercepts))
    print(intercepts[0].shape)

    hidden_layers = [layers.Dense(100, activation='relu', input_shape=[188000])]
    for i in range(16):
        hidden_layers.append(layers.Dense(100, activation='relu'))
    hidden_layers.append(layers.Dense(9))

    model = keras.Sequential(hidden_layers)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse', 'accuracy'])

    for i in range(len(model.layers)):
        print(model.layers[i].get_weights()[0].shape)
        print(model.layers[i].get_weights()[1].shape)
        model.layers[i].set_weights([weights[i], intercepts[i]])

    X_test = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_imagese.txt", dtype=np.dtype("(188000,)u1"))
    y_test = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_imagese.txt", dtype=np.dtype("(3,3)f8"))

    list = []

    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images\\imagese"):
        for file in f:
            list.append(os.path.join(r, file))

    for i in range(0, int(X_test.shape[0] / 2), 2):

        y_predicted1 = model.predict(X_test[i].reshape(1, -1))[0]
        y_predicted2 = model.predict(X_test[i+1].reshape(1, -1))[0]

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


class LRMultiplier(tf.keras.optimizers.Optimizer):

    def __init__(self,
                 optimizer,
                 multipliers,
                 **kwargs):
        """Initialize the optimizer wrapper.
        :param optimizer: The original optimizer.
        :param multipliers: A dict representing the multipliers.
                            The key is the prefix of the weight to be multiplied.
        :param kwargs: Arguments for parent class.
        """
        super(LRMultiplier, self).__init__(**kwargs)
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.multipliers = multipliers
        if hasattr(self.optimizer, 'learning_rate'):
            self.lr_attr = 'learning_rate'
        else:
            self.lr_attr = 'lr'

    @property
    def lr(self):
        return self.optimizer.lr

    @lr.setter
    def lr(self, lr):
        self.optimizer.lr = lr

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        try:
            self.optimizer.learning_rate = learning_rate
        except ValueError:
            self.optimizer._hyper['learning_rate'] = learning_rate

    def _get_multiplier(self, name):
        multiplier, prefix_len = 1.0, 0
        for key, val in self.multipliers.items():
            if name.startswith(key):
                if len(key) > prefix_len:
                    prefix_len = len(key)
                    multiplier = val
        return multiplier

    def get_updates(self, loss, params):
        if len(self.updates) > 0:
            return self.updates
        multiplies = {}
        for param in params:
            multiplier = self._get_multiplier(param.name)
            if multiplier not in multiplies:
                multiplies[multiplier] = []
            multiplies[multiplier].append(param)

        self.updates, self.weights = [], []
        origin_lr = getattr(self, self.lr_attr)
        for i, (multiplier, params) in enumerate(multiplies.items()):
            lr = origin_lr
            if callable(multiplier):
                lr = lr * multiplier(tf.keras.backend.cast(self.optimizer.iterations, K.floatx()))
            elif multiplier != 1.0:
                lr = lr * multiplier
            setattr(self, self.lr_attr, lr)
            with tf.keras.backend.name_scope('Group_{}'.format(i)):
                self.updates += self.optimizer.get_updates(loss, params)
            print(self.multipliers, i, self.optimizer.weights)
            for w in self.optimizer.weights:
                if w not in self.weights:
                    self.weights.append(w)
        setattr(self, self.lr_attr, origin_lr)

        return self.updates

    def get_config(self):
        config = {
            'optimizer': tf.keras.optimizers.serialize(self.optimizer),
            'multipliers': self.multipliers
        }
        base_config = super(LRMultiplier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, v.name[:-2])
        
    def _resource_apply_dense(self, grad, handle):
        return

    def _apply_dense(self, grad, var):
        return

    @classmethod
    def from_config(cls, config):
        optimizer = tf.keras.optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)

if __name__ == '__main__':
    ValidateSegmentation()