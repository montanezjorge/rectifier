import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
#seednumber = 8
#from numpy.random import seed
#seed(seednumber)
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
from math import pi

def GenerateData():

    systemRandom = random.SystemRandom()
    iter = 0

    for dir in os.listdir("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images"):
        #imagesX=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImagesX_2", "ab")
        #imagesY=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImagesY", "ab")
        #imagesShear=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_Vertically_Sheared", "ab")
        #shears=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\vertical_shear", "ab")            
        #transformsY=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformsY", "ab")            
        ##transformsZ=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformsZ_2", "ab")            
        ##anglesX=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\anglesX_2", "ab")   
        #anglesY=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\anglesY", "ab")  
        #anglesZ=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\anglesZ_2", "ab")  

        if iter == 4:
            break

        iter = iter + 1

        for r, d, f in os.walk(os.path.join("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images", dir)):
            for file in f:

                fileName = os.path.join(r, file)
                print(fileName)

                img = cv2.imread(fileName)
                img = cv2.copyMakeBorder(img, top=500, bottom=500, left=0, right=0, borderType=cv2.BORDER_CONSTANT)
                img = cv2.resize(img, (754, 1000), interpolation=cv2.INTER_CUBIC)

                shear = systemRandom.uniform(-0.60, 0.60)
                transform1 = np.array([[1, 0, 0],
                    [shear, 1, 0],
                    [0, 0, 1]]).astype(np.double)
                img1 = cv2.warpPerspective(img, transform1, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

                plt.subplot(1, 2, 1)
                plt.imshow(img1)
                
                plt.show()
                #theta = systemRandom.uniform(-30.0, 30.0)
                #phi = systemRandom.uniform(-45.0, 45.0)
                #gamma = systemRandom.uniform(-30.0, 30.0)
                
                #transform.tofile(transformsX)
                #transform.tofile(transformsY)
                #np.asarray(shear).tofile(shears)
                #img.tofile(imagesShear)
                #im.tofile(imagesY)
                ##im.tofile(imagesZ)
                ##np.asarray(theta).tofile(anglesX)
                #np.asarray(phi).tofile(anglesY)
                #np.asarray(gamma).tofile(anglesZ)

        #imagesShear.close()
        #shears.close()
        #imagesX.close() 
        #imagesY.close() 
        ##imagesZ.close() 
        ##transformsX.close()
        #transformsY.close()
        ##transformsZ.close()
        ##anglesX.close()
        #anglesY.close()
        #anglesZ.close()

def Validate():

    systemRandom = random.SystemRandom()
    nnZ = load_model("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasRectifierZ.h5", custom_objects={'R_squared': R_squared})

    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images\\imagese"):
        for file in f:
            fileName = os.path.join(r, file)
            print(fileName)
            
            #img = cv2.imread(fileName)
            #img = cv2.resize(img, (754, 1000), interpolation=cv2.INTER_CUBIC)

            gamma = systemRandom.uniform(-30, 30)
            print(gamma)

            transformer = ImageTransformer(fileName, shape=(754, 1000))
            img, transform = transformer.rotate_along_axis(gamma=gamma)

            #transform = np.array([[1, 0, 0],
            #    [shear, 1, 0],
            #    [0, 0, 1]]).astype(np.double)
            #img = cv2.warpPerspective(img, transform, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

            plt.imshow(img)
            plt.show()

            gamma = nnZ.predict(img.flatten('K').reshape(1, -1))[0][0]
            print(gamma)

            transform = transformer.get_transformation_matrix(gamma=gamma)

            img = cv2.warpPerspective(img, transform, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

            plt.imshow(img)
            plt.show()


def Train():

    x = img_input = Input(shape=(2262000), name="data")

    for i in range(21):
        x = layers.Dense(units=100, 
                    activation='relu', 
                    kernel_initializer=keras.initializers.glorot_uniform(), 
                    bias_initializer=keras.initializers.glorot_uniform())(x)

    x = Dense(units=1, use_bias=False)(x)

    model = Model(inputs=[img_input], outputs=[x])
    model.summary()

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_R_squared', verbose=1, patience=5, mode='max')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_R_squared', min_delta=0.00005, patience=12, verbose=1, mode='max', baseline=None, restore_best_weights=True)
    model.compile(loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                metrics=[R_squared])

    X_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_Vertically_Sheared", dtype=np.dtype("(2262000,)u1"))[:45000]
    y_train = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\vertical_shear", dtype=np.dtype("(1,)f8"))[:45000]

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_R_squared', verbose=1, patience=5, mode='max')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_R_squared', min_delta=0.00005, patience=12, verbose=1, mode='max', baseline=None, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=30, batch_size=25, validation_split=0.1, verbose=1, shuffle=True, callbacks=[reduce_lr, early_stopping])

    model.save("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasRectifierVerticalShear.h5")


def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    return tf.subtract(1.0, tf.divide(residual, total))

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
            X[i] = (np.fromfile(self.X_src, dtype=np.dtype("(" + str(self.n_features) + ",)u1"), count=1, offset=self.n_features*ID) / 255).astype(np.float32)
            Y[i] = np.fromfile(self.Y_src, dtype=np.dtype("(" + str(self.n_outputs) + ",)f8"), count=1, offset=self.n_outputs*ID*8)

        return X, Y

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

    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=5, layers='all')
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes2.h5")
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
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    #model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    nnX = load_model("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasRectifierX.h5", custom_objects={'R_squared': R_squared})
    nnY = load_model("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasRectifierY_2.h5", custom_objects={'R_squared': R_squared})
    nnZ = load_model("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasRectifierZ.h5", custom_objects={'R_squared': R_squared})
    nnHS = load_model("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasRectifierHorizontalShear.h5", custom_objects={'R_squared': R_squared})
    nnVS = load_model("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\kerasRectifierVerticalShear.h5", custom_objects={'R_squared': R_squared})


    image = cv2.imread("C:\\Users\\jxmr\\Downloads\\IMG_20191207_020224.jpg")

    plt.subplot(1, 2, 1)
    plt.imshow(image)

    result = RectifyImage(image, model, nnX, nnZ, nnHS)

    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

    image = cv2.imread("C:\\Users\\jxmr\\Downloads\\IMG_20191207_020339.jpg")

    plt.subplot(1, 2, 1)
    plt.imshow(image)

    result = RectifyImage(image, model, nnX, nnZ, nnHS)

    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()


    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\sampleDataset\\input_sample\\background00"):
        for file in f:
            if file.endswith("magazine001.avi"):
                p = os.path.join(r, file)
                print(p)
                vidcap = cv2.VideoCapture(p)
                success,image = vidcap.read()
                while success:
                    result = RectifyImage(image, model, nnX, nnZ, nnHS)
                    plt.imshow(result)
                    plt.show()
                    success,image = vidcap.read()


def RectifyImage(image, model, nnX, nnZ, nnHS):
    r = model.detect([image], verbose=1)[0]

    roi = image[int(r['rois'][0][0]*1.025) : int(r['rois'][0][2]*.975), int(r['rois'][0][1]*1.025): int(r['rois'][0][3]*.975), :]
    roi = cv2.resize(roi, (754, 1000), interpolation=cv2.INTER_CUBIC)

    print("X")
    _, image = Rotate([nnX], roi, [FuncX], image, 15, 0.90)
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (5,5),0)
    ret, roi= cv2.threshold(roi,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    roi = cv2.resize(roi, (754, 1000), interpolation=cv2.INTER_CUBIC)

    print("Z")
    roi, image = Rotate([nnZ], roi, [FuncZ], image, 3, 1.90, invert=True)

    roi = cv2.resize(image, (754, 1000), interpolation=cv2.INTER_CUBIC)

    print("ShearH")
    roi, image = Rotate([nnHS], roi, [ShearH], image, 2, 0.080, invert=False)

    return image

def FuncX(transformer, angle):
    return transformer.get_transformation_matrix(theta=angle)

def FuncY(transformer, angle):
    return transformer.get_transformation_matrix(phi=angle)

def FuncZ(transformer, angle):
    return transformer.get_transformation_matrix(gamma=angle)
def ShearH(transformer, shear):
    return np.array([[1, shear, 0],
            [0, 1, 0],
            [0, 0, 1]]).astype(np.double)
def ShearV(transformer, shear):
    return np.array([[1, 0, 0],
            [shear, 1, 0],
            [0, 0, 1]]).astype(np.double)

def Rotate(nns, roi, funcs, orig, iters, min, invert=False):

    intermediate = roi
    composition = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]).astype(np.double)

    index = 0
    modulo = len(nns)

    flag = True
    previous = 0                         
    count = 1
    while True:

        angle = nns[index % modulo].predict(intermediate.flatten('K').reshape(1, -1))[0]
        print(angle)
        if flag:
            previous = angle
            flag = False
            if math.fabs(angle) < min:
                return intermediate, orig
        if previous * angle < 0 or count > iters:
            if invert:
                orig = cv2.warpPerspective(orig, composition, (orig.shape[1], orig.shape[0]), flags=cv2.INTER_LINEAR)
            else:
                orig = cv2.warpPerspective(orig, composition, (orig.shape[1], orig.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            break

        count = count + 1
        previous = angle
        transformer = ImageTransformer("", image=intermediate, shape = None)
        transform = funcs[index % modulo](transformer, angle[0])
        index = index + 1

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                temp = composition
                composition = np.matmul(composition, transform)

                if invert:
                    intermediate = cv2.warpPerspective(roi, composition, (roi.shape[1], roi.shape[0]), flags=cv2.INTER_LINEAR)
                else: 
                    intermediate = cv2.warpPerspective(roi, composition, (roi.shape[1], roi.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            except Warning as e:
                composition = np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]]).astype(np.double)
                if invert:
                    orig = cv2.warpPerspective(orig, temp, (orig.shape[1], orig.shape[0]), flags=cv2.INTER_LINEAR)
                    intermediate = roi = cv2.warpPerspective(roi, temp, (roi.shape[1], roi.shape[0]), flags=cv2.INTER_LINEAR)
                else:
                    orig = cv2.warpPerspective(orig, composition, (orig.shape[1], orig.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                    intermediate = roi = cv2.warpPerspective(roi, temp, (roi.shape[1], roi.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                

    return intermediate, orig

def Test():
    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\OCRDataset\\Segmentation\\sampleDataset\\input_sample\\background00"):
        for file in f:
            if file.endswith("magazine001.avi"):
                p = os.path.join(r, file)
                print(p)
                vidcap = cv2.VideoCapture(p)
                success,img = vidcap.read()
                for i in range(29):
                    success,img = vidcap.read()

                for i in range(0, 10, 1):
                    plt.subplot(1,2,1)
                    plt.imshow(img)

                    transformer = ImageTransformer(p, None, image=img)
                    transformer.focal = 5
                    im, transform = transformer.rotate_along_axis(gamma=90)
                    transformer = ImageTransformer(p, None, image=im)
                    im = transformer.shear(y=-i/10)
                    plt.subplot(1,2,2)
                    plt.imshow(im)
                    plt.show()  

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]                
                    
if __name__ == '__main__':
    RunMaskRCNN()    


















#cv2.dnn_registerLayer('Crop', CropLayer)
#net = cv2.dnn.readNet('deploy.prototxt', 'hed_pretrained_bsds.caffemodel')


#inp = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(image.shape[1], image.shape[0]),
#                           mean=(104.00698793, 116.66876762, 122.67891434),
#                           swapRB=False, crop=False)
#net.setInput(inp)
#out = net.forward()
#out = out[0, 0]
#out = cv2.resize(out, (roi.shape[1], roi.shape[0]))
#out = 255 * out
#out = out.astype(np.uint8)
#out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
#plt.imshow(out)
#plt.show()