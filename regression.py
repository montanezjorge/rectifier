from affine import Affine
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPRegressor
from datetime import datetime
from io import StringIO
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
import sys
import pickle
import warnings

def GenerateData():
    systemRandom = random.SystemRandom()

    iter = 0
    dir_num = 2

    for dir in os.listdir("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images"):
        transforms=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_" + dir + ".txt", "ab")
        images=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_" + dir + ".txt", "ab")
        names=open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\imageNames_" + dir + ".txt", "a")

        if iter != dir_num:
            iter = iter + 1
            continue

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

        break

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    nn = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100), verbose=True, learning_rate='adaptive', early_stopping=True)

    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        nn.fit(X_train, y_train)

    pickle.dump(nn, open("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\nn2.p", "wb"), protocol=4)

    for i in range(X_test.shape[0]):
        y_predicted = nn.predict(X_test[i].reshape(1, -1))[0]
        img = Image.fromarray(X_test[i].reshape(500, 376))
        img_transformed = np.asarray(img.transform((img.size[0], img.size[1]), Image.AFFINE, data=y_predicted[:6], resample=Image.BICUBIC))
    
        plt.imshow(np.asarray(img_transformed), cmap='gray')
        plt.show()  

def ValidateTraining():
    nn = pickle.load( open( "C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\nn.p", "rb" ) )

    X_test = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transformedImages_imagesb.txt", dtype=np.dtype("(47250,)u1"))
    y_test = np.fromfile("C:\\Users\\jxmr\\Desktop\\ProjectIII\\Data2\\transforms_imagesb.txt", dtype=np.dtype("(3,3)f8"))

    list = []

    for r, d, f in os.walk("C:\\Users\\jxmr\\Desktop\\ProjectIII\\rvl-cdip\\images\\imagesb"):
        for file in f:
            list.append(os.path.join(r, file))

    for i in range(0, int(X_test.shape[0] / 2), 2):

        y_predicted1 = nn.predict(X_test[i].reshape(1, -1))[0]
        y_predicted2 = nn.predict(X_test[i+1].reshape(1, -1))[0]

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

if __name__ == '__main__':
    ValidateTraining()


