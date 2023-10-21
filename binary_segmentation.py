import tensorflow as tf
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import glob
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import build_unet

image_names = glob.glob("./data/images/*.png")
mask_names = glob.glob("./data/masks/*.png")

def display_image(cv2_img,cv2_mask):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2_img, cmap='gray')
    plt.subplot(122)
    plt.imshow(cv2_mask, cmap='gray')
    plt.show()

def resize_images(image_names):
    resized_images = []
    for image in image_names:
        resized = cv2.resize(image,(256,256))
        resized_images.append(resized)
    return resized_images

image_names.sort()
mask_names.sort()
subset_length = 100
image_names = image_names[:subset_length]
mask_names = mask_names[:subset_length]

images = [cv2.imread(img, 0) for img in image_names]
masks = [cv2.imread(img,0) for img in mask_names]

images = resize_images(images)
masks = resize_images(masks)

image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis = 3)
mask_dataset = np.array(masks)
mask_dataset = np.expand_dims(mask_dataset,axis=3)

#Normalize images
image_dataset = image_dataset /255.  #Can also normalize or scale using MinMax scaler
#Do not normalize masks, just rescale to 0 to 1.
mask_dataset = mask_dataset /255.  #PIxel values will be 0 or 1

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 42)

# random_index = random.randint(0,image_dataset.shape[0])
# random_image = images[random_index]
# random_mask = masks[random_index]
# display_image(random_image,random_mask)

img_height = image_dataset[0].shape[0]
img_width = image_dataset[0].shape[1]
img_channels = image_dataset[0].shape[2]

input_shape = (img_height,img_width,img_channels)

model = build_unet.build_unet(input_shape=input_shape,num_classes=1)
model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=1e-3),metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=25, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

