# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:17:19 2022

@author: schomsin
"""

import os
# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

import tensorflow as tf
from tensorflow.python.keras import backend as K

# adjust values to your needs
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

"""
Title: Image segmentation with a U-Net-like architecture
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/03/20
Last modified: 2020/04/20
Description: Image segmentation model trained from scratch on the Oxford Pets dataset.
"""
"""
## Download the data
"""

"""shell
curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xf images.tar.gz
tar -xf annotations.tar.gz
"""

"""
## Prepare paths of input images and target segmentation masks
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

# adjust values to your needs
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import os
import sys
import tensorflow as tf

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer

input_dir = "images/"
target_dir = "annotations/trimaps/"
xml_dir = "annotations/xmls/"
img_size = (128, 128) #(160, 160)
num_classes = 3
batch_size = 15#32 fix gpu training

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)
input_xml_paths = sorted(
    [
        os.path.join(xml_dir, fname)
        for fname in os.listdir(xml_dir)
        if fname.endswith(".xml")
    ]
)


# a='annotations/xmls/Siamese_113.xml'.replace("/"," ").split("_")[0].split(" ")[2] 
# input_name = [ a.replace("/"," ").split("_")[0].split(" ")[2] for a in input_xml_paths ]
input_name = [ a.replace("/"," ").split("_")[0].split(" ")[1] for a in input_img_paths ]

# sys.exit()

seen = set()
uniq = [x for x in input_name if x not in seen and not seen.add(x)]
# from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
transfomed_label = encoder.fit_transform(uniq)
n_uniq=len(uniq)
print(transfomed_label)


# input_name_label=encoder.transform(input_name)

# input_name_label = np.asarray(input_name_label)
# mask_label = np.zeros((input_name_label.shape[0],) + img_size + (input_name_label.shape[1],), dtype="uint8")
# for a in range(mask_label.shape[0]) :
#     for b in range(mask_label.shape[3]) :
#         mask_label[a,:,:,b]=mask_label[a,:,:,b]+input_name_label[a,b]


# sys.exit()


print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)

"""
## What does one input image and corresponding segmentation mask look like?
"""

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

# Display input image #7
display(Image(filename=input_img_paths[9]))

# Display auto-contrast version of corresponding target (per-pixel categories)
img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
display(img)

# sys.exit()




"""
## Prepare `Sequence` class to load & vectorize batches of data
"""

class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y

class OxfordPetsMod(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        r = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        g = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        b = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            # img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            # y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
            img = load_img(path, target_size=self.img_size)
            y[j] = img
            r[j] = np.expand_dims(y[j][:,:,0], 2)
            g[j] = np.expand_dims(y[j][:,:,1], 2)  
            b[j] = np.expand_dims(y[j][:,:,2], 2)  
        return x, r, g, b

class OxfordPetsMod1(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            # x[j]=x[j]/255.0
        r = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        g = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        b = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            # img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            # y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
            img = load_img(path, target_size=self.img_size)
            y[j] = img
            # y[j] = y[j]/255.0
            r[j] = np.expand_dims(y[j][:,:,0], 2)
            g[j] = np.expand_dims(y[j][:,:,1], 2)  
            b[j] = np.expand_dims(y[j][:,:,2], 2)  
        return x, r#, g, b
    
class OxfordPetsMod2(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            # x[j]=x[j]/255.0
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            # img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            # y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
            img = load_img(path, target_size=self.img_size)
            y[j] = img
            # y[j] = y[j]/255.0
        return x, y
    
class OxfordPetsMod3():# it can not run with gpu
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = len(input_img_paths)#batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def getitem(self):
        batch_input_img_paths = self.input_img_paths[:]
        batch_target_img_paths = self.target_img_paths[:]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            # x[j]=x[j]/255.0
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            # img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            # y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
            img = load_img(path, target_size=self.img_size)
            y[j] = img
            # y[j] = y[j]/255.0
        return x, y    
    
class OxfordPetsMod4():# it can not run with gpu
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = len(input_img_paths)#batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def getitem(self):
        batch_input_img_paths = self.input_img_paths[:]
        batch_target_img_paths = self.target_img_paths[:]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            # x[j]=x[j]/255.0

        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            # img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            # y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
            img = load_img(path, target_size=self.img_size)
            y[j] = img
            # y[j] = y[j]/255.0
            
        input_name = [ a.replace("/"," ").split("_")[0].split(" ")[1] for a in batch_input_img_paths ]
        seen = set()
        uniq = [x for x in input_name if x not in seen and not seen.add(x)]
        encoder = LabelBinarizer()
        transfomed_label = encoder.fit_transform(uniq)
        # print(transfomed_label)
        
        input_name_label=encoder.transform(input_name)  
        input_name_label = np.asarray(input_name_label)
        mask_label = np.zeros((input_name_label.shape[0],) + img_size + (input_name_label.shape[1],), dtype="uint8")
        for a in range(mask_label.shape[0]) :
            for b in range(mask_label.shape[3]) :
                mask_label[a,:,:,b]=mask_label[a,:,:,b]+input_name_label[a,b]
        
        return x, y, mask_label, input_name_label, input_name, uniq, encoder    

        
class OxfordPetsMod5(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, encoder, n_uniq):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.encoder = encoder
        self.n_uniq=n_uniq

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        mask_label = np.zeros((self.batch_size,) + img_size + (self.n_uniq,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            # x[j]=x[j]/255.0
            
            name=path.replace("/"," ").split("_")[0].split(" ")[1]
            name_label=self.encoder.transform([name])  
            name_label = np.asarray(name_label)
            mask= np.zeros( img_size + (self.n_uniq,), dtype="uint8")
            for b in range(mask.shape[2]) :
                mask[:,:,b]=mask[:,:,b]+name_label[0,b]
            mask_label[j]=mask
            
        # y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        # for j, path in enumerate(batch_target_img_paths):
        #     # img = load_img(path, target_size=self.img_size, color_mode="grayscale")
        #     # y[j] = np.expand_dims(img, 2)
        #     # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        #     # y[j] -= 1
        #     img = load_img(path, target_size=self.img_size)
        #     y[j] = img
        #     # y[j] = y[j]/255.0
        return [x,mask_label], x


"""
## Prepare U-Net Xception-style model
"""

from tensorflow.keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    # outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    outputs = layers.Conv2D(num_classes, 3, activation="linear", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def get_model1(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    # outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    outputs = layers.Conv2D(num_classes, 3, activation="linear", padding="same")(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model





def get_model2(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    inputs1 = keras.Input(shape=img_size + (35,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(128, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    a = layers.Conv2D(128, 3, strides=2, padding="same")(inputs1)
    a = layers.BatchNormalization()(a)
    a = layers.Activation("relu")(a)
    # a = layers.MaxPooling2D(3, strides=2, padding="same")(a)

    # a = layers.Conv2D(32, 3, strides=2, padding="same")(inputs1)
    # a = layers.BatchNormalization()(a)
    # a = layers.Activation("relu")(a)
    # x = layers.add([x, a])  # Add x a

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    
    x = layers.add([x, a])  # Add x a

    for filters in [128]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    # outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    outputs = layers.Conv2D(num_classes, 3, activation="linear", padding="same")(x)

    # Define the model
    model = keras.Model(inputs=[inputs,inputs1], outputs=outputs)
    return model


def get_model2_org(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    inputs1 = keras.Input(shape=img_size + (35,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    a = layers.Conv2D(256, 3, strides=2, padding="same")(inputs1)
    a = layers.BatchNormalization()(a)
    a = layers.Activation("relu")(a)
    a = layers.MaxPooling2D(3, strides=2, padding="same")(a)
    a = layers.MaxPooling2D(3, strides=2, padding="same")(a)
    a = layers.MaxPooling2D(3, strides=2, padding="same")(a)
    
    # a = layers.Conv2D(32, 3, strides=2, padding="same")(inputs1)
    # a = layers.BatchNormalization()(a)
    # a = layers.Activation("relu")(a)
    # x = layers.add([x, a])  # Add x a

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    
    x = layers.add([x, a])  # Add x a

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    # outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    outputs = layers.Conv2D(num_classes, 3, activation="linear", padding="same")(x)

    # Define the model
    model = keras.Model(inputs=[inputs,inputs1], outputs=outputs)
    return model



# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


"""
## Set aside a validation split
"""

import random


# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
# train_input_img_paths = input_img_paths[:-val_samples]
# train_target_img_paths = target_img_paths[:-val_samples]
# val_input_img_paths = input_img_paths[-val_samples:]
# val_target_img_paths = target_img_paths[-val_samples:]

with tf.device("CPU"):
    train_input_img_paths = input_img_paths[:val_samples]#3000 can run gpu
    train_target_img_paths = target_img_paths[:val_samples]#3000 can run gpu
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]
    
    # Instantiate data Sequences for each split
    # train_gen = OxfordPets(
    #     batch_size, img_size, train_input_img_paths, train_target_img_paths
    # )
    train_gen = OxfordPetsMod5(
        batch_size, img_size, train_input_img_paths, train_input_img_paths, encoder, n_uniq
    )
    
    
    # val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    val_gen = OxfordPetsMod5(batch_size, img_size, val_input_img_paths, val_input_img_paths, encoder, n_uniq)

    
    img1 = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(val_gen[10][0][0][-1]))
    print(val_gen[10][0][0][-1].shape)
    display(img1)
 
    print(val_gen[10][0][1][-1].shape)

    img1 = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(val_gen[10][1][-1]))
    print(val_gen[10][1][-1].shape)
    display(img1)
    
    for j in range(val_gen[10][1][-1].shape[-1]):
        img1=np.expand_dims(val_gen[10][1][-1][:,:,j], 2)
        print(img1.shape)
        img1 = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(img1))
        display(img1)
        
    # for j in range(val_mask_label.shape[-1]):
    #     img1=np.expand_dims(val_mask_label[1][:,:,j], 2)
    #     print(img1.shape)
    #     img1 = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(img1))
    #     display(img1)


# sys.exit()



if os.path.exists('oxford_gen_color_r4.h5') and os.path.isfile('oxford_gen_color_r4.h5') :
    pass
    # load
    model=keras.models.load_model('oxford_gen_color_r4.h5',compile=False)
    # model.summary()
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)
    fw = open("summary-oxford.txt", "w") 
    fw.write(str(short_model_summary))
    fw.close()
    
else :
    # Build model
    model = get_model2(img_size, num_classes)
    model.summary()
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    # print(short_model_summary)
    fw = open("summary-oxford.txt", "w") 
    fw.write(str(short_model_summary))
    fw.close()
    
    """
    ## Train the model
    """

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    # model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    adam = tf.keras.optimizers.Adam()
    model.compile(optimizer=adam, loss="mae")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint("oxford_gen_color_r4.h5", save_best_only=False)
    ]
    
    # Train the model, doing validation at the end of each epoch.
    epochs = 80#15
    model.fit(train_gen,batch_size=0 , epochs=epochs, validation_data=val_gen, callbacks=callbacks)   
    pass

"""
## Visualize predictions
"""
# sys.exit()
# Generate predictions for all images in the validation set

# val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
# val_gen = OxfordPetsMod4(batch_size, img_size, val_input_img_paths, val_input_img_paths)
# val_x, val_y, val_mask_label, val_input_name_label, val_input_name, val_uniq, val_encoder=val_gen.getitem()

img1 = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(val_gen[1][0][0][-1]))
print(val_gen[1][0][0][-1].shape)
display(img1)
img1.save('./out/oxford_gen_color_r4_0.png')
img1 = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(val_gen[1][1][-1]))
print(val_gen[1][1][-1].shape)
display(img1)
img1.save('./out/oxford_gen_color_r4_1.png')

val_preds = model.predict(val_gen[1][0])
print(val_preds.shape)

def display_mask(i):
    """Quick utility to display a model's prediction."""
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(val_gen[1][0][0][i]))
    display(img)
    img.save('./out/oxford_gen_color_r4_2.png')
    
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(val_preds[i]))
    display(img)
    img.save('./out/oxford_gen_color_r4_3.png')
    
    for j in range(val_preds[i].shape[-1]):
        img0 = np.expand_dims(val_preds[i][:,:,j], axis=-1)
        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(img0))
        display(img)
        img.save('./out/oxford_gen_color_r4_3_%d.png'%(j))



# Display results for validation image #10
i = 1

# Display input image
# display(Image(filename=val_input_img_paths[i]))

# Display ground-truth target mask
# img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
# display(img)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.


# sys.exit()

img = PIL.ImageOps.autocontrast(load_img('./input/000004.jpg', target_size=img_size))
# img = PIL.ImageOps.autocontrast(load_img('./images/Abyssinian_1.jpg', target_size=(160, 160)))
display(img)
img.save('./out/oxford_gen_color_r4_4.png')
x = np.zeros((1,)+ img_size + (3,), dtype="uint8")
x[0]=img

val_preds = model.predict([x,
                          val_gen[10][0][1][0].reshape(1,val_gen[10][0][1][-1].shape[0], val_gen[10][0][1][-1].shape[1],val_gen[10][0][1][-1].shape[2])])
# val_preds = (np.rint(val_preds)).astype(int)
print(x.shape)
print(val_preds.shape)

img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(val_preds[0]))
display(img)
img.save('./out/oxford_gen_color_r4_5.png')

