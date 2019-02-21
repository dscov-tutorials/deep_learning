
# This is an adaptation of the original notebook. Functions have been moved to dscov_dnn_utils.py

# **0. DISCoV 2/7/19**
# You can find the [presentation link here](https://docs.google.com/presentation/d/1pJi1fPt7i7enJAClSVkPOw3pl515YN0Q44kvO8zSpdc/edit?usp=sharing).

# **1. Python imports.**
import numpy as np  # Note numpy is aliased as np
from PIL import Image
import os
import shutil
from glob import glob  # File path collection
import tensorflow as tf  # Note tensorflow is aliased as tf
from matplotlib import pyplot as plt  # Library for plotting images

# Keras model utilities
from keras.models import Model  # A Keras class for constructing a deep neural network model
from keras.models import Sequential  # A Keras class for connecting deep neural network layers 
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator  # A class for data loading during model training

# Keras ResNet routines
from keras.applications.resnet50 import ResNet50  # Import the ResNet deep neural network
from keras.preprocessing import image  # Routines for loading image data
from keras.applications.resnet50 import preprocess_input  # ResNet-specific routines for preprocessing images
from keras.applications.resnet50 import decode_predictions  # ResNet-specific routines for extracting predictions

# Keras layers
from keras.layers import Dense  # A fully connected neural networld layer
from keras.layers import Activation  # A class for point-wise nonlinearity layers
from keras.layers import Flatten  # Reshape a tensor into a matrix
from keras.layers import Dropout  # A regularization layer which randomly zeros neural network units during training.

# Optimizers
from keras.optimizers import Adam  # Adam optimizer https://arxiv.org/abs/1412.6980
from keras.optimizers import SGD  # Stochastic gradient descent optimizer


# get_ipython().system('pip install google_images_download')
from google_images_download import google_images_download   #importing the library


# **2. Function Definitions**

def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)

def pad_name(f, padding=3):
  """Zero pad a string"""
  if not isinstance(f, basestring):
    f = str(f)
  fl = len(f)
  ll = padding - fl
  for p in range(ll):
    f = '0' + f
  return f

def build_finetune_model(base_model, dropout, fc_layers, num_classes, freeze=True):
  """Load pretrained model, add readout layer, fix the convolutional layers."""
  if freeze:
    for layer in base_model.layers:
      layer.trainable = False

  x = base_model.output
  x = Flatten()(x)
  for fc in fc_layers:
    # New FC layer, random init
    x = Dense(fc, activation='relu')(x) 
    x = Dropout(dropout)(x)

  # New softmax layer
  predictions = Dense(num_classes, activation='softmax')(x) 
  
  finetune_model = Model(inputs=base_model.input, outputs=predictions)

  return finetune_model

def plot_training(history, plot_val=False):
  """Plot the training and validation loss + accuracy"""
  acc = history.history['acc']
  # val_acc = history.history['val_acc']
  loss = history.history['loss']
  # val_loss = history.history['val_loss']
  epochs = range(len(acc))

  f = plt.figure()
  plt.subplot(121)
  plt.plot(epochs, acc, label='Train')
  if plot_val:
    plt.plot(epochs, val_acc, label='Val')
  plt.title('Training and validation accuracy')
  plt.legend()

  plt.subplot(122)
  plt.plot(epochs, loss, label='Train')
  if plot_val:
    plt.plot(epochs, val_loss, label='Val')
  plt.title('Training and validation loss')
  plt.legend()
  plt.savefig('acc_vs_epochs.png')
  plt.show()




# All code from notebook is in main().
# If could be further cleaned up
def main():
  ROOT_DIR = "."
  IMG_DIR  = "%s/image_dataset" % ROOT_DIR
  PROC_DIR = "%s_processed" % IMG_DIR
  print("TensorFlow version: " + tf.__version__)

  make_dir(IMG_DIR)
  make_dir(PROC_DIR)
  
  # **3. Download images for your DNN.**
  height     = 224
  width      = 224
  class_list = ["cat", "dog", "bird", "turtle", "cheetah"]  # Categories of object images
  response = google_images_download.googleimagesdownload()   # Class instantiation
  arguments = {
    "keywords": ",".join(class_list),
    "limit": 10,
    "print_urls": False,
    "format": "jpg",
    "type": "photo",
    "color_type": "full-color",
    "output_directory": IMG_DIR
  }
  paths = response.download(arguments)   # Passing the arguments to the function
  # print(paths)   # Print absolute paths of the downloaded images

  # Copy files to PROC_DIR into the Keras expected format
  categories = glob(IMG_DIR + '/*')
  files = []
  for cat in categories:
    it_dir = "%s/%s" % (PROC_DIR, cat.split('/')[-1])
    make_dir(it_dir)
    print('Filling directory: %s' % it_dir)
    tfiles = glob(cat + '/*.jpg')
    for idx, f in enumerate(tfiles):
      try:
        # Filter to make sure they can load in keras
        img = image.load_img(f, target_size=(height, width))
        path = "%s%s%s.jpg" % (it_dir, os.path.sep, pad_name(idx))
        shutil.copy2(f, path)
        files += [path]
      except Exception as e:
        print '%s %s is not an image: %s' % (cat, idx, f)
  print files


  # **3. Load a pretrained ResNet dnn and process an image.*
  base_model = ResNet50(weights="imagenet")

  img = image.load_img(files[0], target_size=(height, width))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds = base_model.predict(x)
  top_3 = decode_predictions(preds, top=3)[0]
  print("Predicted: ", top_3)

  f = plt.figure()
  plt.imshow(img)
  plt.grid('off')
  plt.axis('off')
  plt.title("Ground truth: %s\nprediction: %s" % (files[0], top_3))
  plt.show()
  # plt.close(f)


  # **4. Initialize model and train with different sized images.**
  base_model = ResNet50(
    weights='imagenet', 
    include_top=False, 
    input_shape=(height, width, 3))

  # **5. Augmentations**
  batch_size = 32  # Number of images to process at once

  train_datagen =  ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
  )

  train_generator = train_datagen.flow_from_directory(
    PROC_DIR, 
    target_size=(height, width),
    batch_size=batch_size)
  
  # get_ipython().system('ls /content/image_dataset_processed/')

  # ** 6. "Finetune" a trained model for your task**
  FC_LAYERS  = [32]  # Add more layers but adding elements to this list
  dropout    = 0.5

  finetune_model = build_finetune_model(
    base_model, 
    dropout=dropout, 
    fc_layers=FC_LAYERS, 
    num_classes=len(class_list))
  print(finetune_model.summary())


  # **7. Save performance and plot results.**
  epochs = 10  # How many loops through the entire dataset
  num_train_images = len(files)
  lr = 1e-4
  adam = Adam(lr=lr)
  finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

  filepath="%s/fietuned_ResNet50_model_weights.h5" % ROOT_DIR
  checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
  callbacks_list = [checkpoint]  # Iteratively executed during training

  history = finetune_model.fit_generator(
    train_generator,
    epochs=epochs,
    workers=8,
    steps_per_epoch=num_train_images // batch_size,
    shuffle=True,
    callbacks=callbacks_list)

  plot_training(history)


if __name__ == "__main__":
    main()
