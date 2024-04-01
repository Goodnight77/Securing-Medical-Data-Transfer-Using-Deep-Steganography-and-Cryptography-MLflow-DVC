import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier.utils.common import custom_loss_1,custom_loss_2,gray_to_rgb,normalize_batch,denormalize_batch
import os
from PIL import Image
import cv2
import tensorflow  as tf

class HidePredictionPipeline:
    def __init__(self,filename1,filename2):
        self.filename1 = filename1
        self.filename2 = filename2
    def predict(self):
        # load model
        hide= tf.keras.models.load_model(
                    "artifacts/split_model/hiding.h5",custom_objects={'custom_loss_2': custom_loss_2,'custom_loss_1': custom_loss_1}
                    , compile=False
                )

        source=np.array(Image.open(self.filename1))
        # Check if the images are grayscale, and convert to RGB if necessary
        cover=np.array(Image.open(self.filename2))

        source=cv2.resize(source, (224,224))
        cover=cv2.resize(cover, (224,224))
        if source.shape[-1] != 3:
            source = gray_to_rgb(source)
        # Plot input images
        if cover.shape[-1] != 3:
            source = gray_to_rgb(source)
        secret=np.array(source/255.0)
        cover=np.array(cover/255.0)
        coverout=hide.predict([normalize_batch(np.reshape(secret,(1,224,224,3))),normalize_batch(np.reshape(cover,(1,224,224,3)))])
        coverout = denormalize_batch(coverout)
        coverout=np.squeeze(coverout)*255.0
        coverout=np.uint8(coverout)
        return coverout