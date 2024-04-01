import numpy as np 
import tensorflow  as tf
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier.utils.common import custom_loss_1,custom_loss_2,gray_to_rgb,normalize_batch,denormalize_batch

import os

class RevealPredictionPipeline:
    def __init__(self):
        pass
    def predict(self,filename):
        # load model
        reveal= tf.keras.models.load_model(
            "artifacts/split_model/reveal.h5",custom_objects={'custom_loss_2': custom_loss_2,'custom_loss_1': custom_loss_1}
            , compile=False
        )
        coverout=np.array(Image.open(filename))

        coverout=cv2.resize(coverout, (224,224))
        if coverout.shape[-1] != 3:
            coverout = gray_to_rgb(coverout)
        coverout=np.array(coverout/255.0)
        coverout=normalize_batch(coverout)
        secretout=reveal.predict(np.reshape(coverout,(1,224,224,3)))
        secretout=denormalize_batch(secretout)
        secretout=np.squeeze(secretout)*255.0
        secretout=np.uint8(secretout)
        return secretout
