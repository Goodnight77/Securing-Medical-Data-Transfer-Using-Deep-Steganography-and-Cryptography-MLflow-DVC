import numpy as np 
import tensorflow  as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier.utils.common import custom_loss_1,custom_loss_2,gray_to_rgb,normalize_batch,denormalize_batch

import os

class RevealPredictionPipeline:
    def __init__(self,filename):
        self.filename = filename
    def predict(self):
        # load model
        reveal= tf.keras.models.load_model(
            "artifacts/split_model/reveal.h5",custom_objects={'custom_loss_2': custom_loss_2,'custom_loss_1': custom_loss_1}
            , compile=False
        )
        imagename = self.filename
        coverout = image.load_img(imagename, target_size = (224,224))
        if coverout.shape[-1] != 3:
            coverout = gray_to_rgb(coverout)
        secretout=reveal.predict(coverout)
        secretout=denormalize_batch(secretout)
        secretout=np.squeeze(secretout)*255.0
        secretout=np.uint8(secretout)
        return secretout
