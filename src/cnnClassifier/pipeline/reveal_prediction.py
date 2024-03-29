import numpy as np 
import tensorflow  as tf
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
        coverout = image.load_img(filename, target_size = (224,224))
        if len(coverout.split()) != 3:
            coverout = gray_to_rgb(coverout)
        secretout=reveal.predict(np.reshape(coverout,(1,224,224,3)))
        secretout=denormalize_batch(secretout)
        secretout=np.squeeze(secretout)*255.0
        secretout=np.uint8(secretout)
        return secretout
