import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier.utils.common import custom_loss_1,custom_loss_2,gray_to_rgb,normalize_batch,denormalize_batch
import os
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
        secret = image.load_img(self.filename1, target_size = (224,224))
        cover = image.load_img(self.filename2, target_size = (224,224))
        if len(secret.split()) != 3:
            secret = gray_to_rgb(secret)
        if len(cover.split()) != 3:
            cover = gray_to_rgb(cover)
        coverout=hide.predict([normalize_batch(np.reshape(secret,(1,224,224,3))),normalize_batch(np.reshape(cover,(1,224,224,3)))])
        coverout = denormalize_batch(coverout)
        coverout=np.squeeze(coverout)*255.0
        coverout=np.uint8(coverout)
        return coverout