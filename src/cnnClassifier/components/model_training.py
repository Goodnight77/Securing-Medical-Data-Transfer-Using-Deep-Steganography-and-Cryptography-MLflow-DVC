from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
import numpy as np
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from src.cnnClassifier.utils.common import custom_loss_1,custom_loss_2
import cv2
import time
import glob

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.TRAIN_NUM=len(glob.glob(str(config.Med_train)+"/*/*"))
        self.VAL_NUM=len(glob.glob(str(config.Med_val)+"/*/*"))
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.base_model_path,custom_objects={'custom_loss_2': custom_loss_2,'custom_loss_1': custom_loss_1}
        )
    
    # Function to convert grayscale image to RGB
    @staticmethod
    def gray_to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Preprocessing functions
    @staticmethod
    def normalize_batch(imgs):
        '''Performs channel-wise z-score normalization'''

        return (imgs -  np.array([0.485, 0.456, 0.406])) /np.array([0.229, 0.224, 0.225])
    @staticmethod
    def denormalize_batch(imgs,should_clip=True):
        '''Denormalize the images for prediction'''

        imgs= (imgs * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])

        if should_clip:
            imgs= np.clip(imgs,0,1)
        return imgs

    # Custom generator for loading training images from directory
    def generate_generator_multiple(self,generator, med_path, cover_path):
        genX1 = generator.flow_from_directory(med_path, target_size=(224, 224), batch_size=self.config.params_batch_size, shuffle=True, class_mode=None)
        genX2 = generator.flow_from_directory(cover_path, target_size=(224, 224), batch_size=self.config.params_batch_size, shuffle=True, class_mode=None)

        while True:
            X1i = self.normalize_batch(genX1.next())
            X2i = self.normalize_batch(genX2.next())

            # Check if the images are grayscale, and convert to RGB if necessary
            if X1i.shape[-1] != 3:
                X1i = self.gray_to_rgb(X1i)
            if X2i.shape[-1] != 3:
                X2i = self.gray_to_rgb(X2i)

            yield ({'secret': X1i, 'cover': X2i}, {'hide_conv_f': X2i, 'revl_conv_f': X1i})

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.generate_generator_multiple(generator=valid_datagenerator, med_path=self.config.Med_val,cover_path=self.config.Cover_val)
        #needs to be two just like stego requirement
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = self.generate_generator_multiple(generator=train_datagenerator, med_path=self.config.Med_train,cover_path=self.config.Cover_train)
        self.validation_generator = self.generate_generator_multiple(generator=valid_datagenerator, med_path=self.config.Med_val,cover_path=self.config.Cover_val)
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def train(self):
        self.steps_per_epoch = self.TRAIN_NUM// self.config.params_batch_size
        self.validation_steps = self.VAL_NUM // self.config.params_batch_size

        self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
    