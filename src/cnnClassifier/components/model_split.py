import sys
from keras.models import Model
import tensorflow as tf
from src.cnnClassifier.utils.common import custom_loss_1,custom_loss_2
from cnnClassifier.entity.config_entity import SplitConfig



class SplitModel:
    def __init__(self, config: SplitConfig):
        self.config = config
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.trained_model_path,custom_objects={'custom_loss_2': custom_loss_2,'custom_loss_1': custom_loss_1}
            , compile=False
        )
    def get_hiding_model(self):
        # Generate hiding network
        encoder=Model([self.model.get_layer('secret').input,self.model.get_layer('cover').input],self.model.get_layer('hide_conv_f').output)
        encoder.save(self.config.hiding_model_path)

    def get_reveal_model(self):
        # Generate reveal network
        decoder=Model(self.model.get_layer('revl_conv3x3_1').input,self.model.get_layer('revl_conv_f').output)
        decoder.save(self.config.reveal_model_path)
 