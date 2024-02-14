from cnnClassifier.entity.config_entity import EvaluationConfig
from pathlib import Path
import tensorflow as tf
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.utils.common import custom_loss_1,custom_loss_2,gray_to_rgb,normalize_batch,denormalize_batch,save_json
import glob

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.TEST_NUM=len(glob.glob(str(config.Med_test)+"/*/*"))

    def generate_generator_multiple(self,generator, med_path, cover_path):
            genX1 = generator.flow_from_directory(med_path, target_size=(224, 224), batch_size=self.config.params_batch_size, shuffle=True, class_mode=None)
            genX2 = generator.flow_from_directory(cover_path, target_size=(224, 224), batch_size=self.config.params_batch_size, shuffle=True, class_mode=None)

            while True:
                X1i = normalize_batch(genX1.next())
                X2i = normalize_batch(genX2.next())

                # Check if the images are grayscale, and convert to RGB if necessary
                if X1i.shape[-1] != 3:
                    X1i = gray_to_rgb(X1i)
                if X2i.shape[-1] != 3:
                    X2i = gray_to_rgb(X2i)

                yield ({'secret': X1i, 'cover': X2i}, {'hide_conv_f': X2i, 'revl_conv_f': X1i})

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = self.generate_generator_multiple(generator=valid_datagenerator, med_path=self.config.Med_test,cover_path=self.config.Cover_test)
    

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path,custom_objects={'custom_loss_2': custom_loss_2,'custom_loss_1': custom_loss_1})
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator,steps=self.TEST_NUM/self.config.params_batch_size, verbose=1)
        self.save_score()
    def save_score(self):
        scores = {"Total loss": self.score[0],"hide_conv_f_loss":self.score[1],"revl_conv_f_loss":self.score[2]}
        save_json(path=Path("scores.json"),data=scores)
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"Total loss": self.score[0],"hide_conv_f_loss":self.score[1],"revl_conv_f_loss":self.score[2]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="DeepStego1")
            else:
                mlflow.keras.log_model(self.model, "model")