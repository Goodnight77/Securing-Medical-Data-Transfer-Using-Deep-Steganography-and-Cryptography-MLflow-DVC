from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_split import SplitModel
from cnnClassifier import logger

STAGE_NAME= "Model_Split"
class ModelSplitPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        split_config = config.get_split_config()
        model_split = SplitModel(config=split_config)
        model_split.get_base_model()
        model_split.get_hiding_model()
        model_split.get_reveal_model()
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelSplitPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e