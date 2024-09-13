from src.SubjectClassifier.config.configuration import ConfigurationManager
from src.SubjectClassifier.components.Trainng import Training
from src.SubjectClassifier import logger
# from src.SubjectClassifier.components.Cuml_Training import Training

class Trainingpipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.prepare_data()
        training.train_ensemble_model()
        # training.train_valid_generator()
        # training.train()
    
print("*************************")
STAGE_NAME="Traning Base Model"
if __name__=="__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj=Trainingpipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
        
    except Exception as e:
        logger.exception(e) 