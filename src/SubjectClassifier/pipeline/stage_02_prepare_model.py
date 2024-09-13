
from src.SubjectClassifier.config.configuration import ConfigurationManager
from src.SubjectClassifier.components.preparefinetunemodel import Preparefinetunemodel
from src.SubjectClassifier import logger



class Preparefintunemodelpipepline:
    def __init__(self):
        pass

    def main(self):
        
        config=ConfigurationManager()
        data_fintune_model_config=config.get_fintune_model_config()
        prepare_base_model=Preparefinetunemodel(config=data_fintune_model_config)
        prepare_base_model.get_finetune_model()
    
print("*************************")
STAGE_NAME="Prepare Base Model"
if __name__=="__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj=Preparefintunemodelpipepline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
        
    except Exception as e:
        logger.exception(e) 