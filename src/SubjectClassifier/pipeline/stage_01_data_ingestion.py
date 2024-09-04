
from src.SubjectClassifier.config.configuration import ConfigurationManager
from src.SubjectClassifier.components.data_ingestion import DataIngestion
from src.SubjectClassifier import logger


 


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        config= ConfigurationManager()
        data_ingestion_config= config.get_data_ingestion_config()
        data_ingestion= DataIngestion(config=data_ingestion_config)
        data_ingestion.synthetic_data()
        data_ingestion.convert_to_dataframe()
    

STAGE_NAME="Data ingestion"
if __name__=="__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj=DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
        
    except Exception as e:
        logger.exception(e) 