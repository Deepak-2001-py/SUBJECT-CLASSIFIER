from src.SubjectClassifier import logger
from src.SubjectClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from src.SubjectClassifier.pipeline.stage_02_prepare_base_model import Preparebasedmodel
from src.SubjectClassifier.pipeline.stage_03_training import Trainingpipeline
from src.SubjectClassifier.pipeline.stage_04_evaluation import Evaluationpipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




# STAGE_NAME = "Prepare base model"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    prepare_base_model = Preparebasedmodel()
#    prepare_base_model.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e




STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = Trainingpipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e






STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = Evaluationpipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e




