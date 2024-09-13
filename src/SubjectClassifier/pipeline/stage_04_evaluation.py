from src.SubjectClassifier.config.configuration import ConfigurationManager
from src.SubjectClassifier.components.Evaluation import Evaluation
from src.SubjectClassifier import logger
# from src.SubjectClassifier.components.Cuml_Evaluation import Evaluation

class Evaluationpipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.evaluation()
        result=evaluation.save_score()
        return result
        # evaluation.log_into_mlflow()
      
    
print("*************************")
STAGE_NAME="Model Evaluation"
if __name__=="__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj=Evaluationpipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
        
    except Exception as e:
        logger.exception(e) 