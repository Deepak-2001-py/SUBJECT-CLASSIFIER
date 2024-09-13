import numpy as np
import pathlib
import os
from src.SubjectClassifier import logger
import joblib
class Predictionpipeline:
    def __init__(self,text):
        self.text=text

    def main(self):
        ensemble_model=joblib(os.path.join("model","Text_Ensemble.pkl"))
        ngram_tidf_pipeline=joblib(os.path.join("model","Text_Ensemble.pkl"))
        coun_vect=joblib(os.path.join("model","Text_Ensemble.pkl"))
        text=coun_vect.tranform([self.text])
        text_tfidf=ngram_tidf_pipeline.transform(text)
        text_tfidf=text_tfidf.toarray()
        prediction=ensemble_model.predict(text_tfidf)
        return prediction[0]
        
print("*************************")
STAGE_NAME="Prediction"
if __name__=="__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj=Predictionpipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
        
    except Exception as e:
        logger.exception(e) 
    

