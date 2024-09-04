import numpy as np
import joblib
import os
import pandas as pd
class Predictionpipeline:
    def __init__(self,text):
        self.text=text

    def predict(self):
        ensemble_model=joblib.load("artifacts/training/ensemble_model.pkl")
        tranformer=joblib.load(os.path .join("artifacts/training/n_gram_tfidf_model.pkl"))
        # Transform the sentence into n-gram features
        test = tranformer.transform([self.text])

        # Convert the features to a DataFrame
        test = pd.DataFrame(test.toarray(), columns=range(test.shape[1])).reset_index(drop=True)

        # Make a prediction using the ensemble model
        prediction = ensemble_model.predict(test)
    
        return prediction[0]

    

