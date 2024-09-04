from pathlib import Path
# import mlflow
# import mlflow.keras
from src.SubjectClassifier import logger
import numpy as np
import joblib
# from urllib.parse import urlparse
from src.SubjectClassifier.entity.config_entity  import (EvaluationConfig)
from src.SubjectClassifier.utils.common import save_json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import os
import pandas as pd
import time
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
class Evaluation:
    def __init__(self,config:EvaluationConfig):
        self.config=config
        self.best_model = None
        self.grid_search = None


    # def evaluation(self):
    #     # Define models
    #     ec=joblib.load(self.config.trained_model_path)
    #     x_train_tfidf=pd.read_csv(os.path.join(self.config.train_data,"X_traintfidf_data.csv"))
    #     x_test_tfidf=pd.read_csv(os.path.join(self.config.eval_data,"X_testtfidf_data.csv"))
    #     y_train=pd.read_csv(os.path.join(self.config.train_data,"y_train_data.csv"))
    #     y_test=pd.read_csv(os.path.join(self.config.eval_data,"y_test_data.csv"))
    #     logger.info("loaded all the train ,test file sucessfully")
    #     # Make predictions
    #     y_pred = ec.predict(x_test_tfidf)
    #     y_test=pd.read_csv(os.path.join(self.config.eval_data,"y_test_data.csv"))
        
    #     # Calculate scores
    #     self.accuracy = accuracy_score(self.config.y_test, y_pred)
    #     self.classification_report = classification_report(y_test, y_pred)
       
    #     # Cross-validation
    #     self.scores = cross_val_score(ec,x_train_tfidf,y_train, cv=10)

    #     # Print results
    #     print(f"Accuracy: {self.accuracy}")
    #     print("Classification Report:\n", self.classification_report)
    #     print("Cross-validated scores:", self.scores)

    # def save_score(self):
    #     scores = {
    #         "accuracy": self.accuracy,
    #         "classification_report": self.classification_report,
    #         "cross_val_score_mean": self.scores.mean(),
    #         "cross_val_score_std": self.scores.std()
    #     }
    #     if self.grid_search:
    #         scores["best_params"] = self.grid_search.best_params_
    #         scores["best_score"] = self.grid_search.best_score_
        
    #     path=Path("scores.json")
    #     save_json(path, data=scores)
    #     logger.info(f"saved scaores at {path}")

    def evaluation(self):
        # Load the trained model
        ec = joblib.load(self.config.trained_model_path)

        # # Load the data
        # x_train_tfidf = pd.read_csv(os.path.join(self.config.train_data, "X_traintfidf_data.csv"),index_col=0, header=0)
        # x_test_tfidf = pd.read_csv(os.path.join(self.config.eval_data, "X_testtfidf_data.csv"), index_col=0,header=0)
        # y_train = pd.read_csv(os.path.join(self.config.train_data, "y_train_data.csv"),index_col=0,header=0)
        # y_test = pd.read_csv(os.path.join(self.config.eval_data, "y_test_data.csv"),index_col=0)
        
        # #reset the index of data frames
        # x_train_tfidf = x_train_tfidf.reset_index(drop=True)
        # x_test_tfidf = x_test_tfidf.reset_index(drop=True)
        # y_train = y_train.reset_index(drop=True)
        # y_test = np.asanyarray(y_test.reset_index(drop=True))
        
        # Load the data
        x_train_tfidf = pd.read_csv(os.path.join(self.config.train_data, "X_traintfidf_data.csv"), index_col=0, header=0)
        x_test_tfidf = pd.read_csv(os.path.join(self.config.eval_data, "X_testtfidf_data.csv"), index_col=0, header=0)
        y_train = pd.read_csv(os.path.join(self.config.train_data, "y_train_data.csv"), index_col=0, header=0)
        y_test = pd.read_csv(os.path.join(self.config.eval_data, "y_test_data.csv"), index_col=0, header=0)

        # Reset the index of data frames
        x_train_tfidf = x_train_tfidf.reset_index(drop=True)
        x_test_tfidf = x_test_tfidf.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Convert y_train and y_test to 1D arrays
        y_train = y_train.values.ravel()  # or np.squeeze(y_train.values)
        y_test = y_test.values.ravel()  # or np.squeeze(y_test.values)
        

        
        logger.info("Loaded all the train and test files successfully")
        x=time.time()
        # Make predictions
        
        y_pred = ec.predict(x_test_tfidf)
        # print(y_pred)
        # print(y_test)
        # print(type(y_pred),type(y_test))
        y=time.time()
        logger.info(f"time taken for predection:{y-x}")
        
        # Calculate scores
        logger.info(f"calculating matrics ")
        self.accuracy = accuracy_score(y_test, y_pred)
        # self.classification_report = classification_report(y_test, y_pred)
        # self.confusion_matrix = confusion_matrix(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        self.f1_score = f1_score(y_test, y_pred, average='weighted')
        self.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation
        self.scores = cross_val_score(ec, x_train_tfidf, y_train, cv=5)
        z=time.time()
        logger.info(f"time taken for cross evalutaion:{z-y}")
        
        # Print results
        logger.info(f"Accuracy: {self.accuracy}\n")
        # logger.info("Classification Report:\n", self.classification_report)
        # logger.info(f"Confusion Matrix:\n{self.confusion_matrix}\n")
        logger.info(f"Precision: {self.precision}\n")
        logger.info(f"F1 Score: {self.f1_score}\n")
        logger.info(f"Recall: {self.recall}\n")
        logger.info(f"Cross-validated scores: {self.scores}")

    def save_score(self):
        scores = {
            "accuracy": self.accuracy,
            # "classification_report": self.classification_report,
            # "confusion_matrix": self.confusion_matrix.tolist(),
            "precision": self.precision,
            "f1_score": self.f1_score,
            "recall": self.recall,
            "cross_val_score_mean": self.scores.mean(),
            "cross_val_score_std": self.scores.std()
        }
        if self.grid_search:
            scores["best_params"] = self.grid_search.best_params_
            scores["best_score"] = self.grid_search.best_score_
        
        path = Path("scores.json")
        save_json(path, data=scores)
        logger.info(f"Saved scores at {path}")

    # def log_into_mlflow(self):
    #     mlflow.set_registry_uri(self.config.mlflow_uri)
    #     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    #     with mlflow.start_run():
    #         mlflow.log_params(self.config.all_params)
    #         mlflow.log_metrics({
    #             "accuracy": self.accuracy,
    #             "cross_val_score_mean": self.scores.mean(),
    #             "cross_val_score_std": self.scores.std()
    #         })

    #         if tracking_url_type_store != "file":
    #             mlflow.sklearn.log_model(self.best_model, "model", registered_model_name="EnsembleModel")
    #         else:
    #             mlflow.sklearn.log_model(self.best_model, "model")

    # def fine_tune(self):
    #     # Define parameter grid
    #     param_grid = {
    #         'voting': ,
    #         'weights': ,
    #         'Random Forest__n_estimators': ,
    #         'Random Forest__max_depth': ,
    #         'Logistic Regression__C': ,
    #     }

    #     # Define models
    #     mnb = MultinomialNB()
    #     rfc = RandomForestClassifier(random_state=42)
    #     lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    #     svc = SVC(probability=True)

    #     # Create a VotingClassifier
    #     ec = VotingClassifier(estimators=[
    #         ('Multinominal NB', mnb), 
    #         ('Random Forest', rfc),
    #         ('Logistic Regression', lr),
    #         ('Support Vector Machine', svc)]
    #     )

    #     # Initialize GridSearchCV
    #     self.grid_search = GridSearchCV(estimator=ec, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

    #     # Fit the grid search to the data
    #     self.grid_search.fit(self.config.x_train_tfidf, self.config.y_train)

    #     # Get the best model
    #     self.best_model = self.grid_search.best_estimator_

    #     # Print the best parameters and score
    #     print(f"Best Parameters: {self.grid_search.best_params_}")
    #     print(f"Best Cross-validation Score: {self.grid_search.best_score_}")

    #     # Evaluate the best model on the test set
    #     y_pred = self.best_model.predict(self.config.x_test_tfidf)
    #     self.accuracy = accuracy_score(self.config.y_test, y_pred)
    #     self.classification_report = classification_report(self.config.y_test, y_pred)

    #     print(f"Test Accuracy: {self.accuracy}")
    #     print("Test Classification Report:\n", self.classification_report)

        # Save the scores including hyperparameter tuning results
        # self.save_score()

# Usage example:
# config = YourConfigClass()
# evaluator = ModelEvaluation(config)
# evaluator.fine_tune()
# evaluator.evaluation()  # Evaluate with the best model found by GridSearchCV
# evaluator.save_score()  # Save the scores including hyperparameter results
# evaluator.log_into_mlflow()
   
    
    # def _valid_generator(self):
    #     datagenerator_kwargs=dict(
    #         rescale=1./255,
    #         validation_split=0.30
    #                                 )
    #     dataflow_kwargs=dict(target_size=self.config.params_image_size[:-1],
    #                             batch_size=self.config.params_batch_size,
    #                             interpolation="bilinear")
    #     valid_datagenerator=tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
    #     self.valid_generator=valid_datagenerator.flow_from_directory(directory =self.config.training_data,
    #                                                                     subset="validation",
    #                                                                     shuffle=False, **dataflow_kwargs)
    # @staticmethod
    # def load_model(path: Path)-> tf.keras.Model:
    #     return tf.keras.models.load_model(path) 
    # def evaluation(self):
    #     self.model=self.load_model(self.config.path_of_model)
    #     self._valid_generator() 
    #     self.score= self.model.evaluate(self.valid_generator)  
    
    # def save_score(self):
    #     scores={"loss":self.score[0],"accuracy":self.score[1]}   
    #     save_json( path= Path("scores.json"),data=scores)   
        
    # def log_into_mlflow(self):
    #     mlflow.set_registery_uri(self.config.mlflow_uri)
    #     tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
    #     with mlflow.start_run():
    #         mlflow.log_params(self.config.all_params)
    #         mlflow.log_metrics(
    #             {"loss":self.score[0],"accuracy":self.score[1]}
    #         )
            
    #         if tracking_url_type_store !="file":
                
    #             mlflow.keras.log_model(self.model,"model",registered_model_name="VGG16Model")
    #         else :
    #             mlflow.keras.log_model(self.model,"model")
            