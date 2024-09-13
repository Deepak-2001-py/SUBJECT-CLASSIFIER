from src.SubjectClassifier.entity.config_entity import (TrainingConfig)
from src.SubjectClassifier import logger
from pathlib import Path
import cuml
import cudf
import cupy as cp
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import  SVC
#from cuml.svm import  SVC
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from sklearn.ensemble import  VotingClassifier
from cuml.feature_extraction.text import  TfidfTransformer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

class Training:
    def __init__(self,config: TrainingConfig):
        self.config=config
            
    def prepare_data(self):
        
        #parameter
        NGRAM_RANGE =eval(self.config.NGRAM_RANGE)
        NORM= self.config.NORM
        SUBLINER_TF= self.config.SUBLINER_TF
        test_size=self.config.test_size
        random_state=self.config.random_state
        data=pd.read_csv(self.config.train_data_path)
        
        #preprocessing
        data['sentence']=data['sentence'].fillna("") 
        
        #train test split
        x_train, x_test, y_train, y_test = train_test_split(data["sentence"],data["subject"], test_size = test_size, random_state =random_state)
        # Convert to cudf Series if they're not already
        if isinstance(x_train, pd.Series):
            x_train = cudf.Series(x_train)
        elif isinstance(x_train, np.ndarray):
            x_train = cudf.Series(x_train.flatten())

        if isinstance(x_test, pd.Series):
            x_test = cudf.Series(x_test)
        elif isinstance(x_test, np.ndarray):
            x_test = cudf.Series(x_test.flatten())

        # Create the TF-IDF pipeline
        tfidf_pipeline = Pipeline([
            ('tfidf', TfidfTransformer(norm=NORM,sublinear_tf=SUBLINER_TF,ngram_range=NGRAM_RANGE))
        ])

        # Fit and transform the training data
        x_train_tfidf = tfidf_pipeline.fit_transform(x_train)

        # Transform the test data
        x_test_tfidf = tfidf_pipeline.transform(x_test)

            
            
        #eval data
        pd.DataFrame(x_test_tfidf.toarray()).to_csv(os.path.join(self.config.eval_data, "X_testtfidf_data.csv"))
        pd.DataFrame(y_test).to_csv(os.path.join(self.config.eval_data, "y_test_data.csv"))  
        
        #train data    
        pd.DataFrame(x_train_tfidf.toarray()).to_csv(os.path.join(self.config.train_data, "X_traintfidf_data.csv"))
        pd.DataFrame(y_train).to_csv(os.path.join(self.config.train_data, "y_train_data.csv"))
        

        logger.info("data is transform  and saved sucessfully")
        
        #save pipeline
        path=os.path.join(self.config.n_gram_tfidf_model)     
        joblib.dump(tfidf_pipeline, path)
        logger.info(f"pipeline saved at {path}")
          
        
        
    def train_ensemble_model(self ):
        
        #parameter
        x_train_tfidf=pd.read_csv(os.path.join(self.config.train_data,"X_traintfidf_data.csv"),index_col=0)
        y_train=pd.read_csv(os.path.join(self.config.train_data,"y_train_data.csv"),index_col=0)
        # print(y_train)
        # print(type(y_train))
        C= self.config.C
        MAX_ITER= self.config.max_iter
        N_JOBS=self.config.n_jobs
        n_estimators =self.config.n_estimators
        max_depth =self.config.max_depth
        random_state=self.config.random_state
        probability =self.config.probability
        voting =self.config.voting
        weights = self.config.weights
        
        # model creation
        mnb = MultinomialNB()
        rfc= RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        lr = LogisticRegression(C = C, max_iter = MAX_ITER, n_jobs=N_JOBS)
        svc = SVC(probability=probability ,kernel='rbf',C=1)
        ec=VotingClassifier(estimators=[('Multinominal NB', mnb), ('Random Forest', rfc),('Logistic Regression',lr),('Support Vector Machine',svc)], voting=voting, weights=weights)
        
        y_train=y_train.values.ravel()
        #train
        ec.fit(x_train_tfidf,y_train)       
        logger.info("traing  completed  successfully")       
        #save th ensemble model
        path=os.path.join(self.config.root_dir,"ensemble_model.pkl")
        joblib.dump(ec, path)
        logger.info(f"ensemble model saved at {path}")