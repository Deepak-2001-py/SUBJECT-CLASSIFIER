from src.SubjectClassifier.entity.config_entity import (TrainingConfig)
from src.SubjectClassifier import logger
from pathlib import Path
import joblib
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

class Training:
    def __init__(self,config: TrainingConfig):
        self.config=config
        
    
    # def get_base_model(self):
    #     self.model=tf.keras.models.load_model(
    #         self.config.updated_base_model_path
    #         )
        
        
    # def train_valid_generator(self):
        
    #     datagenerator_kwargs=dict(rescale=1./255,
    #                             validation_split=0.20)
    #     dataflow_kwargs=dict(
    #         target_size=self.config.params_image_size[:-1],
    #         interpolation="bilinear"
    #     )
            
    #     valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    #         **datagenerator_kwargs
    #     )

    #     self.valid_generator = valid_datagenerator.flow_from_directory(
    #         directory=self.config.training_data,
    #         subset="validation",
    #         shuffle=False,
    #         **dataflow_kwargs
    #     )
        
    #     if self.config.params_is_augmentation:
    #         train_datagenerator=tf.keras.preprocessing.image.ImageDataGenerator(
    #         rotation_range=40,
    #         horizontal_flip=True,
    #         width_shift_range=0.2,
    #         height_shift_range=0.2,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         **datagenerator_kwargs
    #         )
            
            
    #     else:
    #         train_datagenerator= valid_datagenerator
            
    #     self.train_generator=train_datagenerator.flow_from_directory(
    #         directory=self.config.training_data,
    #         subset=None,
    #         **dataflow_kwargs
    #     )
        
    # @staticmethod
    # def save_model(path: Path,model: tf.keras.models):
    #     model.save(path)
        
        
    # def train_dp(self):
    #     self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
    #     self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
    #     self.model.fit(
    #         self.train_generator,
    #         epochs=self.config.params_epochs,
    #         steps_per_epoch=self.steps_per_epoch,
    #                     validation_steps=self.validation_steps,
    #         validation_data=self.valid_generator
    #     )

        # self.save_model(
        #     path=self.config.trained_model_path,
        #     model=self.model
        # )



    
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
        
        #pipline creation
        pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=NGRAM_RANGE)),
    ('tfidf', TfidfTransformer(norm=NORM,sublinear_tf=SUBLINER_TF))
        ])
        
        #transform data
        x_train_tfidf = pipeline.fit_transform(x_train)
        x_test_tfidf = pipeline.transform(x_test)
        

            
            
        #eval data
        pd.DataFrame(x_test_tfidf.toarray()).to_csv(os.path.join(self.config.eval_data, "X_testtfidf_data.csv"))
        pd.DataFrame(y_test).to_csv(os.path.join(self.config.eval_data, "y_test_data.csv"))  
        
        #train data    
        pd.DataFrame(x_train_tfidf.toarray()).to_csv(os.path.join(self.config.train_data, "X_traintfidf_data.csv"))
        pd.DataFrame(y_train).to_csv(os.path.join(self.config.train_data, "y_train_data.csv"))
        

        logger.info("data is transform  and saved sucessfully")
        
        #save pipeline
        path=os.path.join(self.config.n_gram_tfidf_model)     
        joblib.dump(pipeline, path)
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
        svc = SVC(probability=probability)
        ec=VotingClassifier(estimators=[('Multinominal NB', mnb), ('Random Forest', rfc),('Logistic Regression',lr),('Support Vector Machine',svc)], voting=voting, weights=weights)
        
        y_train=y_train.values.ravel()
        #train
        ec.fit(x_train_tfidf,y_train)       
        logger.info("traing  completed  successfully")       
        #save th ensemble model
        path=os.path.join(self.config.root_dir,"ensemble_model.pkl")
        joblib.dump(ec, path)
        logger.info(f"ensemble model saved at {path}")