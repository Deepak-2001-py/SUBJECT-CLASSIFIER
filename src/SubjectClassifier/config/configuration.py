from src.SubjectClassifier.constants import *
import os
from pathlib import Path
from src.SubjectClassifier.utils.common import read_yaml, create_directories
from src.SubjectClassifier.entity.config_entity import (DataIngestionConfig,
                                                        PrepareFinetuneConfig,
                                                # PrepareBaseModelConfig,
                                                # PrepareCallbacksConfig,
                                                TrainingConfig,
                                                EvaluationConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion 

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            subtopics_dict=Path(config.subtopics_dict),
            local_data_file=Path(config.local_data_file),
            sentences_dict=Path(config.sentences_dict ),
            sample_num=config.sample_num
            
        )

        return data_ingestion_config
    

    


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params
        # training_data = os.path.join(self.config.data_ingestion.local_data_file)
        create_directories([
            Path(training.root_dir)
        ])
        create_directories([
            Path(training.train_data)
        ])
        create_directories([
            Path(training.eval_data)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            n_gram_tfidf_model=Path(training.n_gram_tfidf_model),
            train_data=Path(training.train_data),
            eval_data=Path(training.eval_data),
            train_data_path=Path(self.config.data_ingestion.local_data_file),
            NGRAM_RANGE =params.ngram_tfidf_params.NGRAM_RANGE,
            NORM= params.ngram_tfidf_params.NORM,
            SUBLINER_TF= params.ngram_tfidf_params.SUBLINER_TF,
            test_size=params.ngram_tfidf_params.test_size,
            random_state=params.ngram_tfidf_params.random_state,
            n_estimators = params.Random_Forest.n_estimators,
            max_depth = params.Random_Forest.max_depth,
            C = params.Logistic_Regression.C,
            max_iter = params.Logistic_Regression.max_iter,
            n_jobs = params.Logistic_Regression.n_jobs,
            voting = params.VotingClassifier.voting,
            weights   = params.VotingClassifier.weights,
            probability=params.svc.probability
        )

        return training_config

    def get_fintune_model_config(self) -> PrepareFinetuneConfig:
        create_directories([Path(self.config.prepare_finetune_model.root_dir)])
        model_config = PrepareFinetuneConfig(
            fintune_model_path=Path(self.config.prepare_finetune_model.fintune_model_path),
            root_dir=Path(self.config.prepare_finetune_model.root_dir),
            model_repo=self.config.prepare_finetune_model.model_repo,
            model_filename=self.config.prepare_finetune_model.model_filename,
        )
        return model_config 



    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            trained_model_path=Path(self.config.training.trained_model_path),
            train_data=Path(self.config.training.train_data),
            eval_data=Path(self.config.training.eval_data),
            mlflow_uri="https://dagshub.com/entbappy/MLflow-DVC-Chicken-Disease-Classification.mlflow",
            all_params=self.params,
        )
        return eval_config

      