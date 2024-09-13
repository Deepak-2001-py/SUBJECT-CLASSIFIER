from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    subtopics_dict: Path
    local_data_file: Path
    sentences_dict: Path
    sample_num: int


# @dataclass(frozen=True)
# class PrepareBaseModelConfig:
#     root_dir: Path
#     base_model_path: Path
#     updated_base_model_path: Path
#     params_image_size: list
#     params_learning_rate: float
#     params_include_top: bool
#     params_weights: str
#     params_classes: int



# @dataclass(frozen=True)
# class PrepareCallbacksConfig:
#     root_dir: Path
#     tensorboard_root_log_dir: Path
#     checkpoint_model_filepath: Path


@dataclass(frozen=True)
class PrepareFinetuneConfig:
    root_dir: Path
    fintune_model_path: Path
    model_repo: str
    model_filename: str
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    train_data: Path
    train_data_path: Path
    eval_data: Path
    n_gram_tfidf_model: Path
    NGRAM_RANGE :tuple
    NORM: int
    SUBLINER_TF:bool
    test_size: float
    random_state:int
    n_estimators : int
    max_depth : int
    random_state : int
    C : int
    max_iter : int
    n_jobs : int
    voting : str
    weights : list
    probability: bool



@dataclass(frozen=True)
class EvaluationConfig:
    trained_model_path: Path
    train_data: Path
    eval_data: Path
    all_params: dict
    mlflow_uri: str
    # params_image_size: list
    # params_batch_size: int
