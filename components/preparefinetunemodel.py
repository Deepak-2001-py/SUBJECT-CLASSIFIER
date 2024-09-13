from src.SubjectClassifier.entity.config_entity import (PrepareFinetuneConfig)
from src.SubjectClassifier import logger
from pathlib import Path
from huggingface_hub import hf_hub_download
class Preparefinetunemodel:
    def __init__(self,config: PrepareFinetuneConfig):
        self.config=config
        
        

    def get_finetune_model(self):
        # Replace with your model repo and filename
        model_repo = self.config.model_repo
        model_filename = self.config.model_filename

        # Specify the directory where you want to save the model
        save_directory = self.config.fintune_model_path  # Replace with your desired path
        logger.info(f"Downloading model from {model_repo}...")
        # Download the model and save it to the specified directory
        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename, local_dir=save_directory)
        logger.info(f"Model saved at: {model_path}")