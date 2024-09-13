import numpy as np
import joblib
# import cuml
# import cudf
# import cupy as cp
# import 
import pandas as pd
from langchain import PromptTemplate
import os
from langchain.chains import LLMChain
import warnings
warnings.filterwarnings("ignore")

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="artifacts\prepare_finetune_model\unsloth.Q4_K_M.gguf\.huggingface\download\unsloth.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
class Predictionpipeline:
    def __init__(self,text):
        self.text=text

    def predict_using_cassical(self):
        ensemble_model=joblib.load("artifacts/demo/ensemble_model.pkl")
        # count_vect=joblib.load("model/count_vect.pkl")
        tranformer=joblib.load(os.path .join("artifacts/demo/n_gram_tfidf_model.pkl"))
        # text=count_vect.transform([self.text.lower().strip()])
        # Transform the sentence into n-gram features
        test = tranformer.transform([self.text])

        # Convert the features to a DataFrame
        # test = pd.DataFrame(test.toarray(), columns=range(test.shape[1])).reset_index(drop=True)

        # Make a prediction using the ensemble model
        prediction = ensemble_model.predict(test.toarray())
    
        return prediction[0]

    
    def predict_using_llm(self):
        alpaca_prompt = PromptTemplate(
            input_variables=["input_text"],
            template="""
        Below is a task where the model has to be fine-tuned on a specific dataset to classify subjects into predefined categories. The model is trained using this dataset.

        ### Input:
        {input_text}

        ### Response:
        """
        )
        chain = LLMChain(llm=llm, prompt=alpaca_prompt)

        # Define the input text
        input_text = self.text

        # Run the chain
        return chain.run(input_text)
        
