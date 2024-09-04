
import os 
from src.SubjectClassifier import logger
from src.SubjectClassifier.utils.common import get_size
from src.SubjectClassifier.entity.config_entity import (DataIngestionConfig)
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# libereies importing
from langchain.output_parsers import StructuredOutputParser
# from langchain.schema import StructuredOutput
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel, Field
import json
import warnings
warnings.filterwarnings("ignore")
# function for flatten and deduplicate sentences
import itertools
import time
import pandas as pd
# Function to flatten and deduplicate sentences
def flatten_and_deduplicate(sentences):
    return list(set(
    itertools.chain.from_iterable(
        [d["sentence"] if isinstance(d, dict) and "sentence" in d else d for d in sublist]
        for sublist in sentences if isinstance(sublist, list) )
))
load_dotenv()
llm=ChatGroq(model="llama3-70b-8192")
class Sentences(BaseModel):
    """Call this with an to get the list  of sentences form given topic with subtopic"""
    sentences_list: List = Field(description="list of sentences for given topic with subtopic provide list of string")
    
    
    
class DataIngestion:
    def __init__(self, config = DataIngestionConfig):
        self.config = config
        

    def flatten_and_deduplicate(self,sentences):
        return list(set(
        itertools.chain.from_iterable(
            [d["sentence"] if isinstance(d, dict) and "sentence" in d else d for d in sublist]
            for sublist in sentences if isinstance(sublist, list) )
    ))   
        
    def generate_sentences(self,topic:str,num_sentences:int,subtopic:str):
        #for topics with subtopics sentence creation
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output list of sentences of provided topic with subtopic."},

        ]


        generate_subtopics_function=convert_pydantic_to_openai_function(Sentences)

        # Define the prompt template with placeholders
        prompt_template = PromptTemplate(messages=messages,
            input_variables=["topic", "num_sentences","subtopic"],
            template="generate list of {num_sentences} sentences for given topic:{topic} with subtopic:{subtopic} only provide list of sentences",
        )

        # Create an LLMChain
        model_forced_function = llm.bind(functions=[generate_subtopics_function], function_call={"name":"Sentences"})
        chain = prompt_template | model_forced_function


        # Run the chain to get the structured output
        response= chain.invoke({"topic": topic, "num_sentences": num_sentences,"subtopic":subtopic})

        # Extract the subtopics from the structured output
        data_str=response.additional_kwargs.get("function_call").get("arguments")
        data_dict = json.loads(data_str)

        # Extract the list from the dictionary
        return data_dict["sentences_list"]


    def synthetic_data(self):
        # creating  senetence dictionary using created 30 subjects ,subtoptopics_dictionary
        with open(self.config.subtopics_dict, 'r') as f:
            subtopics_dict= json.load(f)
        logger.info(f"subtopics_dictloaded sucessfully !")
        sentences_dict={}

        
        # Populate the dictionary for upto 30 topics
        for topic in list(subtopics_dict.keys())[0:10]:
            api_calls=0
            if topic not in sentences_dict.keys():
               sentences_dict[topic] = []
            try:
                for subtopic in subtopics_dict[topic][:1]:
                    
                        x=time.time()
                        generated_sentences = self.generate_sentences(topic,10,subtopic)

                        # Only append if unique subtopics are added
                        if generated_sentences :
                            sentences_dict[topic].append(generated_sentences)
                            api_calls+=1

                        # Check and re-generate if less than 10 sentences
                        if len(generated_sentences) < 10:
                            additional_sentences =self.generate_sentences(topic,10,subtopic)
                            api_calls+=1
                            if additional_sentences:
                                sentences_dict[topic].append(additional_sentences)
                time_taken=time.time()-x
                
                logger.info(f"time taken for {topic} Sentences creation :{time_taken}")
                logger.info(f"api call taken for {topic} Sentences creation :{api_calls}")
            except Exception as e:
                logger.info(f"Error generating sentences for subtopic '{topic}': {str(e)}")
                continue
            sentences_dict[topic] = flatten_and_deduplicate(sentences_dict[topic])

            logger.info(f"{topic}:\n{len(sentences_dict[topic])}\n")
        path=os.path.join(self.config.sentences_dict)
        with open(path,"w") as f:
            json.dump(sentences_dict,f)
        logger.info(f"sentences_dict saved to : {path}")
        
    def convert_to_dataframe(self):
        
        with open(self.config.sentences_dict,"r") as f:
            sentences_dict=json.load(f)
        dataset={}
        dataset["sentence"]=[]
        dataset["subject"]=[]
        for i in sentences_dict.keys():
            for j in sentences_dict[i]:
                dataset["sentence"].append(j)
                dataset["subject"].append(i)
            
        dataset1=pd.DataFrame(dataset)
        path=os.path.join(self.config.local_data_file)
        sample_data=dataset1.sample(self.config.sample_num,random_state=42)
        #saving data to the local in csv format
        sample_data.to_csv(path,index=False)
        logger.info(f"sample_data saved to : {path}")
    
