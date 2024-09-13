
    
import os
import itertools
import time
import json
import pandas as pd
import warnings
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.output_parsers import StructuredOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from src.SubjectClassifier import logger
from src.SubjectClassifier.utils.common import get_size
from src.SubjectClassifier.entity.config_entity import DataIngestionConfig

warnings.filterwarnings("ignore")

# Function to flatten and deduplicate sentences
def flatten_and_deduplicate(sentences):
    return list(set(
        itertools.chain.from_iterable(
            [d["sentence"] if isinstance(d, dict) and "sentence" in d else d for d in sublist]
            for sublist in sentences if isinstance(sublist, list)
        )
    ))

load_dotenv()
llm = ChatGroq(model="llama3-70b-8192")

class Sentences(BaseModel):
    """Schema for generating a list of sentences from a given topic with subtopics."""
    sentences_list: List[str] = Field(description="List of sentences for a given topic with subtopic.")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def generate_sentences(self, topic: str, num_sentences: int, subtopic: str):
        """Generates sentences for a given topic and subtopic."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output a list of sentences for a provided topic and subtopic."},
        ]

        # Define function conversion for structured output
        generate_subtopics_function = convert_pydantic_to_openai_function(Sentences)

        # Define the prompt template with placeholders
        prompt_template = PromptTemplate(
            messages=messages,
            input_variables=["topic", "num_sentences", "subtopic"],
            template="Generate a list of {num_sentences} sentences for the topic: {topic} with subtopic: {subtopic}. Provide only a list of sentences.",
        )

        # Create an LLMChain and invoke the chain
        model_forced_function = llm.bind(functions=[generate_subtopics_function], function_call={"name": "Sentences"})
        chain = prompt_template | model_forced_function
        response = chain.invoke({"topic": topic, "num_sentences": num_sentences, "subtopic": subtopic})

        # Extract and return sentences from response
        data_str = response.additional_kwargs.get("function_call").get("arguments")
        data_dict = json.loads(data_str)
        return data_dict["sentences_list"]

    def synthetic_data(self):
        """Generates synthetic data for 30 subjects and subtopics and saves the sentences in a dictionary."""
        # Load subtopics dictionary
        with open(self.config.subtopics_dict, 'r') as f:
            subtopics_dict = json.load(f)
        logger.info("Subtopics dictionary loaded successfully!")

        sentences_dict = {}

        # Iterate through topics and subtopics
        for topic in list(subtopics_dict.keys())[:10]:
            api_calls = 0
            sentences_dict[topic] = []
            try:
                for subtopic in subtopics_dict[topic][:100]:
                    start_time = time.time()

                    generated_sentences = self.generate_sentences(topic, 10, subtopic)
                    api_calls += 1

                    if generated_sentences:
                        sentences_dict[topic].append(generated_sentences)

                    # Ensure a minimum of 10 sentences
                    if len(generated_sentences) < 10:
                        additional_sentences = self.generate_sentences(topic, 10, subtopic)
                        api_calls += 1
                        if additional_sentences:
                            sentences_dict[topic].append(additional_sentences)

                    time_taken = time.time() - start_time
                    logger.info(f"Time taken for {topic}: {time_taken:.2f}s, API calls: {api_calls}")

            except Exception as e:
                logger.error(f"Error generating sentences for {topic}: {e}")
                continue

            # Flatten and deduplicate sentences
            sentences_dict[topic] = flatten_and_deduplicate(sentences_dict[topic])
            logger.info(f"{topic} - {len(sentences_dict[topic])} sentences generated.")

        # Save the generated sentences dictionary to a file
        path = os.path.join(self.config.sentences_dict)
        with open(path, "w") as f:
            json.dump(sentences_dict, f)
        logger.info(f"Sentences dictionary saved to: {path}")

    def convert_to_dataframe(self):
        """Converts the generated sentences into a DataFrame and saves it as a CSV."""
        # Load sentences dictionary
        with open(self.config.sentences_dict, "r") as f:
            sentences_dict = json.load(f)

        dataset = {"sentence": [], "subject": []}

        # Prepare the dataset
        for subject, sentences in sentences_dict.items():
            for sentence in sentences:
                dataset["sentence"].append(sentence)
                dataset["subject"].append(subject)

        # Create DataFrame and save as CSV
        dataset_df = pd.DataFrame(dataset)
        sample_data = dataset_df.sample(self.config.sample_num, random_state=42)
        path = os.path.join(self.config.local_data_file)
        sample_data.to_csv(path, index=False)
        logger.info(f"Sampled data saved to: {path}")
