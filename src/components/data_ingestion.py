# Importing packages
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


# Defining the data ingestion class variables using dataclasses
@dataclass
class DataIngestionConfig():
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


# Creating the Data Ingestion class
class DataIngestion():
    # Creating the constructor
    def __init__(self):
        self.ingestion_config =  DataIngestionConfig()
    
    # Creating a function to ingest the data
    def initiate_data_ingestion(self):
        logging.info("Started the data ingestion process.")
        try:
            # Reading as pandas dataframe
            df = pd.read_csv('Notebook\Data\student.csv')
            logging.info("Read the dataset as a pandas dataframe.")
        except Exception as e:
            raise CustomException(e, sys)
        