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
            
            # Creating the artifact directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Saving the raw data in the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Creating the train and test split of the dataframe
            logging.info("Initiating the Train and Test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Saving the train and test set in the Artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data has been completed.")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
             
        except Exception as e:
            raise CustomException(e, sys)
        