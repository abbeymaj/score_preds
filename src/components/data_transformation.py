# Importing packages
import numpy as np
import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Creating a data transformation config class

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


# Creating the data transformation class
class DataTransformation():
    # Creating the constructor
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    # Creating a function for data transformation
    def get_data_transformation_object(self):
        '''
        This function is responsible for transforming the data.
        '''
        try:
            # Defining numerical and categorical columns
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            # Instantiating the numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            logging.info("Numerical columns standard scaling completed.")
            
            # Instantiating the categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder()),
                    ('cat_scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Categorical columns encoding completed.")
            
            # Constructing the column transformer
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    # Creating a function to initiate the data transformation    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading the train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Finished reading the train and test data.")
            
            # Instantiating the preprocessing object
            preprocessing_obj = self.get_data_transformation_object()
            
            logging.info("Instantiated the preprocessing object.")
            
            # Defining the target column 
            target_column = "math_score"
            numerical_columns = ['writing_score', 'reading_score']
            
            # Dropping the target column from the train set and setting the target set for train set
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            # Dropping the target column from the test set and setting the target set for test set
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            # Preprocessing the train and test datasets
            logging.info("Preprocessing the training and test datasets.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                object = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )  
        
        except Exception as e:
            raise CustomException(e, sys)