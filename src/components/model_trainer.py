# Importing packages
import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Creating the model trainer config class
@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# Creating a class to train the model
class ModelTrainer():
    # Creating the constructor
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    # Creating a function to initiate model training
    def initiate_model_trainer(self, training_array, test_array):
        '''
        This function initiates the training of the model.
        '''
        try:
            # Splitting the datasets into feature and target sets
            logging.info("Splitting the train and test datasets into feature and target.")
            X_train, y_train, X_test, y_test = (
                training_array[:, :-1],
                training_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Creating a dictionary of all models
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(),
                'Catboost Regressor': CatBoostRegressor(),
                'Adaboost Regressor': AdaBoostRegressor()
            }
            
            # Creating a variable to store all model evaluations
            model_reports:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # Obtaining the best model score from the dictionary
            best_model_score = max(sorted(model_reports.values()))
            
            # Obtaining the best model name from the dictionary
            best_model_name = list(model_reports.keys())[list(model_reports.values()).index(best_model_score)]
            
            # Instantiating the best model
            best_model = models[best_model_name]
            
            # Creating a custom exception to indicate no best model found if best model score < 0.6
            if best_model_score < 0.6:
                raise CustomException("No best model found!")
            
            logging.info(f"Best model found on training and test datasets.")
            
            # Saving the model path
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )
            
            # Validating the best model's predictive score
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2
        
        except Exception as e:
            raise CustomException(e, sys)