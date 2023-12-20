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
            
            # Listing a set of params for Hyperparameter tuning
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            # Creating a variable to store all model evaluations
            model_reports:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
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