# Import packages
import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException

# Creating a function to save the preprocessing object
def save_object(file_path, object):
    try:
        # Creating the directory to store the preprocessing object
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(object, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)