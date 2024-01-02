# Importing packages
import numpy as np 
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Creating an app instance in Flask
app = Flask(__name__)

# Creating a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Creating the prediction path
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        # Converting the data from form to dataframe
        pred_df = data.get_data_as_data_frame()
        
        # Instantiating the prediction pipeline and making predictions
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
        
    