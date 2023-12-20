# Importing packages
import numpy as np 
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

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
        pass
    