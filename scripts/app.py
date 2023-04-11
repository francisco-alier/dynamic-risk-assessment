from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import subprocess
import diagnostics

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

@app.route('/')
def index():
    print("Starting index...")
    return "Hello World"


@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """
    call the prediction function you created in Step 3
    
    Returns:
        preds_json: model predictions
    """
    filepath = request.get_json()['filepath']

    data = pd.read_csv(filepath)
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X = data[features]
    

    preds = diagnostics.model_predictions(X)
    preds_json = jsonify(preds.tolist())
    
    return preds_json


@app.route("/scoring", methods=['GET','OPTIONS'])
def calculate_score():        
    """
    Scoring endpoint that runs the script scoring.py and
    gets the score of the deployed model
    
    Returns:
        str: model f1 score
    """
    
    f1_score = subprocess.run(['python', 'scoring.py'],capture_output=True)
    
    return f1_score


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    """
    Calls summary statistics functions from diagnostics module
    """
    return jsonify(diagnostics.dataframe_summary())


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    
    #Get functions results
    missing = diagnostics.get_missings()
    time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()

    results = {
        'missing_percentage': missing,
        'execution_time': time,
        'outdated_packages': outdated
    }

    return jsonify(results)


#######################Prediction Endpoint
if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
    #app.run(debug=True, threaded=True)