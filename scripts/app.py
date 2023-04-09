from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import model_predictions, dataframe_summary, get_missings, execution_time, outdated_packages_list


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
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
    

    preds = model_predictions(X)
    preds_json = jsonify(preds.tolist())
    
    return preds_json


#######################Scoring Endpoint
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
    return jsonify(dataframe_summary())


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    
    #Get functions results
    missing = get_missings()
    time = execution_time()
    outdated = outdated_packages_list()

    results = {
        'missing_percentage': missing,
        'execution_time': time,
        'outdated_packages': outdated
    }

    return jsonify(results)


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
