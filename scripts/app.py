from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import subprocess
from diagnostics import model_predictions, dataframe_summary, get_missings, execution_time, outdated_packages_list



with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

@app.route('/')
def index():
    return "Hello World"


#######################Prediction Endpoint
if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
    #app.run(debug=True, threaded=True)