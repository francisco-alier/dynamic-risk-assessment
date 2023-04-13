"""
Author: Francisco Nogueira
Date: April 2023
This script is used for scoring an ML classification model
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score
import json
import logging

logging.basicConfig(filename='logs/scoring.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 

#################Function for model scoring
def score_model():
    '''
    this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    it should write the result to the latestscore.txt file
    '''
    
    #read data
    logging.info(f"Reading test data from {test_data_path}")
    data_test = pd.read_csv(os.getcwd()+ "/" + test_data_path + "/" + "testdata.csv")
    
    #read model
    logging.info(f"Loading classification model")
    lr = pickle.load(open(os.getcwd()+ "/" + model_path + "/" + "trainedmodel.pkl", 'rb'))
    
    #split features and label
    logging.info(f"Preparing to calculate f1 score...")
    
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X_test = data_test[features]
    y_test = data_test["exited"]
    
    #F1 score
    y_pred = lr.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    logging.info(f"F1 score on test set is {f1:.2%}")
    
    #Saving f1 score
    logging.info(f"Saving f1 score on {model_path}")
    with open(os.path.join(os.getcwd()+ "/" + model_path + "/" + 'latestscore.txt'), "w") as file:
        file.write(str(f1))
        
    return print(f"{f1}")

if __name__ == '__main__':
    score_model()