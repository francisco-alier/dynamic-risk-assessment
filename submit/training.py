"""
Author: Francisco Nogueira
Date: April 2023
This script is used for training an ML classification model
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json
import logging

logging.basicConfig(filename='logs/training.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    
    #read data
    logging.info(f"Reading files from {dataset_csv_path}")
    data = pd.read_csv(os.getcwd()+ "/" + dataset_csv_path + "/" + "finaldata.csv")
    
    #seperate features and y
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X = data[features]
    y = data["exited"]
    
    #use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None, max_iter=100,
                            multi_class='auto', n_jobs=None, penalty='l2',
                            random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                            warm_start=False)

    #fit the logistic regression to your data
    logging.info("Fitting Logistic Regression model")
    lr.fit(X,y)
    
    #Get simple metric
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    logging.info(f"AUC Score of model is {auc:.2%}")
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info("Saving Trained Model")
    if not os.path.exists(os.getcwd() + "/" + model_path):
        os.makedirs(os.getcwd() + "/" + model_path)
        
    pickle.dump(lr, open(os.getcwd()+ "/" + model_path + "/" + "trainedmodel.pkl", 'wb'))

    return None

    
if __name__ == '__main__':
    train_model()
    
