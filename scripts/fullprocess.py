
import logging
import json
import os
import pandas as pd
from sklearn.metrics import f1_score
import subprocess

import ingestion as ig
import training as tr
import scoring as sc
import deployment as dp
import diagnostics as dg
import reporting as rp

logging.basicConfig(filename='logs/fullprocess.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
input_folder_path = os.path.join(config['input_folder_path']) 
dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 

def full_process():
    """
    Function to run all scripts needed for the full pipeline
    """
    ##################Check and read new data
    #first, read ingestedfiles.txt
    logging.info("Step 1 - Check for new data")
    datasets = []
    with open(os.getcwd()+ "/" + prod_deployment_path + "/" + "ingestedfiles.txt") as file:
        for line in file:
            datasets.extend(line.strip().split(';'))
    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files_path = os.listdir(os.getcwd()+ "/" + input_folder_path)
    source_files = [file for file in source_files_path if file.endswith(".csv")]

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if len(set(source_files).difference(set(datasets))) == 0:
        logging.info("No new data!!")
        return None

    # If there is new data lets merge it!
    logging.info("Step 2 - Merging New Data!")
    ig.merge_multiple_dataframe()

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    logging.info("Step 3 - Model Drift Analysis")
    with open(os.getcwd()+ "/" + prod_deployment_path + "/" + "latestscore.txt") as file:
        deployed_score = file.read()
        deployed_score = float(deployed_score)

    new_data = pd.read_csv(os.getcwd() + "/" +  dataset_csv_path + "/" + 'finaldata.csv')
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X_test = new_data[features]
    y_test = new_data["exited"]
    
    y_pred = dg.model_predictions(X_test, model_path=prod_deployment_path)
    new_score = f1_score(y_test, y_pred)

    # Deciding whether to proceed, part 2
    logging.info(f"Deployed score = {deployed_score}")
    logging.info(f"New score = {new_score}")

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    #if(new_score >= deployed_score):
    #    logging.info("No model drift occurred")
    #    return None


    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
     # Re-training
    logging.info("Re-training model")
    tr.train_model()
    logging.info("Re-scoring model")
    sc.score_model()

     # Re-deployment
    logging.info("Re-deploying model")
    dp.store_model_into_pickle()
    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    logging.info("Calling the API")
    subprocess.run(["python", "scripts/apicalls.py"])

    #confusion matrix
    rp.plot_confusion_matrix()

    #return None


if __name__ == '__main__':
    full_process()





