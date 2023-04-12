
import logging
import json
import os

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


    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here



    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model


if __name__ == '__main__':
    full_process()





