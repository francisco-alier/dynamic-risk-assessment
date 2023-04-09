"""
Author: Francisco Nogueira
Date: April 2023
This script copies files into the production directory
"""

import json
import shutil
import os
import logging
    
logging.basicConfig(filename='logs/deployment.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path']) 


####################function for deployment
def store_model_into_pickle():
    """
    copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    """
    
    #cleaning and leaving some arguments
    files_paths = [os.getcwd()+ "/" + model_path + "/" + "trainedmodel.pkl",
                   os.getcwd()+ "/" + model_path + "/" + "latestscore.txt",
                   os.getcwd()+ "/" + dataset_csv_path + "/" + "ingestedfiles.txt"
                    ]
    dst_file_paths = [os.getcwd()+ "/" + prod_deployment_path + "/" + "trainedmodel.pkl",
                   os.getcwd()+ "/" + prod_deployment_path + "/" + "latestscore.txt",
                   os.getcwd()+ "/" + prod_deployment_path + "/" + "ingestedfiles.txt"
                    ]
    if not os.path.exists(os.getcwd() + "/" + prod_deployment_path):
        os.makedirs(os.getcwd() + "/" + prod_deployment_path)

    logging.info(f"Copying files trainedmodel.pkl, ingestfiles.txt and latestscore.txt into production directory {prod_deployment_path}")
    for file, dst in zip(files_paths, dst_file_paths):
        shutil.copyfile(file, dst)

    return None


if __name__ == '__main__':
    store_model_into_pickle()
    
        
        

