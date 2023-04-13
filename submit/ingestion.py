"""
Author: Francisco Nogueira
Date: April 2023
This script is used for ingesting data
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging


logging.basicConfig(filename='logs/ingest_data.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    """
    This function compiles all csv files in a single data set ready for modeling
    """
    #check for datasets, compile them together, and write to an output file
    
    final_dataframe = pd.DataFrame()
    ingested_files = []
    
    logging.info(f"Reading files from {input_folder_path}")
    
    filenames = [filename for filename in os.listdir(os.getcwd() + "/" + input_folder_path) if filename.endswith(".csv")]
    
    for each_filename in filenames:
        currentdf = pd.read_csv(os.getcwd()+ "/" + input_folder_path + "/" + each_filename)
        final_dataframe = final_dataframe.append(currentdf).reset_index(drop=True)
        ingested_files.append(each_filename)
    
    #Get duplicates out
    logging.info("Removing duplicates")
    final_dataframe_no_dups = final_dataframe.drop_duplicates().reset_index(drop=1)
    
    
    #Save information on read CSV's
    logging.info("Saving info on CSV's")
    with open(os.path.join(os.getcwd()+ "/" + output_folder_path + "/" + 'ingestedfiles.txt'), "w") as file:
        file.write(";".join(ingested_files))
        
    #saving final dataset
    if not os.path.exists(os.getcwd() + "/" + output_folder_path):
        os.makedirs(os.getcwd() + "/" + output_folder_path)
    
    logging.info("Write final data")
    final_dataframe_no_dups.to_csv(os.getcwd() + "/" + output_folder_path + "/" + "finaldata.csv", index = False)
    
    return None

if __name__ == '__main__':
    merge_multiple_dataframe()
