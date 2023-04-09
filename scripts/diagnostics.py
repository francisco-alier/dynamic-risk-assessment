
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import logging
import subprocess
import pip

logging.basicConfig(filename='logs/diagnosis.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 

##################Function to get model predictions
def model_predictions(X_inf: pd.DataFrame):
    """
    read the deployed model and a test dataset, calculate predictions
    
    arguments:
    - X_inf: dataframe with features for inference
    
    returns:
    - y_inf: list containing all predictions
    """
    
    #read model
    logging.info(f"Loading classification model")
    lr = pickle.load(open(os.getcwd()+ "/" + model_path + "/" + "trainedmodel.pkl", 'rb'))
    
    logging.info("Predicting on data")
    y_inf = lr.predict(X_inf)
    
    return y_inf

##################Function to get summary statistics
def dataframe_summary():
    """
    returns summary statistics (mean, median and std deviatio) for all numeric columns
    """
    logging.info("Reading finaldata.csv")
    data = pd.read_csv(os.getcwd()+ "/" + dataset_csv_path + "/" + "finaldata.csv")
    
    logging.info("Selecting numerical columns")
    num_cols = [var for var in data.columns if data[var].dtype != "O"]
    
    stats = {}
    for col in num_cols:
        mean = data[col].mean()
        median = data[col].median()
        std = data[col].std()
    
        stats[col] = {"mean": mean, "median": median, "std, deviation": std}
        
        logging.info(f"Statistics for provided dataset and feature {col}: {stats[col]}")
    
    
    return stats


def get_missings():
    """
    Calculates percentage of missing data for each available column
    
    Returns:
        missings (dict): For every column name get percentage
    """
    
    logging.info("Reading finaldata.csv")
    data = pd.read_csv(os.getcwd()+ "/" + dataset_csv_path + "/" + "finaldata.csv")

    logging.info("Calculating missing data")
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_percentage_lst = list(percent_missing)
    
    return missing_percentage_lst


##################Function to get timings
def execution_time():
    """
    calculate timing of training.py and ingestion.py
    
    returns:
        timings (list): list with measurement of ingestion and training modules in seconds
    """
    
    logging.info("Timing the ingestion.py module")
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing_ingestion = timeit.default_timer() - starttime
    
    logging.info("Timing the training.py module")
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'training.py'], capture_output=True)
    timing_training = timeit.default_timer() - starttime
    
    timings = [str(timing_ingestion), str(timing_training)]
    
    return timings

##################Function to check dependencies
def outdated_packages_list():
    """
    Check dependencies status from requirements.txt file using pip-outdated
    Returns:
        dependencies: a table with three columns: the first column will show the name of a Python module that you're using; the second column will show the currently installed version of that Python module, and the third column will show the most recent available version of that Python module.
    """
    
    
    logging.info("Checking outdated dependencies")
    # Get a list of installed packages and their versions
    with open("requirements.txt") as f:
        packages = [line.strip().split('==')[0] for line in f if not line.startswith('#')]
        
    # Create a dictionary to store the installed package versions
    installed_versions = {}
    for package in packages:
        try:
            installed_version = pip.get_installed_distributions()[package].version
            installed_versions[package] = installed_version
        except:
            installed_versions[package] = 'Not installed'
    
    # Create a dictionary to store the latest available package versions
    latest_versions = {}
    for package in installed_versions:
        try:
            latest_version = pip.get_latest_version(package)
            latest_versions[package] = latest_version
        except:
            latest_versions[package] = 'Unknown'
    
    # Print a table of the package versions
    print('{:<20} {:<20} {:<20}'.format('Package', 'Installed Version', 'Latest Version'))
    packages = []
    installed = []
    lastest = []
    
    for package in installed_versions:
        print('{:<20} {:<20} {:<20}'.format(package, installed_versions[package], latest_versions[package]))
        
        packages.append(package)
        installed.append(installed_versions[package])
        lastest.append(latest_versions[package])
        
    outdated = {"Package": packages, "Installed Version": installed, 'Latest Version': lastest}
    
    return outdated 


if __name__ == '__main__':
    
    data = pd.read_csv(os.getcwd()+ "/" + test_data_path + "/" + "testdata.csv")
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X_inf = data[features]
    

    print("Model predictions on testdata.csv:",
          model_predictions(X_inf), end='\n\n')

    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4), end='\n\n')

    print("Missing percentage")
    print(json.dumps(get_missings(), indent=4), end='\n\n')

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end='\n\n')

    print("Outdated Packages")
    dependencies = outdated_packages_list()
    print(dependencies)





    
