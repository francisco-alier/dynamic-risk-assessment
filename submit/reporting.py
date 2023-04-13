import pandas as pd
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from diagnostics import model_predictions

logging.basicConfig(filename='logs/reporting.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path'])


##############Function for reporting
def plot_confusion_matrix():
    """
    calculates a confusion matrix using the test data and the deployed model
    and writes the confusion matrix to the workspace in a png format
    """
    
    logging.info(f"Reading test data from {test_data_path}")
    data_test = pd.read_csv(os.getcwd()+ "/" + test_data_path + "/" + "testdata.csv")
    
    #seperate data
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X_test = data_test[features]
    y_test = data_test["exited"]
    
    logging.info("Predicting test data")
    y_pred = model_predictions(X_test)

    logging.info("Calculating confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    logging.info("Plotting and saving confusion matrix")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.getcwd()+ "/" + output_model_path + "/" + "confusion_matrix.png")
    
    return None


if __name__ == '__main__':
    plot_confusion_matrix()
