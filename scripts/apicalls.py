import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:5000"
with open('../config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])

#Call each API endpoint and store the responses
response1 = requests.post(f'{URL}/prediction',json={'filepath': os.getcwd()+ "/" + test_data_path + "/" + "testdata.csv"}).text
response2 = requests.get(f'{URL}/scoring').text
response3 = requests.get(f'{URL}/summarystats').text
response4 = requests.get(f'{URL}/diagnostics').text

#combine all API responses
responses = {"prediction": response1,
             "scoring": response2,
             "summarystats": response3,
             "diagnostics": response4}

with open(os.path.join(os.getcwd()+ "/" + test_data_path + "/", 'apireturns.txt'), 'w') as f: 
    for key, value in responses.items(): 
        f.write('%s:%s\n' % (key, value))
