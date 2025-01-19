# for each folder 
import os
import json
path = 'results/evaluations/drugs/sentence_classification/12_16_first_try'
result = {}
for case_dir, dirs, files in os.walk(path):
    for file in files:
        #if file is json
        if file.endswith('.json'):
            # take this value in the file "PRAUC": 0.8630478187105187
            with open(os.path.join(case_dir, file)) as f:
                data = json.load(f)
                key = case_dir.split('12_16_first_try/')[-1].split('/')[0]
                result[key] = round(data["PRAUC"], 5)

# to csv
import pandas as pd
df = pd.DataFrame(result.items(), columns=['case_dir', 'PRAUC'])
df.to_csv('results/evaluations/drugs/sentence_classification/12_16_first_try/result.csv')