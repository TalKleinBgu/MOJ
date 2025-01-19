import pandas as pd
import os


def compare_specipic_feature(target_feature:str):
    db_path = '/home/ezradin/pred-sentencing/results/db/2017'
    fe_tagged_path = '/home/ezradin/pred-sentencing/resources/data/tagging/gt/Feature Extraction.xlsx'
    similarity_path = '/home/ezradin/pred-sentencing/resources/data/tagging/gt/similarity_paris.csv'
    
    fe_tagged_df = pd.read_excel(fe_tagged_path)
    similarity_df = pd.read_csv(similarity_path)
    
    for root, dirs, files in os.walk(db_path):
        for file in files:
            case_name = root.split('/')[-1]
            if file == 'qa_features.csv':
                file_path = os.path.join(root, file)
                qa_df = pd.read_csv(file_path)
                print()
                

compare_specipic_feature('OFFENCE_TYPE')