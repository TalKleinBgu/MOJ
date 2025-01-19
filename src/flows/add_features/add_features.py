import os
import sys
import ast
import pandas as pd
import json
import re
from collections import defaultdict
from collections import Counter
import importlib

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir,
                                                    '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import config_parser, get_date, setup_logger
from utils.evaluation.weapon.evaluation import change_case_name_foramt, change_format, convert_tagging_to_pred_foramt
from utils.punishment_extractor import case_mapping_dict
from utils.features import *


# fe_df = fe_df.melt(
#         id_vars='Text',        # Columns to keep as identifiers (unchanged)
#         var_name='Feature_key',      # Name for the 'old' column headers
#         value_name='Extraction'     # Name for the values from those columns
#     )
#     fe_df = fe_df.dropna(subset=['value'])
#     fe_df = fe_df[['Feature_key', 'Text', 'Extraction']]


def main(params, domain):     
    
    logger = setup_logger(os.path.join(params['result_path'],
                        'logs'), file_name='add_feature_extraction_test')
    
    dirs_path = params['prediction_dfs']
    aggregated_rows = []
    metadata = pd.read_csv(params['metadata_path'])
    for dir in os.listdir(dirs_path):
        if 'ME' in dir or 'SH' in dir:
            pred_path = os.path.join(dirs_path, dir)
            aggregated_row = {}
            for file in os.listdir(pred_path):
                if file == 'features_extraction.csv':
                    fe_df = pd.read_csv(os.path.join(pred_path, file))
                    
                    # Update aggregated_row by appending unique values
                    for col_name in fe_df.columns:
                        # Get unique values as strings
                        unique_values = fe_df[col_name].dropna().astype(str).unique()
                        
                        # If the column already exists, append new unique values
                        if col_name in aggregated_row:
                            existing_values = set(aggregated_row[col_name].split(" "))
                            new_values = existing_values.union(set(unique_values))
                            aggregated_row[col_name] = " ".join(new_values)
                        else:
                            # If column doesn't exist, initialize it with unique values
                            aggregated_row[col_name] = " ".join(unique_values)
                    
                    # Add the file name to identify the source (optional)
                    aggregated_row['verdict'] = dir
                    for index, row in metadata.iterrows():
                        row_update = row['Case Number'].split('-')
                        # change betwen the first ans last in the list- row_update and keep other
                        if len(row_update) > 1:
                            row_update[0], row_update[-1] = row_update[-1], row_update[0]
                        row_update = '-'.join(row_update)
                        if row_update in dir:
                            columns_to_add = ['Date', 'Judges', 'Prosecutores', 'Defendants', 'Court', 'Location', 'Defendants_Race', 'Judges_Race']
                            for col in columns_to_add:
                                aggregated_row[col] = row[col]
                            break
                if file == 'punishment_range.json':
                    with open(os.path.join(pred_path, file), 'r') as f:
                        punishment_range = json.load(f)
                        aggregated_row['PUNISHMENT_Range'] = punishment_range
                # Append the aggregated row to the list
            aggregated_rows.append(aggregated_row)
    # Combine all aggregated rows into a single DataFrame
    final_df = pd.DataFrame(aggregated_rows)
    final_path = os.path.join(params['output_path'], 'features_extraction_cases.csv')
    columns_ro_remove = ['text', 'reject','PUNISHMENT','verdict']
    reorder_columns = ['verdict'] + [col for col in final_df.columns if col not in columns_ro_remove]
    final_df = final_df[reorder_columns]

    final_df.to_csv(final_path, index=False, encoding='utf-8')

    print(f"Aggregated data saved to {final_df}")



        
if __name__ == '__main__':
    # Load main configuration parameters
    main_params = config_parser("", "main_config")
    # Extract the domain from the main configuration
    domain = main_params["domain"]
    
    # Parse the specific feature extraction parameters for the given domain
    params = config_parser("add_features", domain)
    
    # Run the main function with the parsed parameters
    main(params, domain)