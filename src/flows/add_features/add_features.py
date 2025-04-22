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
    # metadata = pd.read_csv(params['metadata_path'])
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
                    # for index, row in metadata.iterrows():
                    #     row_update = row['Case Number'].split('-')
                    #     # change betwen the first ans last in the list- row_update and keep other
                    #     if len(row_update) > 1:
                    #         row_update[0], row_update[-1] = row_update[-1], row_update[0]
                    #     row_update = '-'.join(row_update)
                    #     if row_update in dir:
                    #         columns_to_add = ['Date', 'Judges', 'Prosecutores', 'Defendants', 'Court', 'Location', 'Defendants_Race', 'Judges_Race']
                    #         for col in columns_to_add:
                    #             aggregated_row[col] = row[col]
                    #         break
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


    #DELETE CIR_PUNISHMENT column
    if 'CIR_PUNISHMENT' in final_df.columns:
        del final_df['CIR_PUNISHMENT']
    
    final_df.to_csv(final_path, index=False, encoding='utf-8')

    print(f"Aggregated data saved to {final_df}")


def main_2(params):
    final_df = pd.read_csv(os.path.join(params['output_path'], 'features_extraction_cases.csv'))
    #ITERATE OVER ROWS
    for index, row in final_df.iterrows():
        ######## CONFESSION ########
        if type(row['CONFESSION']) == float:
            #update df to no
            final_df.at[index, 'CONFESSION'] = 'לא'
        elif 'כן' in row['CONFESSION']:
            #update df to yes
            final_df.at[index, 'CONFESSION'] = 'כן'
        else:
            #update df to no
            final_df.at[index, 'CONFESSION'] = 'לא'

        ######### RESPO ##########
        if type(row['RESPO']) == float:
            #update df to no
            final_df.at[index, 'RESPO'] = 'לא'
        elif 'כן' in row['RESPO']:
            #update df to yes
            final_df.at[index, 'RESPO'] = 'כן'
        else:
            #update df to no
            final_df.at[index, 'RESPO'] = 'לא'

        ######### REGRET ##########
        if type(row['REGRET']) == float:
            #update df to no
            final_df.at[index, 'REGRET'] = 'לא'
        elif 'כן' in row['REGRET']:
            #update df to yes
            final_df.at[index, 'REGRET'] = 'כן'
        else:
            #update df to no
            final_df.at[index, 'REGRET'] = 'לא'

        ######### PUNISHMENT_Range ##########
        punishment_range = row['PUNISHMENT_Range']
        if isinstance(punishment_range, float):
            #update df to no
            final_df.at[index, 'PUNISHMENT_Range'] = ''
        else:
            punishment_range = ast.literal_eval(punishment_range)
            if isinstance(punishment_range, dict):
                try:
                    minimun = punishment_range.get('lower', '')
                    maximum = punishment_range.get('top', '')
                    punishment_range = f"{minimun} - {maximum}"
                    final_df.at[index, 'PUNISHMENT_Range'] = punishment_range
                except Exception as e:
                    print(f"Error processing punishment_range: {e}")
                    final_df.at[index, 'PUNISHMENT_Range'] = ''
            else:
                # If it's not a dictionary, just keep it as is
                final_df.at[index, 'PUNISHMENT_Range'] = ''

        ######### PUNISHMENT_FINE ##########
        fines = row['PUNISHMENT_FINE']
        if isinstance(fines, float):
            #update df to no
            final_df.at[index, 'PUNISHMENT_FINE'] = ''
        else:
            fines = fines.split('] [')
            numbers = []
            for fine in fines:
                all_digits = ''
                for char in fine:
                    if char.isdigit():
                        all_digits += char
                if all_digits != '':
                    numbers.append(all_digits)
            # Check if the list is empty
            if len(numbers) == 0:
                fine = ''
            else:
                #remove duplicates
                numbers = list(set(numbers))
            if len(numbers) >= 1:
                real_fine = f'{numbers[0]} שח'
                final_df.at[index, 'PUNISHMENT_FINE'] = real_fine
            else:
                final_df.at[index, 'PUNISHMENT_FINE'] = ''

        ######### PUNISHMENT_SUSPENDED ##########
        suspended = row['PUNISHMENT_SUSPENDED']
        if isinstance(suspended, float):
            #update df to no
            final_df.at[index, 'PUNISHMENT_SUSPENDED'] = ''
        else:
            suspended = suspended.split('] [')
            numbers = []
            for suspend in suspended:
                all_digits = ''
                for char in suspend:
                    if char.isdigit():
                        all_digits += char
                if all_digits != '':
                    numbers.append(all_digits)
            # Check if the list is empty
            if len(numbers) == 0:
                suspend = ''
            else:
                #remove duplicates
                numbers = list(set(numbers))
            if len(numbers) >= 1:
                real_suspend = ''
                for number in numbers:
                    real_suspend += f'{number} חודשי מאסר על תנאי, '
                #remove the last ', '
                real_suspend = real_suspend[:-2]
                final_df.at[index, 'PUNISHMENT_SUSPENDED'] = real_suspend
            else:
                final_df.at[index, 'PUNISHMENT_SUSPENDED'] = ''
            
            ######### PUNISHMENT_ACTUAL ##########
        actual = row['PUNISHMENT_ACTUAL']
        if isinstance(actual, float):
            #update df to no
            final_df.at[index, 'PUNISHMENT_ACTUAL'] = ''
        else:
            actuals = actual.split('] [')
            numbers = []
            actual = actuals[-1]
            if actual == ']':
                if len(actuals) > 1:
                    actual = actuals[-2]
                else:
                    actual = ''
            all_digits = ''
            for char in actual:
                if char.isdigit():
                    all_digits += char
            if all_digits != '':
                numbers.append(all_digits)
            # Check if the list is empty
            if len(numbers) == 0:
                actual = ''
            else:
                #remove duplicates
                numbers = list(set(numbers))
            if len(numbers) >= 1:
                real_actual = f'{numbers[0]} חודשי מאסר בפועל'
                final_df.at[index, 'PUNISHMENT_ACTUAL'] = real_actual
            else:
                final_df.at[index, 'PUNISHMENT_ACTUAL'] = ''

        ########## CIR_EQ ##########
        cir_eq = row['CIR_EQ']
        if isinstance(cir_eq, float):
            #update df to no
            final_df.at[index, 'CIR_EQ'] = 'לא'
        elif 'כן' in cir_eq:
            #update df to yes
            final_df.at[index, 'CIR_EQ'] = 'כן'
        else:  
            #update df to no
            final_df.at[index, 'CIR_EQ'] = 'לא'

        ########## CIR_ROLE ##########
        cir_role = row['CIR_ROLE']
        if isinstance(cir_role, float):
            #update df to no
            final_df.at[index, 'CIR_ROLE'] = ''
        else:
            all_list = []
            if 'לא בעל המעבדה' in cir_role:
                all_list.append('לא בעל המעבדה')
            if 'לא בעל הסמים' in cir_role:
                all_list.append('לא בעל הסמים')
            # if 'בעל המעבדה' in cir_role:
            #     all_list.append('בעל המעבדה')
            if 'בעל הסמים' in cir_role:
                all_list.append('בעל הסמים')
            #from list to string
            cir_role = ', '.join(all_list)
            #update df to no
            final_df.at[index, 'CIR_ROLE'] = cir_role

        ########## CIR_TYPE ##########
        cir_type = row['CIR_TYPE']
        if isinstance(cir_type, float):
            #update df to no
            final_df.at[index, 'CIR_TYPE'] = ''
        else:
            pattern = r'\[\'(.*?)\'\]'
            matches = re.findall(pattern, cir_type)
            all_types = []
            if len(matches) > 0:
                for match in matches:
                    text = match.split(' ')
                    # find index of הוא
                    try:
                        index1 = text.index('הוא')
                        #get the next word
                        if index1 + 1 < len(text):
                            next_word = text[index1 + 1]
                            all_types.append(next_word)
                    except ValueError:
                        # 'הוא' not found in the list
                        pass
            #update df to no
            final_df.at[index, 'CIR_TYPE'] = all_types

        ########## CIR_AMOUNT ##########
        cir_amount = row['CIR_AMOUNT']
        if isinstance(cir_amount, float):
            #update df to no
            final_df.at[index, 'CIR_AMOUNT'] = ''
        else:
            pattern = r'\[[^\[\]]+\]'
            matches = re.findall(pattern, cir_amount)
            all_amounts = []
            if len(matches) > 0:
                for match in matches:
                    match = match.replace('[', '').replace(']', '')
                    text = match.split(' - ')
                    all_amounts.append(f'{text[0]} {text[1]}')
                # print(row)
            #update df to no
            final_df.at[index, 'CIR_AMOUNT'] = all_amounts

    #save the df to the csv file
    final_path = os.path.join(params['output_path'], 'features_extraction_cases_2.csv')
    #delet CIRCUM_OFFENSE column
    if 'CIRCUM_OFFENSE' in final_df.columns:
        del final_df['CIRCUM_OFFENSE']
    final_df.to_csv(final_path, index=False, encoding='utf-8')

def main3(params):     
    final_df = pd.read_csv(os.path.join(params['output_path'], 'features_extraction_cases_2.csv'))
    metadata = pd.read_csv('/home/tak/MOJ/resources/data/tagging/drugs/gt/one_defendant_new_.csv')
    aggregated_rows = []

    for index, row in metadata.iterrows():
        row_update = row['Case Number'].split('-')
        # change betwen the first ans last in the list- row_update and keep other
        if len(row_update) > 1:
            row_update[0], row_update[-1] = row_update[-1], row_update[0]
        row_update = '-'.join(row_update)
        f = final_df['verdict'].values
        # f to list
        f = f.tolist()
        for t in f:
            if row_update in t:
                # Get the index of the matching row in final_df
                matching_index = final_df[final_df['verdict'] == t].index[0]
                # Create a new row with the same structure as final_df
                aggregated_row = final_df.iloc[matching_index].copy()
                # Update the aggregated_row with the values from metadata
                columns_to_add = ['Date', 'Judges', 'Prosecutores', 'Defendants', 'Court', 'Location', 'Defendants_Race', 'Judges_Race']
                for col in columns_to_add:
                    aggregated_row[col] = row[col]
                # Append the aggregated_row to the list
                aggregated_rows.append(aggregated_row)
                break
    # Combine all aggregated rows into a single DataFrame
    final_df = pd.DataFrame(aggregated_rows)
    final_path = os.path.join(params['output_path'], 'features_extraction_cases_3.csv')
    final_df.to_csv(final_path, index=False, encoding='utf-8')

def main4(params):
    # df_regular = pd.read_csv(os.path.join(params['output_path'], 'features_extraction_cases_3.csv'))
    # df_pun = pd.read_excel('/home/tak/MOJ/src/flows/final_gpt_drugs.xlsx')

    # # הוספת עמודה אם לא קיימת
    # if 'PUNISHMENT_Range' not in df_regular.columns:
    #     df_regular['PUNISHMENT_Range'] = None

    # for index, row in df_regular.iterrows():
    #     verdict_value = row['verdict']
    #     match = df_pun[df_pun['שם קובץ התיק'] == verdict_value]

    #     if not match.empty:
    #         extract_value = match.iloc[0]['extract']
    #         df_regular.at[index, 'PUNISHMENT_Range'] = extract_value
    # # שמירה
    # final_path = os.path.join(params['output_path'], 'features_extraction_cases_4.csv')
    # df_regular.to_csv(final_path, index=False, encoding='utf-8')
    df_regular = pd.read_csv(os.path.join(params['output_path'], 'features_extraction_cases_6.csv'))
    max_punishment = []
    min_punishment = []
    range_punishment = []
    for index, row in df_regular.iterrows():
        punishment_range = row['PUNISHMENT_Range']
        if isinstance(punishment_range, str):
            punishment_range = punishment_range.split(' ')
            if len(punishment_range) == 3:
                min_punishment.append(punishment_range[0])
                max_punishment.append(punishment_range[2])
                range = int(punishment_range[2]) - int(punishment_range[0]) 
                range_punishment.append(range)
            elif len(punishment_range) == 5:
                min_punishment.append(punishment_range[0])
                max_punishment.append(punishment_range[3])
                range = int(punishment_range[3]) - int(punishment_range[0]) 
                range_punishment.append(range)
        else:
            min_punishment.append('')
            max_punishment.append('')
            range_punishment.append('')
    # Add the new columns to the DataFrame
    df_regular['min_punishment'] = min_punishment
    df_regular['max_punishment'] = max_punishment
    df_regular['range_punishment'] = range_punishment
    # Save the updated DataFrame to a new CSV file
    final_path = os.path.join(params['output_path'], 'features_extraction_cases_6.csv')
    df_regular.to_csv(final_path, index=False, encoding='utf-8')
        
def main5(params):
    df = pd.read_csv('/home/tak/MOJ/results/evaluations/drugs/features_extraction/features_extraction_cases_5.csv')
    path = params['prediction_dfs']
    for index, row in df.iterrows():
        verdict_value = row['verdict']
        for file in os.listdir(os.path.join(path,verdict_value)):
            if file == 'punishment_range_gpt.json':
                with open(os.path.join(path, verdict_value, file), 'r') as f:
                    punishment_range = json.load(f)
                    # Update the 'PUNISHMENT_Range' column in the DataFrame
                    df.at[index, 'PUNISHMENT_Range'] = punishment_range[-1]
                    break
    # Save the updated DataFrame to a new CSV file
    final_path = os.path.join(params['output_path'], 'features_extraction_cases_6.csv')
    df.to_csv('/home/tak/MOJ/results/evaluations/drugs/features_extraction/features_extraction_cases_6.csv', index=False, encoding='utf-8')
def main6(params):
    df = pd.read_csv('/home/tak/MOJ/results/evaluations/drugs/features_extraction/features_extraction_cases_6.csv')
    all_offenses = []
    verdict_value = df['verdict'].values
    path = '/home/tak/MOJ/results/db/drugs'
    for t in verdict_value:
        for file in os.listdir(os.path.join(path,t)):
            # print(file)
            if file.startswith('qa'):
                data = pd.read_csv(os.path.join(path, t, file))
                # print(data)
                try:
                    # סינון לפי OFFENCE_NUMBER בשדה feature_key
                    filtered = data[data['feature_key'] == 'OFFENCE_NUMBER']
                    # print(data['feature_key'].unique())

                    # שליפת הערכים מתוך עמודת feature_value
                    OFFENCE_NUMBERs = filtered['extraction'].astype(str).tolist()
                    #remove ', [] from the list of strings
                    for i in range(len(OFFENCE_NUMBERs)):
                        OFFENCE_NUMBERs[i] = OFFENCE_NUMBERs[i].replace('[','').replace(']','').replace('\'','')
                    unique_clauses = set()

                    for item in OFFENCE_NUMBERs:
                        clauses = [clause.strip() for clause in item.split(',')]
                        unique_clauses.update(clauses)
                                        #remove the spaces from the strings

                    # print(filtered)
                    # ייחוד, מיזוג לרשימה מופרדת בפסיקים
                    # unique_values = ', '.join(sorted(set(OFFENCE_NUMBERs)))

                    all_offenses.append(unique_clauses)
                except:
                    all_offenses.append('')
                break
    # Add the new column to the DataFrame
    df['OFFENCE_NUMBER'] = all_offenses
    # Save the updated DataFrame to a new CSV file
    final_path = os.path.join(params['output_path'], 'features_extraction_cases_7.csv')
    df.to_csv(final_path, index=False, encoding='utf-8')
def updates():
    # Load the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(params['output_path'], 'features_extraction_cases.csv'))
    column_names = ['CIR_AMOUNT','CIR_TYPE']
    dirs_path = params['prediction_dfs']
    aggregated_rows = []
    # metadata = pd.read_csv(params['metadata_path'])
    for dir in os.listdir(dirs_path):
        if 'ME' in dir or 'SH' in dir:
            pred_path = os.path.join(dirs_path, dir)
            aggregated_row = {}
            for file in os.listdir(pred_path):
                if file == 'features_extraction.csv':
                    fe_df = pd.read_csv(os.path.join(pred_path, file))
                    for i,row in fe_df.iterrows():
                        # Update aggregated_row by appending unique values
                        if (type(row['CIR_AMOUNT']) == float and type(row['CIR_TYPE']) != float) or (type(row['CIR_AMOUNT']) != float and type(row['CIR_TYPE']) == float):
                            aggregated_row['CIR_AMOUNT'] = ''
                            aggregated_row['CIR_TYPE'] = ''
                            #delete the row
                            fe_df.drop(i, inplace=True)
                        if (type(row['CIR_AMOUNT']) == float and type(row['CIR_TYPE']) == float):
                            aggregated_row['CIR_AMOUNT'] = ''
                            aggregated_row['CIR_TYPE'] = ''
                            #delete the row
                            fe_df.drop(i, inplace=True)
                    for col_name in column_names:
                        #if in both columns have some values
                        # Get unique values as strings
                        unique_values = fe_df[col_name].dropna().astype(str)
                        
                        # If the column already exists, append new unique values
                        if col_name in aggregated_row:
                            existing_values = aggregated_row[col_name].split(" ")
                            # new_values = existing_values.union(unique_values)
                            aggregated_row[col_name] = " ".join(unique_values)
                        else:
                            # If column doesn't exist, initialize it with unique values
                            aggregated_row[col_name] = " ".join(unique_values)
                    
                    # Add the file name to identify the source (optional)
                    aggregated_row['verdict'] = dir

            aggregated_rows.append(aggregated_row)
    # Combine all aggregated rows into a single DataFrame
    final_df = pd.DataFrame(aggregated_rows)
    final_path = os.path.join(params['output_path'], 'features_extraction_cases.csv')
    column_names = ['CIR_AMOUNT','CIR_TYPE']
    #save the columns to the df variable
    df[column_names] = final_df[column_names]
    #save the df to the csv file
    df.to_csv(final_path, index=False, encoding='utf-8')

    #DELETE CIR_PUNISHMENT column

if __name__ == '__main__':
    # Load main configuration parameters
    main_params = config_parser("", "main_config")
    # Extract the domain from the main configuration
    domain = main_params["domain"]
    
    # Parse the specific feature extraction parameters for the given domain
    params = config_parser("add_features", domain)
    
    # Run the main function with the parsed parameters
    # main(params, domain)

    # main_2(params)
    # main3(params)  
    # main4(params)
    # main5(params)
    main6(params)
    # updates()