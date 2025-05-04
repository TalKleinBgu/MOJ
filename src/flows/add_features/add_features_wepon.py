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


def aggregate_data(params, domain):     
    
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
                    print(dir)
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
    
    # final_df.to_csv(final_path, index=False, encoding='utf-8')

    # print(f"Aggregated data saved to {final_df}")
    return final_df

def create(params, logger):

    columns_translation = {
        'מספר תיק': 'CASE_NUMBER',
        'הודאה': 'CONFESSION',
        'מספר עבירה': 'OFFENCE_NUMBER',
        'סוג עבירה': 'OFFENCE_TYPE',
        'אופן קבלת הנשק': 'OBTAIN_WAY_WEP',
        'סטטוס הנשק': 'STATUS_WEP',
        'חרטה': 'REGRET',
        'אופן החזקת הנשק': 'HELD_WAY_WEP',
        'תכנון': 'PLANNING',
        'מטרה-סיבת העבירה': 'PURPOSE',
        'שימוש': 'USE',
        'סוג הנשק [אקדח]': 'TYPE_WEP',
        'מתחם ענישה - שופט': 'PUNISHMENT_RANGE'
    }

    # Initialize an empty DataFrame with the translated columns
    gt_table = pd.DataFrame(columns=columns_translation.values())
    
    db_path = params['db_path']

    # Iterate through each folder in db_path
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder)
        if os.path.isdir(folder_path):
            # Iterate through each file in the folder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.startswith('qa_features'):
                    file_content = pd.read_csv(file_path)
                    grouped_df = file_content.groupby('feature_key')['extraction'].apply(
                        lambda x: ', '.join(
                            sorted(set(
                                item.strip("'[] ")  # Clean brackets and quotes
                                for sublist in x if isinstance(sublist, str)  # Ensure it's a string
                                for item in sublist.strip('[]').split(',')  # Split list-like strings
                                if item.strip() not in [None, '', '[]']  # Filter invalid values
                            ))
                        )
                    ).reset_index()
                    pivoted_df = grouped_df.set_index('feature_key').T
                    pivoted_df.columns.name = None  # Remove column name
                    pivoted_df.reset_index(drop=True, inplace=True)

                    # Update CONFESSION column
                    if 'CONFESSION' in pivoted_df.columns:
                        if pivoted_df['CONFESSION'].str.contains('כן').any():
                            pivoted_df['CONFESSION'] = 'כן'
                        else:
                            pivoted_df['CONFESSION'] = 'לא'
                    if 'TYPE_WEP' in pivoted_df.columns:
                        result = []
                        weapons = [
                            "אקדח", "תת מקלע", "תת מקלע מאולתר", "בקבוק תבערה", "מטען חבלה", "רימון רסס", 
                            "רובה סער", "רימון הלם/גז", "טיל לאו", "טיל מטאדור", "רובה צייד", "רובה צלפים", 
                            "מטען חבלה מאולתר", "רובה סער מאולתר"
                        ]
                        wep = pivoted_df['TYPE_WEP'].values[0]
                        for weapon in weapons:
                            if weapon in wep:
                                result.append(weapon)
                        #delete תת מקלע if תת מקלע מאולתר is in the list
                        if 'תת מקלע מאולתר' in result:
                            result = [i for i in result if i != 'תת מקלע']
                        #delete רובה סער if רובה סער מאולתר is in the list
                        if 'רובה סער מאולתר' in result:
                            result = [i for i in result if i != 'רובה סער']
                        #delete מטען חבלה if מטען חבלה מאולתר is in the list
                        if 'מטען חבלה מאולתר' in result:
                            result = [i for i in result if i != 'מטען חבלה']
                        pivoted_df['TYPE_WEP'] = ', '.join(result)

                    # Align and append data to the cleared `gt_table`
                    for col in pivoted_df.columns:
                        if col not in gt_table.columns:
                            gt_table[col] = None

                    # Extract metadata for the case
                    meta_data = pd.read_csv(params['metadata_path'])
                    case_number = folder.split('-')
                    if case_number[0] in ['ME', 'SH']:
                        case_number = f"{case_number[3]}-{case_number[2]}-{case_number[1]}"
                    else:
                        case_number = folder

                    if case_number in meta_data['Case Number'].values:
                        case_row = meta_data.loc[meta_data['Case Number'] == case_number]
                        extracted_values = {
                            "DATE": case_row['Date'].values[0],
                            "JUDGES": case_row['Judges'].values[0],
                            "PROSECUTORS": case_row['Prosecutores'].values[0],
                            "DEFENDANTS": case_row['Defendants'].values[0],
                            "COURT": case_row['Court'].values[0],
                            "LOCATION": case_row['Location'].values[0],
                            "DEFENDANTS_RACE": case_row['Defendants_Race'].values[0],
                            "JUDGE_RACE": case_row['Judges_Race'].values[0]

                        }
                    else:
                        extracted_values = {key: None for key in ["DATE", "JUDGES", "PROSECUTORS", "DEFENDANTS", "COURT", "LOCATION", "Defendants_Race", "DEFENDANTS_RACE"]}

                    for key, value in extracted_values.items():
                        if key not in gt_table.columns:
                            gt_table[key] = None
                        pivoted_df[key] = value

                    # Extract the last appearance of `punishment` under `feature_key` with valid content
                    if 'PUNISHMENT' in file_content['feature_key'].values:
                        valid_punishments = file_content.loc[
                            (file_content['feature_key'] == 'PUNISHMENT') & (file_content['extraction'] != '[]'),
                            'extraction'
                        ].dropna()
                        last_punishment = valid_punishments.iloc[-1] if not valid_punishments.empty else None
                    else:
                        last_punishment = None
                    if last_punishment:
                        sentence_punishment_range = replace_words_with_numbers(last_punishment, number_dict)

                        if sentence_punishment_range:
                            punishment_range = punishment_range_extract(sentence_punishment_range)
                        else:
                            punishment_range = 'None'
                        # Add `punishment` to the pivoted DataFrame
                        if punishment_range is not None:
                            pivoted_df['PUNISHMENT_RANGE'] = [punishment_range['lower'] + ' - ' + punishment_range['top']+ ' ' + 'חודשי מאסר']
                    else:
                        pivoted_df['PUNISHMENT_RANGE'] = None 

                    # Extract gender of the judge
                    file_pre_path = os.path.join(folder_path, 'preprocessing.csv')
                    if os.path.exists(file_pre_path):
                        file_pre = pd.read_csv(file_pre_path)
                        if 'מין השופט/ת' in file_pre['part'].values:
                            case_row = file_pre.loc[file_pre['part'] == 'מין השופט/ת']
                            judge_gender = case_row['text'].values[0]
                            pivoted_df['JUDGE_GENDER'] = judge_gender
                        else:
                            pivoted_df['JUDGE_GENDER'] = None
                    else:
                        pivoted_df['JUDGE_GENDER'] = None

                    # Extract gender of the defendant
                    if 'מין הנאשם/ת' in file_pre['part'].values:
                        case_row = file_pre.loc[file_pre['part'] == 'מין הנאשם/ת']
                        defendant_gender = case_row['text'].values[0]
                        pivoted_df['DEFENDANT_GENDER'] = defendant_gender
                    else:
                        pivoted_df['DEFENDANT_GENDER'] = None

                    # Extract punishment
                    file_punishment_path = os.path.join(folder_path, 'qa_features_dictalm2.0-instruct_06_12.csv')
                    if os.path.exists(file_punishment_path):
                        file_punishment_content = pd.read_csv(file_punishment_path)
                        for i in range(3):
                            column = ''
                            if i ==0:
                                column = "ACTUAL PUNISHMENT"
                            elif i ==1:
                                column = "PROBATION"
                            elif i ==2:
                                column = "FINE"
                            if f'PUNISHMENT_{i}' in file_punishment_content['feature_key'].values:
                                punishment = file_punishment_content.loc[
                                    (file_punishment_content['feature_key'] == f'PUNISHMENT_{i}') & (file_punishment_content['extraction'] != '[]'),
                                    'extraction'
                                ].dropna()


                                if i == 0:
                                    punishment = punishment.iloc[-1] if not punishment.empty else None
                                    punishment = ' '.join(re.findall(r'\b[\w]+\b', punishment))

                                    if punishment is not None:
                                        # sentence_punishment = replace_words_with_numbers(punishment, number_dict)
                                        punishment = punishment_extract(punishment)
                                        if len(punishment) > 0:
                                            pivoted_df[column] = [punishment]
                                elif i == 1:
                                    punishment = " ".join(punishment.astype(str))
                                    punishment = ' '.join(re.findall(r'\b[\w]+\b', punishment))

                                    #make it to list

                                    punishment = punishment_extract_probation(punishment)
                                    if len(punishment) > 0:
                                        pivoted_df[column] = [punishment] 
                                elif i == 2:
                                    punishment = " ".join(punishment.astype(str))
                                    punishment = ' '.join(re.findall(r'\b\d{1,8}(?:,\d{3})*(?:\.\d+)?\b', punishment))

                                    punishment = fine_extract(punishment)
                                    if len(punishment) > 0:
                                        pivoted_df[column] = [punishment] 
                            else:
                                pivoted_df[column] = None

                    # Extract general circumstances
                    file_circum_path = os.path.join(folder_path, 'qa_features_dictalm2.0-instruct_06_12.csv')
                    if os.path.exists(file_circum_path):
                        file_circum_content = pd.read_csv(file_circum_path)
                        result_numbers = {}
                        
                        for _, row in file_circum_content.iterrows():
                            if row['feature_key'] == 'GENERAL_CIRCUM':
                                circum = row['extraction']
                                
                                # Convert the string representation of the list to an actual list
                                circum_list = ast.literal_eval(circum)
                                
                                # Process each item in the list
                                for item in circum_list:
                                    key, value = item.split(' : ')
                                    key = key.strip()
                                    value = value.strip()
                                    
                                    # Update the result dictionary
                                    if key not in result_numbers:
                                        result_numbers[key] = value
                                    elif value == 'כן':  # Prioritize "כן" if conflicting values exist
                                        result_numbers[key] = 'כן'
                        
                        # Get all keys with the value "כן"
                        keys_with_yes = sorted(set([key for key, value in result_numbers.items() if value == 'כן']))
                        
                        # Add to pivoted_df as a comma-separated string for better readability
                        pivoted_df['GENERAL_CIRCUM'] = [', '.join(keys_with_yes)] * len(pivoted_df)
                    else:
                        pivoted_df['GENERAL_CIRCUM'] = None

                    # extract Regret, if there is one כן 
                    file_regret_path = os.path.join(folder_path, 'qa_features_dictalm2.0-instruct_06_12.csv')
                    if os.path.exists(file_regret_path):
                        file_regret_content = pd.read_csv(file_regret_path)
                        # there is some rows of regret, if one of them is כן then the regret is כן
                        regret = 'לא'
                        for _, row in file_regret_content.iterrows():
                            if row['feature_key'] == 'REGRET':
                                regret_ = row['extraction']
                                #find כן using regex
                                if 'כן' in regret_:
                                    regret = 'כן'
                                    break
                        pivoted_df['REGRET'] = regret
                    else:
                        pivoted_df['REGRET'] = None

                    # add case number
                    pivoted_df['CASE_NUMBER'] = folder

                    # Ensure all columns exist in gt_table before appending
                    for col in pivoted_df.columns:
                        if col not in gt_table.columns:
                            gt_table[col] = None

                    # Append pivoted_df to gt_table
                    gt_table = pd.concat([gt_table, pivoted_df], ignore_index=True)

                    logger.info(f"Finished processing file in folder {folder}")

    # Save the final table
    output_path = 'processed_file.csv'
    #remove from gt_table the PUNISHMENT column
    gt_table = gt_table.drop(columns=['PUNISHMENT'])
    # Define the desired column order
    desired_order = [
        "CASE_NUMBER", "DATE", "COURT", "LOCATION", "OFFENCE_NUMBER", "OFFENCE_TYPE",
        "JUDGES", "PROSECUTORS", "DEFENDANTS", "DEFENDANT_GENDER", "JUDGE_GENDER", "JUDGE_RACE", "DEFENDANTS_RACE",
        "CONFESSION",  "TYPE_WEP", "OBTAIN_WAY_WEP", "STATUS_WEP",
        "REGRET", "HELD_WAY_WEP", "PLANNING", "PURPOSE", "USE",
        "PUNISHMENT_RANGE", "ACTUAL PUNISHMENT", "PROBATION", "FINE",
        "GENERAL_CIRCUM"
    ]
    reordered_data = gt_table[desired_order]
    reordered_data.to_csv(output_path, index=False)

def add_metadata(params,final_df):     
    # final_df = pd.read_csv(os.path.join(params['output_path'], 'features_extraction_cases_2.csv'))
    # metadata = pd.read_csv('/home/tak/MOJ/resources/data/tagging/drugs/gt/one_defendant_drugs.csv')
    metadata = pd.read_csv(params['metadata_path'])
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
    # final_path = os.path.join(params['output_path'], 'features_extraction_cases_3.csv')
    # final_df.to_csv(final_path, index=False, encoding='utf-8')
    return final_df

def punishment_foramt(params,df_regular):
    # df_regular = pd.read_csv(os.path.join(params['output_path'], 'features_extraction_cases_3.csv'))
    df_pun = pd.read_excel('/home/tak/MOJ/src/flows/final_gpt_drugs_docx.xlsx')

    # הוספת עמודה אם לא קיימת
    if 'PUNISHMENT_Range' not in df_regular.columns:
        df_regular['PUNISHMENT_Range'] = None

    for index, row in df_regular.iterrows():
        path = f'/home/tak/MOJ/results/db/drugs_docx/{row['verdict']}/punishment_range_gpt.json'

        #check if the file exists
        if not os.path.exists(path):
            df_regular.at[index, 'PUNISHMENT_Range'] = '-'
            continue
        with open(path, 'r') as f:
            data = json.load(f)
            try:
                d = data[-1]
                #remove last .
                numbers = re.findall(r'\d+', d)
                result_str  = f'{numbers[0]} - {numbers[1]} חודשים'
                df_regular.at[index, 'PUNISHMENT_Range'] = result_str 
            except:
                df_regular.at[index, 'PUNISHMENT_Range'] = ''
    # שמירה

    max_punishment = []
    min_punishment = []
    range_punishment = []
    for index, row in df_regular.iterrows():
        punishment_range = row['PUNISHMENT_Range']
        if isinstance(punishment_range, str):
            punishment_range = punishment_range.split(' ')
            if len(punishment_range) == 4:
                min_punishment.append(punishment_range[0])
                max_punishment.append(punishment_range[2])
                range = int(punishment_range[2]) - int(punishment_range[0]) 
                range_punishment.append(range)
            elif len(punishment_range) == 1:
                min_punishment.append("")
                max_punishment.append("")
                # range = int(punishment_range[3]) - int(punishment_range[0]) 
                range_punishment.append("")
            else:   
                min_punishment.append(" not found")
                max_punishment.append("Not found")
                # range = int(punishment_range[3]) - int(punishment_range[0]) 
                range_punishment.append("not found")
        else:
            min_punishment.append('')
            max_punishment.append('')
            range_punishment.append('')
    # Add the new columns to the DataFrame
    df_regular['min_punishment'] = min_punishment
    df_regular['max_punishment'] = max_punishment
    df_regular['range_punishment'] = range_punishment
    # Save the updated DataFrame to a new CSV file
    return df_regular
    # final_path = os.path.join(params['output_path'], 'features_extraction_cases_6.csv')
    # df_regular.to_csv(final_path, index=False, encoding='utf-8')
        
def add_punishment_range(params, df):
    # df = pd.read_csv('/home/tak/MOJ/results/evaluations/drugs/features_extraction/features_extraction_cases_5.csv')
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
    # final_path = os.path.join(params['output_path'], 'features_extraction_cases_6.csv')
    # df.to_csv('/home/tak/MOJ/results/evaluations/drugs/features_extraction/features_extraction_cases_6.csv', index=False, encoding='utf-8')
    return df

def add_OFFENCE_NUMBER(params,df):
    # df = pd.read_csv('/home/tak/MOJ/results/evaluations/drugs/features_extraction/features_extraction_cases_6.csv')
    all_offenses = []
    verdict_value = df['verdict'].values
    path = '/home/tak/MOJ/results/db/drugs_docx'
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
    # final_path = os.path.join(params['output_path'], 'features_extraction_cases_7.csv')
    # df.to_csv(final_path, index=False, encoding='utf-8')
    return df


def add_age(params, df):
    # df = pd.read_csv('/home/tak/MOJ/results/evaluations/drugs/features_extraction/features_extraction_cases_6.csv')
    verdict_value = df['verdict'].values
    path = '/home/tak/MOJ/results/db/drugs_docx'
    index = 0
    for t in verdict_value:
        for file in os.listdir(os.path.join(path,t)):
            # print(file)
            if file.startswith('pre'):
                preprocessed_data = pd.read_csv(os.path.join(path, t, file))
                #for each row in the preprocessed data
                for idx, row in preprocessed_data.iterrows():
                    text = row['text']
                    pattern = r"בן\s(\d+)"
                    matches = re.findall(pattern, text)
                    if len(matches) > 0:
                        #update the age in the df
                        df.at[index, 'age'] = matches[0]
                    else:
                        pattern = r"בת\s(\d+)"
                        matches = re.findall(pattern, text)
                        if len(matches) > 0:
                            #update the age in the df
                            df.at[index, 'age'] = matches[0]
                        else:
                            pattern = r"יליד\s(\d+)"
                            matches = re.findall(pattern, text)
                            if len(matches) > 0:
                                df.at[index, 'age'] = matches[0]
                            else:
                                pattern = r"ילידת\s(\d+)"
                                matches = re.findall(pattern, text)
                                if len(matches) > 0:

                                    df.at[index, 'age'] = matches[0]
                    pattern = r"נשוי"
                    matches = re.findall(pattern, text)
                    if len(matches) > 0:
                        #update the age in the df
                        df.at[index, 'status'] = "נשוי"
                    else:
                        pattern = r"נשואה"
                        matches = re.findall(pattern, text)
                        if len(matches) > 0:
                            #update the age in the df
                            df.at[index, 'status'] = "נשואה"
                        else:
                            pattern = r"רווק"
                            matches = re.findall(pattern, text)
                            if len(matches) > 0:
                                #update the age in the df
                                df.at[index, 'status'] = "רווק"
                            else:
                                pattern = r"רווקה"
                                matches = re.findall(pattern, text)
                                if len(matches) > 0:
                                    #update the age in the df
                                    df.at[index, 'status'] = "רווק"

                    pattern = r"נאשם"
                    matches = re.findall(pattern, text)
                    if len(matches) > 0:
                        #update the age in the df
                        df.at[index, 'sex'] = "בן"
                    else:
                        pattern = r"נאשמת"
                        matches = re.findall(pattern, text)
                        if len(matches) > 0:
                            #update the age in the df
                            df.at[index, 'sex'] = "בת"
        #update the index
        index += 1
                                
    return df

if __name__ == '__main__':
    # Load main configuration parameters
    main_params = config_parser("", "main_config")
    # Extract the domain from the main configuration
    domain = main_params["domain"]
    
    # Parse the specific feature extraction parameters for the given domain
    params = config_parser("add_features", domain)
    
    # Run the main function with the parsed parameters
    df = aggregate_data(params, domain)
    df = add_metadata(params, df)

    df = punishment_foramt(params,df)
    # df = add_punishment_range(params, df)
    df = add_OFFENCE_NUMBER(params, df)

    df = add_age(params, df)

    # Save the final DataFrame to a CSV file
    final_path = os.path.join(params['output_path'], 'final_table.csv')
    df.to_csv(final_path, index=False, encoding='utf-8')