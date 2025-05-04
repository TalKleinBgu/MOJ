import os
import sys
import logging
import glob
import pandas as pd
from setfit import SetFitModel, SetFitTrainer
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime
import numpy as np
import shutil
import ast
import re

number_dict = {
    "אחד": 1,
    "אחת": 1,
    "שנה": 1,
    "לשנה": 1,
    "שנים": 2,
    "שתים": 2,
    "שנתיים":2,
    "לשנתיים":2,
    "שתי": 2,
    "שני": 2,
    "שלוש": 3,
    "שלושה": 3,
    "ארבע": 4,
    "ארבעה": 4,
    "חמש": 5,
    "חמישה": 5,
    "שש": 6,
    "שישה": 6,
    "ששה": 6,
    "שבעה": 7,
    "שבע": 7,
    "שמונה": 8,
    "תשע": 9,
    "תשעה": 9,
    "עשר": 10,
    "עשרה": 10,
    "אחד עשרה": 11,
    "אחת עשרה": 11,
    "שנים עשר": 12,
    "שתים עשרה": 12,
    "שלשה עשר": 13,
    "שלש עשרה": 13,
    "ארבע עשרה": 14,
    "ארבעה עשר": 14,
    "שבע עשרה": 17,
    "שבעה עשר": 17,
    "שמונה עשר": 18,
    "שמונה עשר": 18,
    "תשעה עשר": 19,
    "תשע עשרה": 19,
    "עשרים": 20
}
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)


from utils.files import config_parser, setup_logger, write_yaml
from scripts.preprocess.range_punishment_extractor import Punishment_range_extractor
from utils.punishment_extractor import replace_words_with_numbers, punishment_range_extract, punishment_extract_probation, punishment_extract, fine_extract

                    
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
                    if folder == 'ME-18-04-34027-630':
                        print()
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
                    meta_data = pd.read_csv(params['meta_data_path'])
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

def new_df():
    df = pd.read_csv('processed_file.csv')
    new_df = []
    list_of_cases = ['ME-13-05-31862-33','ME-16-03-41585-22', 'ME-20-01-17935-7', 'SH-16-12-32560-784', 'ME-16-01-189-123', 'ME-16-01-31498-921', 'ME-17-02-3454-397', 'ME-18-05-32659-754', 'ME-14-07-28889-461', 'SH-16-05-42991-439']
    # add specific rows based on the CASE NUMBER column
    for case in list_of_cases:
        new_df.append(df.loc[df['CASE_NUMBER'] == case])
    
    new_df = pd.concat(new_df)
    new_df.to_csv('sample.csv', encoding='utf-8-sig', index=False)

def main(params):
    logger = setup_logger(save_path=os.path.join(params['result_path'], 'logs'),
                          file_name='feature_extraction_table')
    create(params, logger)
    # new_df()
    


if __name__ == '__main__':
    params = config_parser('main_config')
    main(params)