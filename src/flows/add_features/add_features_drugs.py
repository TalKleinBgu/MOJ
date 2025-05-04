import os
import sys
import ast
import pandas as pd
import json
import re
from collections import defaultdict
from collections import Counter
import importlib
import openai

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir,
                                                    '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import config_parser, get_date, setup_logger
from utils.evaluation.weapon.evaluation import change_case_name_foramt, change_format, convert_tagging_to_pred_foramt
from utils.punishment_extractor import case_mapping_dict
from utils.features import *





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

def updates_amount_type(df):
    # Load the CSV file into a DataFrame
    # df = pd.read_csv(os.path.join(params['output_path'], 'features_extraction_cases.csv'))
    column_names = ['CIR_AMOUNT','CIR_TYPE']
    dirs_path = params['prediction_dfs']
    aggregated_rows = []
    all_types = []
    # metadata = pd.read_csv(params['metadata_path'])
    for dir in os.listdir(dirs_path):
        if 'ME' in dir or 'SH' in dir:
            pred_path = os.path.join(dirs_path, dir)
            aggregated_row = {}
            amount_type = ''

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
                        elif (type(row['CIR_AMOUNT']) == float and type(row['CIR_TYPE']) == float):
                            aggregated_row['CIR_AMOUNT'] = ''
                            aggregated_row['CIR_TYPE'] = ''

                            #delete the row
                            fe_df.drop(i, inplace=True)
                        else:
                            cir_type = row['CIR_TYPE']
                            lst = ['amb-fub', 'GHB', 'החומר', 'קאטמין', 'מטילמטאקטינון', 'מסקלין', 'AMB-FUM', 'P-METHYLMETHCATHINONE', 'CHIMINACA-AB', 'ALFA-PVP', 'AB-FUBINACA', 'בופרנורין', 'נייס', 'Methy', 'קנאביס', 'METHAMPHETAMINE', 'קנביס/חשיש', 'מתאמפטמין', 'ADB-5F', 'MDMB-Butinaca-4F', 'קוקאין', 'הידרו', '5F-AMB', '5F', 'אמפטמין', 'Dimethlmethcathinone', 'מטדון', 'UR144', 'דאב', 'Phenethylamines', 'm-methylmethcathinone', 'Pentedrone', '22-PB-F5', 'DIMETHOXYPPHENETHYLAMINE', 'methyimthcathinonc', 'AMB-FUB', 'm-', 'FENTANYLUM', 'דוסה', 'Alfa-PVP', 'קנאבוס', 'F5-ADB', 'N-Ethylpentylone', 'קודאין', 'קאתינון', 'MDMA', 'קנביס', 'BUTANEDIOL', 'FUB-AMB', 'AB-CHIMINACA', 'AMB-F5', 'בופרנורפין', 'METHAMPHETAMIN', 'm-METHYLMETHCATHINONE', 'AB-CHMINACA', 'ketamine', 'מתקאתינון', 'AM2201', 'MDMB-4en-PINACA', 'MMC', 'מתאמפטאמין', 'JWH-210', 'דיאםטי', 'דימתילטריפטאמין', 'מתיאמפטמין', 'מת\\\\אמפטמין', 'אלפא', 'MDMB-4EN-PINACA', 'MethylMethCathinone-m', 'MDPV', 'AMB-5F', 'הרואין', 'MDMB-4en', 'ALFA-PVT', 'ADB', 'OXYCODONE', 'פסילוצין', 'ADB-F5', 'MDMB', 'Fentanyl', '22-PB', 'בונורפין', 'קססונית', 'קטאמין', 'מתילמטקתינון', 'DMT', 'קטמין', 'קנבוס', 'KETAMIN', 'KETAMINE', 'אינדול', 'f4-mdmb-butinaca', 'M-METHYLMETHCATHINONE', 'GBL', 'BUTIMNACA', 'F4-', 'methamphetamine', 'AM-2201', '-MMC', 'קנבינואידים', 'Methylmethcathinone', 'TOC', 'UR-144', 'קנבואידים', 'BDO', 'METHYLMETHCATHINONE', 'דלתא', 'INDAZOL', 'PB-22', 'בופרנופין', 'מסקאלין', '5F-ADB', 'קקטוס', 'AB', 'מתילמטאקתינון', 'עלי', 'חשיש', 'LSD', 'Dimethylmethcathinone', 'כימיקלים']
                            #iterate over cir_type
                            # cir_type = cir_type.split(',')
                            def parse_space_separated_lists_no_eval(s):
                                # Find all `[ '...' ]` blocks
                                blocks = re.findall(r"\[ *'(.*?)' *\]", s)

                                # Wrap each captured string as a one-element list
                                return [item for item in blocks]
                            cir_type = parse_space_separated_lists_no_eval(cir_type)
                            the_type = ''
                            for cir in cir_type:
                                for i in lst:
                                    if i in cir:
                                        # all_types.append(i)
                                        the_type = i
                                        break
                                if the_type:
                                    break


                                # aggregated_row['amount_type'] = all_types

                            ########## CIR_AMOUNT ##########
                            cir_amount = row['CIR_AMOUNT']
                            if isinstance(cir_amount, float):
                                #update df to no
                                all_type = ''
                            else:
                                pattern = r'\[[^\[\]]+\]'
                                matches = re.findall(pattern, cir_amount)
                                all_amounts = []
                                if len(matches) > 0:
                                    for match in matches:
                                        match = match.replace('[', '').replace(']', '')
                                        text = match.split(' - ')
                                        try:
                                            all_amounts.append(f'{text[0]} {text[1]}')
                                        except:
                                            all_amounts.append(text[0])
                                    # print(row)
                                #update df to no
                                kg = 0
                                gram = 0
                                other = []

                                for amou in all_amounts:
                                    amount_val = amou.split(' ')[0].strip().replace('"', '').replace("'", '')
                                    try:
                                        num = float(amount_val)
                                        if 'קילוגרם' in amou or 'ק\"ג' in amou or 'ק"ג' in amou or "ק\'ג" in amou:
                                            kg += num
                                        elif 'גרם' in amou:
                                            gram += num
                                        else:
                                            other.append(amou)
                                    except:
                                        other.append(amou)
                                total = (kg * 1000) + gram
                                # total = kg + (gram / 1000)
                                # Convert to string with 2 decimal places
                                total = f'{total:.2f} קילוגרם'
                                amount_type = f'{the_type} {total}'
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
            all_types.append(amount_type)
    # Combine all aggregated rows into a single DataFrame
    final_df = pd.DataFrame(aggregated_rows)
    # final_path = os.path.join(params['output_path'], 'features_extraction_cases.csv')
    column_names = ['CIR_AMOUNT','CIR_TYPE']
    #save the columns to the df variable
    df[column_names] = final_df[column_names]
    df['amount_type'] = all_types
    return df
    #save the df to the csv file
    # df.to_csv(final_path, index=False, encoding='utf-8')

    #DELETE CIR_PUNISHMENT column


def change_formats(params,final_df):
    # final_df = pd.read_csv(os.path.join(params['output_path'], 'features_extraction_cases.csv'))
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

        # ######### PUNISHMENT_Range ##########
        # punishment_range = row['PUNISHMENT_Range']
        # if isinstance(punishment_range, float):
        #     #update df to no
        #     final_df.at[index, 'PUNISHMENT_Range'] = ''
        # else:
        #     punishment_range = ast.literal_eval(punishment_range)
        #     if isinstance(punishment_range, dict):
        #         try:
        #             minimun = punishment_range.get('lower', '')
        #             maximum = punishment_range.get('top', '')
        #             punishment_range = f"{minimun} - {maximum}"
        #             final_df.at[index, 'PUNISHMENT_Range'] = punishment_range
        #         except Exception as e:
        #             print(f"Error processing punishment_range: {e}")
        #             final_df.at[index, 'PUNISHMENT_Range'] = ''
        #     else:
        #         # If it's not a dictionary, just keep it as is
        #         final_df.at[index, 'PUNISHMENT_Range'] = ''

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
        numbers = []
        actual = row['PUNISHMENT_ACTUAL']
        if isinstance(actual, float):
            #update df to no
            final_df.at[index, 'PUNISHMENT_ACTUAL'] = ''
        else:
            actuals = actual.replace('\'', '').replace('[', '').replace(']', '').split(', ')
            for pun in actuals:
                pn = pun.split(' ')
                for i in range(len(pn)):
                    try:
                        num = float(pn[i])
                        # print(row['verdict'])
                        if 'חודשי' in pn[i + 1]:
                            # If the next word is 'חודשי', add it to the list
                            numbers.append(num)
                            break
                        else:
                            if 'שנות' in pn[i + 1] or 'שנה' in pn[i + 1] or 'שנים' in pn[i + 1]:
                                # If the next word is 'שנות', add it to the list
                                numbers.append(num * 12)
                                break
                    except:
                        # If the next word is not 'חודשי', just add the number
                        continue
            # actuals = actual.split('] [')
            # numbers = []
            # actual = actuals[-1]
            # if actual == ']':
            #     if len(actuals) > 1:
            #         actual = actuals[-2]
            #     else:
            #         actual = ''
            # all_digits = ''
            # for char in actual:
            #     if char.isdigit():
            #         all_digits += char
            # if all_digits != '':
            #     numbers.append(all_digits)
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
        all_types = []
        lst = ['amb-fub', 'GHB', 'החומר', 'קאטמין', 'מטילמטאקטינון', 'מסקלין', 'AMB-FUM', 'P-METHYLMETHCATHINONE', 'CHIMINACA-AB', 'ALFA-PVP', 'AB-FUBINACA', 'בופרנורין', 'נייס', 'Methy', 'קנאביס', 'METHAMPHETAMINE', 'קנביס/חשיש', 'מתאמפטמין', 'ADB-5F', 'MDMB-Butinaca-4F', 'קוקאין', 'הידרו', '5F-AMB', '5F', 'אמפטמין', 'Dimethlmethcathinone', 'מטדון', 'UR144', 'דאב', 'Phenethylamines', 'm-methylmethcathinone', 'Pentedrone', '22-PB-F5', 'DIMETHOXYPPHENETHYLAMINE', 'methyimthcathinonc', 'AMB-FUB', 'm-', 'FENTANYLUM', 'דוסה', 'Alfa-PVP', 'קנאבוס', 'F5-ADB', 'N-Ethylpentylone', 'קודאין', 'קאתינון', 'MDMA', 'קנביס', 'BUTANEDIOL', 'FUB-AMB', 'AB-CHIMINACA', 'AMB-F5', 'בופרנורפין', 'METHAMPHETAMIN', 'm-METHYLMETHCATHINONE', 'AB-CHMINACA', 'ketamine', 'מתקאתינון', 'AM2201', 'MDMB-4en-PINACA', 'MMC', 'מתאמפטאמין', 'JWH-210', 'דיאםטי', 'דימתילטריפטאמין', 'מתיאמפטמין', 'מת\\\\אמפטמין', 'אלפא', 'MDMB-4EN-PINACA', 'MethylMethCathinone-m', 'MDPV', 'AMB-5F', 'הרואין', 'MDMB-4en', 'ALFA-PVT', 'ADB', 'OXYCODONE', 'פסילוצין', 'ADB-F5', 'MDMB', 'Fentanyl', '22-PB', 'בונורפין', 'קססונית', 'קטאמין', 'מתילמטקתינון', 'DMT', 'קטמין', 'קנבוס', 'KETAMIN', 'KETAMINE', 'אינדול', 'f4-mdmb-butinaca', 'M-METHYLMETHCATHINONE', 'GBL', 'BUTIMNACA', 'F4-', 'methamphetamine', 'AM-2201', '-MMC', 'קנבינואידים', 'Methylmethcathinone', 'TOC', 'UR-144', 'קנבואידים', 'BDO', 'METHYLMETHCATHINONE', 'דלתא', 'INDAZOL', 'PB-22', 'בופרנופין', 'מסקאלין', '5F-ADB', 'קקטוס', 'AB', 'מתילמטאקתינון', 'עלי', 'חשיש', 'LSD', 'Dimethylmethcathinone', 'כימיקלים']
        #iterate over cir_type
        # cir_type = cir_type.split(',')
        def parse_space_separated_lists_no_eval(s):
            # Find all `[ '...' ]` blocks
            blocks = re.findall(r"\[ *'(.*?)' *\]", s)

            # Wrap each captured string as a one-element list
            return [item for item in blocks]
        cir_type = parse_space_separated_lists_no_eval(cir_type)

        for cir in cir_type:
            for i in lst:
                if i in cir:
                    all_types.append(i)
                    break


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
                    try:
                        all_amounts.append(f'{text[0]} {text[1]}')
                    except:
                        all_amounts.append(text[0])
                # print(row)
            #update df to no
            kg = 0
            gram = 0
            other = []

            for amou in all_amounts:
                amount_val = amou.split(' ')[0].strip().replace('"', '').replace("'", '')
                try:
                    num = float(amount_val)
                    if 'קילוגרם' in amou or 'ק"ג' in amou:
                        kg += num
                    elif 'גרם' in amou:
                        gram += num
                    else:
                        other.append(amou)
                except:
                    other.append(amou)

            total = kg + (gram / 1000)
            # Convert to string with 2 decimal places
            total = f'{total:.2f}'
            total_str = f'{total} קילוגרם'
            other.append(total_str)
            final_df.at[index, 'CIR_AMOUNT'] = all_amounts

    #save the df to the csv file
    # final_path = os.path.join(params['output_path'], 'features_extraction_cases_2.csv')
    #delet CIRCUM_OFFENSE column
    if 'CIRCUM_OFFENSE' in final_df.columns:
        del final_df['CIRCUM_OFFENSE']
    # final_df.to_csv(final_path, index=False, encoding='utf-8')
    return final_df

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
    # df_pun = pd.read_excel('/home/tak/MOJ/src/flows/final_gpt_drugs_docx.xlsx')

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

def delete_cases(params, dt):

    # Create dict for verdict → indices
    verdict_dict = {}
    for idx, row in dt.iterrows():
        verdict = row['verdict']
        verdict_dict.setdefault(verdict, []).append(idx)

    dt = dt[dt['verdict'].notna()]

    # Values and paths
    verdict_value = dt['verdict'].values
    path = '/home/tak/MOJ/results/db/drugs_docx'
    #delete rows if in verdict column there is nan
    # Loop through all verdicts
    counti = 0
    ver_list = []
    all_ver_list = []
    all = 0
    for verdict in verdict_value:
        verdict_path = os.path.join(path, verdict)
        if not os.path.isdir(verdict_path):
            continue
        for file in os.listdir(verdict_path):
            if file.startswith('pre'):
                preprocessed_data = pd.read_csv(os.path.join(verdict_path, file))
                for _, row in preprocessed_data.iterrows():
                    text = row['text']
                    if 'פקודת הסמים המסוכנים' in text:
                        counti += 1
                        ver_list.append(verdict)
                        break
                all += 1
                all_ver_list.append(verdict)




    #show distinct verdicts betwen the two lists
    ver_list = list(set(ver_list))
    all_ver_list = list(set(all_ver_list))
    diffrent = list(set(all_ver_list) - set(ver_list))

    #delete the rows from the df
# Collect all indices to drop
    indices_to_drop = []
    for verdict in diffrent:
        if verdict in verdict_dict:
            indices_to_drop.extend(verdict_dict[verdict])

    # Drop once, then reset index
    dt.drop(index=indices_to_drop, inplace=True)
    dt.reset_index(drop=True, inplace=True)
    print(len(dt))
    print(dt['CIR_TYPE'].iloc[0])
    print(type(dt['CIR_TYPE'].iloc[0]))
    # Convert CIR_TYPE from string to list (only if needed)
    dt['CIR_TYPE'] = dt['CIR_TYPE'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    # Now safely remove rows with empty lists in CIR_TYPE
    dt = dt[~dt['CIR_TYPE'].apply(lambda x: isinstance(x, list) and len(x) == 0)].copy()    
    print(len(dt))

    dt = dt[dt['PUNISHMENT_Range'].notna()].copy()

    print(len(dt))
    # Save the updated DataFrame
    # dt.to_csv('/home/tak/MOJ/results/evaluations/drugs/features_extraction/final_table_3.csv', index=False, encoding='utf-8')
    return dt

def amount_type_gpt(df):
    verdicts = df['verdict'].values
    path = '/home/tak/MOJ/results/db/drugs_docx'
    list_to_drop = []
    for index, row in df.iterrows():
        if isinstance(row['CIR_TYPE'], float):
            #update df to no
            list_to_drop.append(index)
        elif isinstance(row['PUNISHMENT_Range'], float):
            #update df to no
            list_to_drop.append(index)
    # drop the rows from the df
    df.drop(list_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    all_verdict_answers = []
    for verdict in verdicts:
        verdict_path = os.path.join(path, verdict)
        if not os.path.isdir(verdict_path):
            all_verdict_answers.append('')  # fallback if directory is missing
            continue
        answers = []

        for file in os.listdir(verdict_path):
            if file.startswith('feat'):
                preprocessed_data = pd.read_csv(os.path.join(verdict_path, file))
                # for each row in the preprocessed data
                for index, row in preprocessed_data.iterrows():
                    if isinstance(row['CIR_TYPE'], float) or isinstance(row['CIR_AMOUNT'], float):
                        continue  # Skip if either is missing
                    text = row['text']
                    message = f"""אני אתן לך משפט ותגיד לי מה הסם ומהי הכמות בפורמט :
                        [כמות-סם], לדוגמא [הרואין-0.5 גרם], [קוקאין -1.2 קילוגרם]
                        אם יש כמה סמים, תרשום אותם ביחד עם פסיק בינהם.
                        המשפט :
                    
                    {text}

                    אל תחזיר שום דבר אחר חוץ מהתשובה
                    """ 
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[{"role": "user", "content": message}]
                        )
                        answer = response.choices[0].message.content.strip()
                        answers.append(answer)
                    except Exception as e:
                        print(f"⚠️ Error processing: {verdict} | {e}")

                # save the answers to json file
                with open(os.path.join(verdict_path, 'amount_type_gpt.json'), 'w') as f:
                    json.dump(answers, f, ensure_ascii=False, indent=4)
                print(f"Saved answers for {verdict} to JSON file.")
                # save the answers of each file to csv file

                break  # Process only one file per verdict

        

        all_verdict_answers.append(", ".join(answers) if answers else '')

# Add the new column to the DataFrame
    df = df.iloc[:len(all_verdict_answers)].copy()
    df['amount_type'] = all_verdict_answers
    df.to_csv('/home/tak/MOJ/results/evaluations/drugs/features_extraction/final_table_4.csv', index=False, encoding='utf-8')

    return df


if __name__ == '__main__':
    # Load main configuration parameters
    main_params = config_parser("", "main_config")
    # Extract the domain from the main configuration
    domain = main_params["domain"]
    
    # Parse the specific feature extraction parameters for the given domain
    params = config_parser("add_features", domain)
    
    # Run the main function with the parsed parameters
    # df = aggregate_data(params, domain)
    # df = updates_amount_type(df)
    # df = punishment_foramt(params,df)
    # # df = add_punishment_range(params, df)
    # df = add_OFFENCE_NUMBER(params, df)
    # df = change_formats(params, df)
    # df = add_metadata(params, df)

    # df = add_age(params, df)
    path = '/home/tak/MOJ/results/evaluations/drugs/features_extraction/final_table_4.csv'
    df = pd.read_csv(path)
    print(len(df))
    # df = delete_cases(params, df)
    # df = amount_type_gpt(df)
    




    # Save the updated DataFrame to a new CSV file

    # Save the final DataFrame to a CSV file
    # final_path = os.path.join(params['output_path'], 'final_table.csv')
    # df.to_csv(final_path, index=False, encoding='utf-8')