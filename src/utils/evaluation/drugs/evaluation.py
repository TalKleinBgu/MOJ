import ast
from collections import Counter
import re
import pandas as pd
import numpy as np
import os
import sys  
column_to_drop = ['מספר תיק_ישן','חותמת זמן', 'מתייג ', 'עבירות נוספות',
                  'כסף ששולם', 'מכירה לסוכן', 'כמות תחמושת', 'תכנון', 
                  'נזק', 'בריחה מרדף', 'מתחם ענישה - מאשימה ', 'מתחם ענישה - שופט ', 'עונש',
                  'הערות ומחשבות', 'חרטה', 'מתחם ענישה - מאשימה', 'מתחם ענישה - שופט']


# column_to_drop = ['חותמת זמן', 'מתייג ', 'עבירות נוספות',
#                   'כסף ששולם','כמות תחמושת', 'תכנון', 
#                   'נזק', 'מתחם ענישה - מאשימה ', 'מתחם ענישה - שופט ', 'עונש',
#                   'הערות ומחשבות', 'חרטה']
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from resources.number_mapping import number_dict


doplicate = ['מספר תיק']
def replace_words_with_numbers(input_string,
                               number_dict):  # TODO fix it, doesnt work so well, Think about the right split!
    """
       replacing words with thier corresponding number, with the dict at the top of the file here
        """
    words = re.split(r'\s|,|\.|ל', input_string)
    # words = input_string.split()
    for i, word in enumerate(words):
        if word in number_dict:
            words[i] = str(number_dict[word])
    return ' '.join(words)

def convert_tagging_to_pred_foramt(df):
    dummy_columns = [col for col in df.columns if col.startswith('עבירות')]
    # remove עבירות נלוות from dummy columns
    dummy_columns = [col for col in dummy_columns if not col.startswith('עבירות נלוות') or col.startswith('עבירות סמים')]
    
    new_off_type_col = []
    amount_col = []
    type_drug_col = []
    drug_role_col = []
    lab_role_col = []
    actual_pun_col = []
    probation_pun_col = []
    fine_pun_col = []
    other_pun_col = []
    off_not_in_list_col = []
    type_amount_col = []
    for _, row in df.iterrows():
        new_off_type_case = []
        amount = []
        type_drug = []
        drug_role = []
        lab_role = []
        actual_pun = []
        probation_pun = []
        fine_pun = []
        other_pun = []
        off_not_in_list = []
        # handle offence
        for column_name in dummy_columns:
            if column_name.startswith('עבירות נלוות'):
                continue
            if row[column_name] != 0 and (not pd.isna(row[column_name])):
                    match = re.search(r"\[(.*?)\]", column_name)
                    if match:
                        captured_text = match.group(1)
                        text = ''
                        if ',' in row[column_name]:
                            text = row[column_name].split(',')
                            #remove empty spaces from list
                            text = [text_.strip() for text_ in text]
                        else:
                            text = [row[column_name]]
                        for text_ in text:
                            if type(captured_text) == list:
                                new_off_type_case.append(f'{captured_text[0]}{text_}')
                            elif text_ == '""':
                                new_off_type_case.append(f'{captured_text}')
                            else:
                                new_off_type_case.append(f'{captured_text}{text_}')
        new_off_type_col.append(list(set(new_off_type_case)))


        # handle type and amount
        type_amount = row['סוג הסם, כמות']
        type_amount =  re.split(r',(?!(\d{3,3}(,\d{3})*(\.\d+)?))', type_amount)
        type_amount = [item for item in type_amount if item not in [None, '']]

        for i in range(len(type_amount)):
            t = type_amount[i]
            #check if text_ is not '' or none
            if type_amount[i] is not None and type_amount[i] != '':  
                if 'MDMAמ' in type_amount[i]:
                    type_amount[i] = type_amount[i].replace('MDMAמ', 'MDMA')
                if 'LSDמ' in type_amount[i]:
                    type_amount[i] = type_amount[i].replace('LSDמ', 'LSD')
                
        type_amount = [text_.strip() for text_ in type_amount]
        type_amount = [text_ for text_ in type_amount if text_ != '']
        #remove empty spaces 
        type_amount = [text_.replace(" ","") for text_ in type_amount]

        # type_amount = [text_.split('-') for text_ in type_amount]
        # type_amount = [text_ for text_ in type_amount if text_ != '']
        # for i in range(len(type_amount)):
        #     type_amount[i] = [text_.strip() for text_ in type_amount[i]] 
        # for text in type_amount:
        #     if len(text) == 1:
        #         type_drug.append(text[0])
        #     else:
        #         amount.append(f'{text[0]} {text[1]}')
        #         type_drug.append(text[2])

        # amount_col.append(amount)
        # type_drug_col.append(type_drug)
        type_amount_col.append(type_amount)


        # handle role
        role = row['תפקיד ']
        #split by comma the role
        if role is np.nan:
            drug_role_col.append([])
            lab_role_col.append([])
        else:
            role = role.split(',')
            for r in role:
                if 'סמים' in r:
                    drug_role.append(r)
                if 'מעבדה' in r:
                    lab_role.append(r)

            drug_role_col.append(drug_role)
            lab_role_col.append(lab_role)
            
        def get_punishment(punishment):
            if 'עבודות' in actual_punishment:
                number = re.findall(r'\d+', actual_punishment)
                actual_pun.append(f'{number[0]} חודשי שירות')
            elif 'שעות' in actual_punishment or 'של"צ' in actual_punishment or 'של"ץ' in actual_punishment:
                number = re.findall(r'\d+', actual_punishment)
                actual_pun.append(f'{number[0]} שעות שירות')
            elif 'מותנה' in actual_punishment or 'על תנאי' in actual_punishment or 'על-תנאי' in actual_punishment:
                number = re.findall(r'\d+', actual_punishment)
                probation_pun.append(f'{number[0]} חודשי מאסר על תנאי')
            elif 'מאסר' in actual_punishment or 'חודשים' in actual_punishment or 'שנות' in actual_punishment or 'שנים' in actual_punishment:
                number = re.findall(r'\d+', actual_punishment)
                #get the next word after the number
                lst = actual_punishment.split()
                index = lst.index(number[0])
                next_word = lst[index + 1]
                if 'שנות' in next_word or 'שנים' in next_word:
                    number[0] = int(number[0]) * 12
                actual_pun.append(f'{number[0]} חודשי מאסר בפועל')
            elif 'קנס' in actual_punishment:
                number = re.findall(r'\d+', actual_punishment)
                fine_pun.append(f'{number[0]} ש"ח קנס')
            else:
                other_pun.append(actual_punishment)

        # handle actual punishment
        actual_punishment = row['עונש עיקרי']
        actual_punishment = replace_words_with_numbers(actual_punishment, number_dict)
        get_punishment(actual_punishment)

        punishment = row['ענש נלווה']
        #split by enter(\n) the punishment
        if punishment is np.nan:
            actual_pun_col.append(actual_pun)
            probation_pun_col.append(probation_pun)
            fine_pun_col.append(fine_pun)
            other_pun_col.append(other_pun)
        else:
            punishment = punishment.split('\n')
            for actual_punishment in punishment:
                actual_punishment = replace_words_with_numbers(actual_punishment, number_dict)
                get_punishment(actual_punishment)
            actual_pun_col.append(actual_pun)
            probation_pun_col.append(probation_pun)
            fine_pun_col.append(fine_pun)
            other_pun_col.append(other_pun)
        

        actual_punishment = row['עבירת סמים שלא הייתה ברשימה']
        if actual_punishment is np.nan:
            off_not_in_list_col.append([])
        else:
            #find index of word סעיף
            words = actual_punishment.split()
            index = words.index('סעיף')
            #get the next word after the סעיף
            next_word = words[index + 1]
            off_not_in_list_col.append(next_word)

    print(len(actual_pun_col))
    df['עבירות'] = new_off_type_col
    # df['סוג הסם'] = type_drug_col
    # df['כמות'] = amount_col
    df['סוג הסם, כמות'] = type_amount_col
    # df['תפקיד סמים'] = drug_role_col
    # df['תפקיד מעבדה'] = lab_role_col
    df['עונש בפועל'] = actual_pun_col
    df['עונש תנאי'] = probation_pun_col
    df['עונש קנס'] = fine_pun_col
    df['עונש אחר'] = other_pun_col
    df['עבירת סמים שלא הייתה ברשימה'] = off_not_in_list_col
    
    df.drop(dummy_columns, axis=1, inplace=True)
    # delete 'סוג הסם, כמות' column
    # df.drop('סוג הסם, כמות', axis=1, inplace=True)
    # df.drop('תפקיד ', axis=1, inplace=True)
    df.drop('עונש עיקרי', axis=1, inplace=True)

    
    column_mapping = {'עבירות':'OFFENCE_NUMBER',
                    'סוג הסם': 'CIR_TYPE',
                    'כמות': 'CIR_AMOUNT',
                    'אופן החזקת הנשק': 'HELD_WAY',
                    'עונש בפועל': 'ACTUAL_PUNISHMENT',
                    'עונש תנאי': 'PUNISHMENT_SUSPENDED',
                    'עונש קנס': 'PUNISHMENT_FINE',
                    'מעבדה': 'CIR_EQ',
                    'תפקיד ' : 'CIR_ROLE',
                    'סוג הסם, כמות': 'CIR_TYPE_AMOUNT',
                    }
    df.rename(columns=column_mapping, inplace=True)

    # existing_columns_to_drop = [col for col in column_to_drop if col in df.columns]

    # df.drop(existing_columns_to_drop, axis=1, inplace=True)
    
    # df.rename(columns=column_mapping, inplace=True)
    df.drop_duplicates(subset=['שם קובץ התיק'], inplace=True)
    return df

def change_case_name_foramt(df, path_df=None):
    mapping_df = pd.read_csv('/home/tak/pred-sentencing/resources/appendices/2017_mapping.csv')
    new_format_case_name = [] 
    for _, row in df.iterrows():
        try:
            new_format_case_name.append(mapping_df[mapping_df['name'] == row['מספר תיק']]['directory'].values[0])
        except:
            new_format_case_name.append(list(mapping_df[mapping_df['name'] == row['מספר תיק']]['directory'].values))
    df['מספר תיק'] = new_format_case_name
    if path_df is not None:
        df.to_csv(path_df)
        print(f"Save succesfuly DF in new format in {path_df}!")
    return df 

def check_inclusion(set1, set2):
    for obj1 in set1:
        for obj2 in set2:
            if obj1 in obj2 and obj1 != obj2:
                return True
            if obj2 in obj1 and obj1 != obj2:
                return True
    
def change_format(object_, feature_type, tagger = False):
    new_format = []
    if isinstance(object_, pd.Series):
        if len(object_) == 0:
            return []   
    elif pd.isna(object_.values[0]):
            return []
      
    if tagger:
        if (type( object_.values[0]) != list) :

            features = object_.values[0].split(',')
        else:
            features = []
            for feature in object_.values[0]:
                if len(feature) == 1:
                    continue
                if '/' in feature:
                    feature = feature.split('/')[0]
                features.append(feature.split(','))
            features = [re.sub(r'\s*\([^)]*\)', '', value) for sublist in features for value in sublist]
        
        features = [feature.strip() for feature in features]
        return features
    
    if feature_type in ['OFFENCE_NUMBER', 'OFFENCE_TYPE']:
        t = set(object_)
        for feature in set(object_):
            # pattern = r'[^\u0590-\u05FF]+'
            # feature = re.sub(pattern, '', feature)
            new_format.append(feature.replace('\\','').replace('[','').replace(']','').replace("'",'').strip())      
        return new_format
    
    for feature in set(object_):
        
        try:
            feature = ast.literal_eval(feature)
        except:
            feature = [feature]
        
        if not isinstance(feature, list):
            new_format.append(feature)
        else:
            feature_list = feature  
            for feature in feature_list:
                if feature_type == 'STATUS_WEP' and feature == 'נשק תקול':
                    feature = feature.split('נשק')[-1].strip()
                feature = re.sub(r'\s*\([^)]*\)', '', feature)
                if feature_type == 'TYPE_WEP':
                    feature = feature.replace('-', ' ')
                new_format.append(feature.replace('[','').replace(']','').replace("'",'').replace(".",'').replace("EOS_TOKEN",'').replace("EOS_TOKEN:",'').replace('xa0xa0','').replace('mentlowersplit','').strip())
    return new_format


def extract_sentence_from_excel_tagged(tagged_path, error_analisys_path, feature):
    """
    Extracet sentence by cases and feature from sentence tagges excel (by tagger)
    """
    tagged_df = pd.read_excel(tagged_path)
    error_analysis_df = pd.read_csv(error_analisys_path)
    error_analysis_df = error_analysis_df[error_analysis_df['feature'] == feature]

    tagged_df['verdict'] = tagged_df['verdict'].str.replace('_tagged', '')
    cases_list = list(error_analysis_df[error_analysis_df['feature'] == feature]['name_case'])
    if feature == 'HELD_WAY':
        feature = 'CIR_' + feature + '_WEP'
    if feature in ['USE','PURPOSE']:
        feature = 'CIR_' + feature
    
    error_analysis_df.rename(columns={'name_case': 'verdict'}, inplace=True)
    filter_df = tagged_df[(tagged_df['verdict'].isin(cases_list)) & (tagged_df[feature] == 1)][['verdict', 'text']]
    merge_df = pd.merge(filter_df, error_analysis_df, on='verdict')
    merge_df = merge_df.drop('feature', axis=1)
    merge_df.to_csv(f'{feature}_sentence_by_feature.csv', index=False)
    
# tagged_path = r'/home/ezradin/pred-sentencing/resources/data/tagging/sentence_classfication/test/2017_gt.xlsx'
# error_analisys_path = r'/home/ezradin/pred-sentencing/error_analysis.csv'
# extract_sentence_from_excel_tagged(tagged_path,
#                                    error_analisys_path,
#                                    'PURPOSE')