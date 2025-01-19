import ast
from collections import Counter
import re
import pandas as pd
# 
column_to_drop = ['מספר תיק_ישן','חותמת זמן', 'מתייג ', 'עבירות נוספות',
                  'כסף ששולם', 'מכירה לסוכן', 'כמות תחמושת', 'תכנון', 
                  'נזק', 'בריחה מרדף', 'מתחם ענישה - מאשימה ', 'מתחם ענישה - שופט ', 'עונש',
                  'הערות ומחשבות', 'חרטה', 'מתחם ענישה - מאשימה', 'מתחם ענישה - שופט']


# column_to_drop = ['חותמת זמן', 'מתייג ', 'עבירות נוספות',
#                   'כסף ששולם','כמות תחמושת', 'תכנון', 
#                   'נזק', 'מתחם ענישה - מאשימה ', 'מתחם ענישה - שופט ', 'עונש',
#                   'הערות ומחשבות', 'חרטה']

column_mapping = {'מספר עבירה':'OFFENCE_NUMBER',
                  'סוג העבירה': 'OFFENCE_TYPE',
                  'אופן קבלת הנשק': 'OBTAIN_WAY',
                  'סטטוס הנשק': 'STATUS_WEP',
                  'אופן החזקת הנשק': 'HELD_WAY',
                  'שימוש': 'USE',
                  'הודאה': 'CONFESSION',
                  'מטרה-סיבת העבירה' : 'PURPOSE',
                  'case': 'מספר תיק'
                  }
doplicate = ['מספר תיק']

def convert_tagging_to_pred_foramt(df):
    dummy_columns = [col for col in df.columns if col.startswith('סוג הנשק')]
    
    new_wep_type_col = []
    for _, row in df.iterrows():
        new_wep_type_case = []
        for column_name in dummy_columns:
            if row[column_name] != 0 and (not pd.isna(row[column_name])):
                    match = re.search(r"\[(.*?)\]", column_name)
                    if match:
                        captured_text = match.group(1)
                        if type(captured_text) == list:
                            new_wep_type_case.append(captured_text[0])
                        else:
                            new_wep_type_case.append(captured_text)

        new_wep_type_col.append(list(set(new_wep_type_case)))
    
    
    df['TYPE_WEP'] = new_wep_type_col
    df.drop(dummy_columns, axis=1, inplace=True)
    existing_columns_to_drop = [col for col in column_to_drop if col in df.columns]

    df.drop(existing_columns_to_drop, axis=1, inplace=True)
    
    df.rename(columns=column_mapping, inplace=True)
    df.drop_duplicates(subset=['מספר תיק'], inplace=True)
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
        features = []

        if (type( object_.values[0]) != list) :
        # Check if the first element is not a list
            if isinstance(object_.values[0], str):
                # Process as a comma-separated string
                for item in object_:
                    features.extend(item.split(','))
            else:
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