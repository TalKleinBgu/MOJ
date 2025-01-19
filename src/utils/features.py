import os
import sys
import pandas as pd
import re
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir,
                                                   '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.evaluation.weapon.evaluation import change_format


def flatten_list(lst):
    if isinstance(lst[0], list):
        return [item for sublist in lst for item in sublist]
    else:
        return lst


def unique_list(lst):
    return list(set(lst))


def aggrigateion_and_union(feature_df):
    
    grouped_df = feature_df.groupby('feature_key').agg(lambda x: x.tolist()).reset_index()
    grouped_df['extraction'] = grouped_df['extraction'].apply(flatten_list)
    grouped_df['extraction'] = grouped_df['extraction'].apply(unique_list)
    return grouped_df


def is_convertible_to_int(case_name):
    try:
        int(case_name)
        return True
    except ValueError:
        return False
    
    
def row_to_similarity_vector(row):
    return row.drop(['source', 'target', 'label']).tolist()


def train_format_conversion(old_df, binary_label):
    new_data = []
    for _, row in old_df.iterrows():
        source = row['source']
        target = row['target']
        if binary_label:
            if row['label'] >= 3:
                label = 1
            else:
                label = 0
        else: 
            label = row['label']
        similarity_vector = row_to_similarity_vector(row)
        new_data.append([source, target, similarity_vector, label])

    new_df = pd.DataFrame(new_data, columns=['source', 'target', 'similarity_vector', 'label'])
    return new_df

def exctract_example_sentence(target_feature):
    sentence_exampels = []
    files_path = '/home/ezradin/pred-sentencing/resources/data/tagging/sentence_classfication/test'
    for file in os.listdir(files_path):
        senentence_tagging_df = pd.read_excel(os.path.join(files_path,file))
        try:
            sentence_exampels.extend(list(senentence_tagging_df[senentence_tagging_df[target_feature] == 1]['text'].values))
        except:
            print()
    
    sentence_example_df = pd.DataFrame(sentence_exampels)
    sentence_example_df.to_csv(f"{target_feature}_sentence_example.csv")
    print()
        
        
def add_verdict_num(verdict_path: str, db_path: str):
    sentence_case_dict = {}
    fe_df = pd.read_csv(verdict_path)

    for case_dir in os.listdir(db_path):
        preproc_df = pd.read_csv(os.path.join(db_path, case_dir, 'sentence_tagging.csv'))
        for i, row in preproc_df.iterrows():
            if row['reject'] == 1: 
                continue

            sentence_case_dict[row['text']] = case_dir
    verdicts = []
    for i,row in fe_df.iterrows():
        try:
            verdicts.append(sentence_case_dict[row['text']])
        except:
            verdicts.append(None)
        
    fe_df['verdict'] = verdicts
    fe_df.to_csv(verdict_path)
    print(f'Save new version of feature DF in {verdict_path}')  
# target_feature = 'CIR_OBTAIN_WAY_WEP'
# exctract_example_sentence(target_feature)



def weapon_status_reduction(fe_categories_):
    severity_order = [
    'נשק מפורק', 
    'תקול', 
    'תקין', 
    'נשק מופרד מתחמושת', 
    'נשק עם מחסנית בהכנס', 
    'נשק עם כדור בקנה'
    ]
    severity_map = {severity: index for index, severity in enumerate(severity_order)}
    the_worst = max(fe_categories_, key=lambda feature: severity_map[feature])
    return [the_worst]

def get_tagger_categories(tagged_feature, group_df, feature):
    """Helper function to extract tagger categories."""
    if isinstance(tagged_feature, list) or isinstance(tagged_feature, pd.Series):
        # If it's a list or Series, treat it as non-NA
        tagged_feature_ = group_df.iloc[0][feature]
        tagged_feature_ = pd.Series(tagged_feature_) if isinstance(tagged_feature_, list) else tagged_feature_
        tagger_categories = set(change_format(tagged_feature_, feature, tagger=True))
    elif pd.notna(tagged_feature):
        tagged_feature_ = group_df.iloc[0][feature]
        tagger_categories = set(change_format(pd.Series(tagged_feature_), feature, tagger=True))
    else:
        tagger_categories = set()
    return tagger_categories
    
def handle_weapon_status(feature, fe_categories):
        """Handle specific logic for weapon status categories."""
        if set(['ירי', 'ניסיון לירי']).issubset(fe_categories):
            fe_categories = {'ירי'}
        
        if feature == 'STATUS_WEP' and not fe_categories:
            fe_categories = {'תקין'}
        
        return fe_categories

def get_TYPE_AMOUNT(fe_match_case,drugs_list):
    TYPE_AMOUNT = set()
    #group the data by text column
    grouped = fe_match_case.groupby('text')
    for name, group in grouped:
        if len(group) > 1:
            if 'CIR_AMOUNT' in group['feature_key'].values and 'CIR_TYPE' in group['feature_key'].values:
                type_extration = group[group['feature_key'] == 'CIR_TYPE']['extraction'].values[0]
                type_extration = type_extration.replace('[','').replace(']','').replace('\'','')
                amount_extraction = group[group['feature_key'] == 'CIR_AMOUNT']['extraction'].values[0]
            #GET THE WORD IN TYPT_EXTRATION AFTER THE WORD הוא or מסוג
                if len(type_extration) > 0:
                    #split the text by space
                    type_extration = type_extration.split(' ')
                    #indx of the word הוא or מסוג
                    type_word_indx = type_extration.index('הוא') if 'הוא' in type_extration else type_extration.index('מסוג')
                    #get the next word after הוא or מסוג
                    type_word = type_extration[type_word_indx + 1]
                if len(amount_extraction) > 0:
                    pattern = r'\[[^\[\]]+\]'
                    matches = re.findall(pattern, amount_extraction)
                    if len(matches) > 0:
                        for match in matches:
                            match = match.replace('[','').replace(']','')
                            #remove spaces
                            match = match.replace(' ','')
                if type_word and match:
                    TYPE_AMOUNT.add(f'{match}-{type_word}')

    return TYPE_AMOUNT
    
def handle_offence_status(feature, fe_categories):
    """Handle specific logic for offence status categories."""
    #split by ,
    if feature == 'OFFENCE_NUMBER':
        if isinstance(fe_categories, set):
            # Flatten the set into a list or string
            fe_categories = list(fe_categories)

            # If needed as a string:
            fe_categories = ','.join(fe_categories)

        # Proceed with split
        fe_categories = {item.strip() for item in fe_categories.split(',')}
        # return it to set      
        fe_categories = set(fe_categories)
    return fe_categories

def calculate_dice_coefficient(tagger_categories, fe_categories):
        """Calculate the Dice coefficient between tagger and model categories."""
        intersection = len(tagger_categories.intersection(fe_categories))
        union = len(tagger_categories) + len(fe_categories)
        
        if union == 0:
            return 1
        return (2.0 * intersection) / union  # Dice coefficient formula


def create_error_analysis_entry(name_case, fe_match_case, feature, fe_categories, tagger_categories):
    """Create a dictionary entry for error analysis."""
    return {
        'name_case': name_case,
        'text': fe_match_case[fe_match_case.feature_key == feature].text,
        'model_extraction': list(fe_categories),
        'tagger_extraction': list(tagger_categories),
        'feature': feature,
    }
    

def save_agreements(all_agreements, save_path):
    """Save agreements and write the results to a file."""
    with open(save_path, 'w') as file:
        file.write("Mean Dice coefficient for:\n")
        for feature, agreements in all_agreements.items():
            mean_agreement = sum(agreements) / len(agreements)
            file.write(f"\t {feature}: {round(mean_agreement, 3)}\n")
        print(f"\t {feature}: {round(mean_agreement, 3)}\n")
        

def save_error_analysis(error_analysis_data, save_path):
        """Save error analysis data to CSV."""
        error_analysis_df = pd.DataFrame(error_analysis_data)
        error_analysis_df.to_csv(save_path, index=False)
        # error_analysis_df.to_csv('error_analysis.csv', index=False)
        print(f'analysis data saved to {save_path}')
        
        
def generate_sentence_cls_df(db_path):
    """
    Traverse each case directory, load the 'qa_features.csv' file,
    and append the sentence classifications to one merged DataFrame
    for pipeline evaluation.
    """
    merged_df = pd.DataFrame()
    for root, cases_dir, dir_  in os.walk(db_path):
        if cases_dir not in [['sentence_calssification'], ['10.8_newsetfit']]:
            for case in cases_dir:
                try:
                    cls_sentence_df = pd.read_csv(os.path.join(root, case, 'qa_features.csv'))

                    cls_sentence_df['verdict'] = case
                    
                    merged_df = pd.concat([merged_df, cls_sentence_df], ignore_index=True)
                except  Exception as e:
                    print(f'An error occurred: {e}')
    
    return merged_df

