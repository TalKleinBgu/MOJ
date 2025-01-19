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
                                                    '..', '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import config_parser, get_date, setup_logger
# from utils.evaluation.weapon.evaluation import change_case_name_foramt, change_format, convert_tagging_to_pred_foramt
from utils.punishment_extractor import case_mapping_dict
from utils.features import *


def dynamic_import(import_type):
    # Construct the base module path using the import_type
    module_base = f"utils.evaluation.{import_type}"

    # Dynamically import the three classes from their respective modules
    # change_case_name_foramt = importlib.import_module(f"{module_base}.evaluation").change_case_name_foramt
    change_format = importlib.import_module(f"{module_base}.evaluation").change_format
    convert_tagging_to_pred_foramt = importlib.import_module(f"{module_base}.evaluation").convert_tagging_to_pred_foramt
    
    # Return the imported classes as a tuple
    return change_format, convert_tagging_to_pred_foramt


class Evaluation():
    
    def __init__(self, path_tagging_df, save_path, db_path = None, logger = None) -> None:
        self.tagging_df = pd.read_csv(path_tagging_df)
        self.new_foramt_tagging_df = convert_tagging_to_pred_foramt(self.tagging_df)   
        # self.new_foramt_tagging_df = change_case_name_foramt(new_foramt_tagging_df)
        self.save_path = save_path
        self.db_path = db_path
        self.logger = logger
        
    def feature_extraction_eval(self, ignore_columns, feature_extraction_path=None, feature_extraction_df=None, case_mapping_dict=None, error_analysis=False):
        """
        Comparison between the features extracted by the taggers and the extraction of the model from
        the sentences tagged by the taggers (in order to evaluate the accuracy of the model in the task 
        of extracting the sentences regardless of the first stage of the pipeline).
        """
    
        # Load and process tagged data
        tagged_df = self.new_foramt_tagging_df.drop(self.new_foramt_tagging_df.index[-1])
        if feature_extraction_path is not None:
            fe_df = pd.read_csv(feature_extraction_path)
        else:
            fe_df = feature_extraction_df
        print(tagged_df.columns)
        grouped = tagged_df.groupby('שם קובץ התיק')
        all_agreements = defaultdict(list)
        error_analysis_data = []

        # Iterate through grouped data
        for name_case, group_df in grouped:
            if group_df.empty:
                continue
            
            fe_match_case = fe_df[fe_df['verdict'] == name_case]
            if fe_match_case.empty:
                continue

            # Process features
            for feature in group_df:
                if feature in ignore_columns :
                    continue

                tagged_feature = group_df.iloc[0][feature]

                try:
                    tagger_categories = get_tagger_categories(tagged_feature, group_df, feature)
                except Exception as e:
                    print(f"Error in {name_case} for feature {feature}: {e}")
                    tagger_categories = set()
                if feature == 'CIR_TYPE_AMOUNT':
                    drugs=[]
                    fe_categories = get_TYPE_AMOUNT(fe_match_case,drugs)
                else:
                    fe_categories = set(change_format(fe_match_case[fe_match_case['feature_key'] == feature]['extraction'], feature))
                #check if fe_categories is empty
                if feature =='CIR_EQ' and not fe_categories:
                    fe_categories = {'לא'}

                # Specific logic for weapon status
                # fe_categories = handle_weapon_status(feature, fe_categories)

                fe_categories = handle_offence_status(feature, fe_categories)

                # Calculate Dice coefficient
                agreement = calculate_dice_coefficient(tagger_categories, fe_categories)
                all_agreements[feature].append(agreement)

                # Error analysis
                if error_analysis and agreement < 1:
                    error_analysis_data.append(create_error_analysis_entry(name_case, fe_match_case, feature, fe_categories, tagger_categories))

        # Save error analysis if required
        if error_analysis:
            save_path = self.save_path + '/error_analysis.csv'
            save_error_analysis(error_analysis_data, save_path)

        # Save agreements and log results
        save_path = self.save_path + '/dice_coefficient.txt'
        save_agreements(all_agreements, save_path)
        self.logger.info(f'Save to {save_path}!!')
    
    # def maxim_test_feature_extraction_eval(self, extraction_type = 'qa', error_analisys=False):
    #     feature_avg_scores = {}
    #     all_agreements = defaultdict(list)
    #     delete = []
    #     new_foramt_tagging_df = convert_tagging_to_pred_foramt(self.tagging_df)   
    #     new_foramt_tagging_df = change_case_name_foramt(new_foramt_tagging_df)
    #     fe_features_path =  "/sise/home/maximbr/DICTA-Chat-2.0-LLM/output/"
    #     for case_df in os.listdir(fe_features_path):
    #         feature_extraction_df = pd.read_csv(os.path.join(fe_features_path, case_df))
    #         if feature_extraction_df[feature_extraction_df['feature_key'] == 'TYPE_WEP']['extraction'].values[0] == "['ERROR']":
    #             continue
    #         case_dir = case_df.split('.txt')[0]
    #         feature_tagging_df = new_foramt_tagging_df[new_foramt_tagging_df['מספר תיק'] == case_dir]
    #         if len(feature_tagging_df) > 0:
    #             group_dfs = feature_extraction_df.groupby('feature_key')
    #             for feature, group_df in group_dfs:
    #                 delete.append(feature)
    #                 try:
    #                     feature_extraction_categories = set(change_format(group_df['extraction'],
    #                                                                       feature_type=feature))
    #                     tagger_categories = set(change_format(feature_tagging_df[feature],
    #                                                           feature_type=feature, 
    #                                                           tagger = True))
    #                     if 'ERROR' in feature_extraction_categories:
    #                         continue
    #                     tagger_categories.discard('מוסלק - מוסתר')
    #                     feature_extraction_categories.discard('מוסלק - מוסתר')
    #                 except:
    #                     continue
    #                 intersection = len(feature_extraction_categories.intersection(tagger_categories))
    #                 union = len(feature_extraction_categories) + len(tagger_categories)
    #                 if union == 0:
    #                     agreement = 1
    #                 else:
    #                     agreement = (2.0 * intersection) / union  # Dice coefficient formula
    #                 if error_analisys:
    #                     if agreement < 1:
    #                         all_agreements[feature].append((feature_extraction_categories, 
    #                                                         tagger_categories,
    #                                                         agreement,
    #                                                         case_dir))
    #                 else:
    #                     all_agreements[feature].append(agreement)
    #     if error_analisys:
    #         error_analisys_df = pd.DataFrame([
    #             {'feature': feature, 'feature extraction': feature_extraction_categories, 'feasture tagged': tagger_categories, 'agreement': agreement}
    #             for feature, (feature_extraction_categories, tagger_categories, agreement) in all_agreements.items()
    #         ])
    #         error_analisys_df.to_csv(self.save_path)
    #         return all_agreements
        
    #     for feature, scores in all_agreements.items():
    #         if scores:
    #             avg_score = sum(scores) / len(scores)
    #             feature_avg_scores[feature] = (round(avg_score, 3), len(scores))
        
    #     # Save the feature_avg_scores as a DataFrame
    #     feature_avg_scores_df = pd.DataFrame([
    #         {'feature': feature, 'Dice score': score, 'case_num': num_scores}
    #         for feature, (score, num_scores) in feature_avg_scores.items()
    #     ])
        
    #     feature_avg_scores_df.to_csv(self.save_path, index=False)
    #     print(f"Results save in {self.save_path} successfully!")
    #     return feature_avg_scores
    
    # def tagging_agreement(self):
    #     grouped = self.tagging_df.groupby('מספר תיק')
    #     all_agreements = defaultdict(list)
    #     for name, group_df in grouped:
    #         if len(group_df) > 1:
    #             for feature in group_df:
    #                 if feature not in ignore_columns:
    #                     if group_df[feature].nunique() not in [0]:
    #                         tagger1_categories = set(map(str.strip, str(group_df.iloc[0][feature]).split(',')))
    #                         tagger2_categories = set(map(str.strip, str(group_df.iloc[1][feature]).split(',')))
                        
    #                         intersection = len(tagger1_categories.intersection(tagger2_categories))
    #                         union = len(tagger1_categories) + len(tagger2_categories)
    #                         agreement = (2.0 * intersection) / union  # Dice coefficient formula
    #                         all_agreements[feature].append(agreement)
        
    #     with open(self.save_path, 'w') as file:
    #         for key, agreements in all_agreements.items():
    #             mean_agreement = sum(agreements) / len(agreements)
    #             file.write(f"Mean Dice coefficient for {key}: {round(mean_agreement, 3)}\n")
        
    #     self.logger.info(f'Save to {self.save_path}!!')

    
    # def feature_extraction_counter(self, extraction_type = 'qa'):
    #     counter = Counter()
    #     for case_dir in os.listdir(self.db_path):
    #         for file in os.listdir(os.path.join(self.db_path, case_dir)):
    #             if extraction_type in file:
    #                 extraction_df = pd.read_csv(os.path.join(self.db_path, case_dir,file))
    #                 counter.update(extraction_df['feature_key'])
        
    #     return counter
    
    # # TODO move
    # def sentence_cls_counter(self):
    #     sentence_counter = Counter()
    #     case_conter = Counter()
    #     for case_dir in os.listdir(self.db_path):
    #         for file in os.listdir(os.path.join(self.db_path, case_dir)):
    #             if 'tagging' in file:
    #                 extraction_df = pd.read_csv(os.path.join(self.db_path, case_dir,file))
    #                 col_counts = extraction_df.apply(lambda col: (col == 1).sum())
    #                 for col, count in col_counts.items():
    #                     sentence_counter[col] += count
    #                     if count > 0:
    #                         case_conter[col] += 1
    #     print()
    #     print('CLS Case counter:')
    #     print()
    #     for feature in case_conter:
    #         print(f'{feature}: {case_conter[feature]}')
    #     print()
    #     print('CLS Sentence counter:')
    #     print()
    #     for feature in sentence_counter:
    #         print(f'{feature}: {sentence_counter[feature]}')
    #     return sentence_counter, case_conter
    
        
def main(params, domain):     
    
    # mapping_path = 'resources/appendices/2017_mapping.csv'
    
    # case_mapping_dict_ = case_mapping_dict(mapping_path)
    # case_mapping_dict_ = {val:key for key,val in case_mapping_dict_.items()}
    

    ## pipline evaluation - on clssified sentence case
    logger = setup_logger(os.path.join(params['result_path'],
                        'logs'), file_name='evaluation_feature_extraction_test')
    
    feature_extraction_df = generate_sentence_cls_df(params['db_path'])
    save_path = params['save_path']
    #make sure the path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    eval_ = Evaluation(path_tagging_df=params['path_tagging_df'],
                       save_path=save_path,
                       db_path=params['db_path'],
                       logger=logger)
    eval_.feature_extraction_eval(ignore_columns = params['ignore_columns'],
                                  feature_extraction_df = feature_extraction_df,
                                  error_analysis=True)
        
if __name__ == '__main__':
    # Load main configuration parameters
    main_params = config_parser("", "main_config")
    # Extract the domain from the main configuration
    domain = main_params["domain"]
    
    change_format, convert_tagging_to_pred_foramt = dynamic_import(domain)

    # Parse the specific feature extraction parameters for the given domain
    params = config_parser("evaluation_features", domain)
    
    # Run the main function with the parsed parameters
    main(params, domain)
