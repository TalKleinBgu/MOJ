import ast
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
from collections import Counter, defaultdict
from imblearn.over_sampling import SMOTE


current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..' , '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.evaluation.weapon.evaluation import change_format, convert_tagging_to_pred_foramt 
from utils.files import config_parser, setup_logger
from utils.errors.predict_sentence_cls import *

class SimilarityPairsCreation:
    """
    This class is responsible for creating similarity pairs from given data sources.
    It facilitates the creation of DataFrames for training and testing similarity models,
    based on provided tagged pairs and embedding features. It computes similarity scores
    between pairs of cases, and organizes them into structured datasets
    """
    
    def __init__(self, db_path:str = None, features_type:str = None,
                 result_path:str = None, output_dir_name:str = None,
                 tagged_pairs_path:str = None, embeddding_features_path:str = None, 
                 logger:logging.Logger = None, label_type:str = 'Tagged label', project_path='',
                 feature_from_tagged=False, tagged_feature_extraction='', ignore_column =[]):
        """

        :param db_path: 
               features_type: rgx/qa
               tagged_pairs: path csv file that contain - source|target|tagged/auto label
        """
        self.logger = logger
        self.features_type = features_type
        if db_path is not None:
            self.db_path = db_path
        if tagged_pairs_path is not None:
            self.tagged_pairs = pd.read_csv(tagged_pairs_path)
            # self.tagged_pairs = self.tagged_pairs.drop_duplicates(subset=['Source_Verdict', 'Target_Verdict'])
        if embeddding_features_path is not None:
            self.embeddding_features_path = embeddding_features_path
            self.verdict_emb_dict = self.load_embedding_feature()
            self.feature_name = list(self.verdict_emb_dict[list(self.verdict_emb_dict.keys())[0]].keys())

        else:
            self.verdict_fe_dict = self.load_features_vec(from_tagged=feature_from_tagged, project_path=project_path, tagged_feature_extraction=tagged_feature_extraction)
            self.feature_name = set()
            for key in self.verdict_fe_dict.keys():
                self.feature_name.update((self.verdict_fe_dict[key].keys()))

            
        self.label_type = label_type
        self.result_path = result_path
        self.output_dir_name = output_dir_name
        self.ignore_column = ignore_column
    
    def populate_feature_dict(self, case_name):
        try:
            feature_case_path = os.path.join(self.db_path, case_name, 'qa_features.csv')
            feature_extraction_df = pd.read_csv(feature_case_path)
            # if not is_organizing:
            # feature_extraction_df = feature_extraction_df.melt(
            #     id_vars='Text',        # Columns to keep as identifiers (unchanged)
            #     var_name='Feature_key',      # Name for the 'old' column headers
            #     value_name='Extraction'     # Name for the values from those columns
            # )
            # feature_extraction_df = feature_extraction_df.dropna(subset=['value'])
            # feature_extraction_df = feature_extraction_df[['Feature_key', 'Text', 'Extraction']]
            group_dfs = feature_extraction_df.groupby('feature_key')
            for feature, group_df in group_dfs:
                try:
                    if feature == 'CONFESSION':
                        counter = Counter(group_df['extraction'])
                        most_common_feature = ast.literal_eval(counter.most_common(1)[0][0])
                        feature_extraction_set = set(most_common_feature)
                    else:
                        feature_extraction_set = set(change_format(group_df['extraction'],
                                                                feature_type=feature))
                        if feature == 'USE':
                            if set(['ירי', 'ניסיון לירי']).issubset(feature_extraction_set):
                                feature_extraction_set = {'ירי'}
                                
                        if feature == 'STATUS_WEP' and len(feature_extraction_set) == 0:        
                            feature_extraction_set = {'תקין'}

                    if len(feature_extraction_set) > 0:
                        self.verdict_fe_dict_[case_name][feature] = feature_extraction_set


                except:
                    continue
            
            
        except:
            self.logger.error(f"{case_name} isn't exist in DB!")
    
    def load_features_vec(self, from_tagged=False, project_path='', tagged_feature_extraction=''):
        
        self.verdict_fe_dict_ = defaultdict(dict)
        if from_tagged:
            tagged_df = pd.read_excel(os.path.join(project_path,tagged_feature_extraction))
            tagged_df = convert_tagging_to_pred_foramt(tagged_df)
            grouped = tagged_df.groupby('מספר תיק')
            for name_case, group_df in grouped:
                if len(group_df) > 0:
                        for feature in group_df:
                            if feature not in self.ignore_column:
                                tagged_feature = group_df.iloc[0][feature]
                                if feature == 'USE' and pd.notna(tagged_feature):
                                    tagged_feature = tagged_feature.replace('כן,', '')

                                try:
                                    if pd.notna(tagged_feature):
                                        tagged_feature_ = group_df.iloc[0][feature]
                                        if feature == 'USE':
                                            tagged_feature_ = tagged_feature_.replace('כן,', '')
                                            tagged_feature_ = tagged_feature_.replace('לא,', '')
                                        tagger_categories = set(change_format(pd.Series(tagged_feature_),
                                                                            feature,
                                                                            tagger=True))
                                    
                                    else:
                                        tagger_categories = set()
                                except:
                                    try:
                                        if pd.notna(tagged_feature).all():
                                            tagger_categories = set(change_format(pd.Series(group_df.iloc[0][feature]),
                                                                feature))
                                        else:
                                            tagger_categories = set()
                                    except:
                                        print()
                                if len(tagger_categories) > 0:
                                    self.verdict_fe_dict_[name_case][feature] = tagger_categories
        else:
            for source_case, target_case in self.tagged_pairs[['Source_Verdict', 'Target_Verdict']].values:
                    self.populate_feature_dict(source_case)
                    self.populate_feature_dict(target_case)

        return self.verdict_fe_dict_    
                       

    def load_embedding_feature(self):
        """
        load embedding file (that contain embedding for each verdict),
        Check if the row is valid and than, and convert it to dict representaion 
        """
        verdict_emb_dict = {}
        self.features_vector_df = pd.read_excel(self.embeddding_features_path)

        for _, row in self.features_vector_df.iterrows():
            if row['Verdict']:  # Check if lists are not empty
                embd_vector = {}
                for column_name, value in row.items():
                    if 'text' not in column_name and 'Unnamed' not in column_name and (column_name not in ['OFFENCE_NUMBER', 'Verdict']):
                        embd_vector[column_name] = value
                        
                verdict = row['Verdict']  # Get first element of verdict list
                verdict_emb_dict[verdict] = embd_vector
        return verdict_emb_dict

    def augment_and_analyze_data(self, data, num_copies_dict, noise_level=0.01):
        """
        הגדלת הנתונים והוספת רעש לדוגמאות הקיימות.
        
        :param input_path: נתיב לקובץ ה-CSV המקורי.
        :param output_path: נתיב לשמירת הקובץ המוגדל.
        :param label_column: שם העמודה המכילה את הלייבלים.
        :param noise_level: רמת הרעש להוספה (ברירת מחדל: 0.01).
        """
        
        # בדיקת כמות הדוגמאות בכל קטגוריה בלייבל לפני ההגדלה
        label_counts_before = data['label'].value_counts()
        print("כמות הדוגמאות בכל קטגוריה לפני ההגדלה:\n", label_counts_before)
        
        augmented_data = data.copy()
        new_rows = []

        # מעבר על כל שורה והוספת רעש לערכים
        for i, row in data.iterrows():
            label = row['label']
            num_copies = num_copies_dict[label]
            for copy_num in range(1, num_copies + 1):
                new_row = row.copy()
                for col in data.columns:
                    if col not in ['source', 'target', 'label']:
                        noise = np.random.normal(0, noise_level)
                        new_row[col] += noise
                    elif col in ['source', 'target']:
                        new_row[col] = f"{row[col]}_copy{copy_num}"
                
                new_rows.append(new_row)
        
        augmented_data = pd.concat([augmented_data, pd.DataFrame(new_rows)], ignore_index=True)
        
        # בדיקת כמות הדוגמאות בכל קטגוריה בלייבל אחרי ההגדלה
        label_counts_after = augmented_data['label'].value_counts()
        print("The number of examples in each category after the enlargement:\n", label_counts_after)

        return augmented_data
    
    # def create_sn_dataset(self):
    #     data = []
    #     for _, row in self.tagged_pairs.iterrows():
    #         source_features = self.verdict_fe_dict[row['Source_Verdict']]
    #         target_features = self.verdict_fe_dict[row['Target_Verdict']]
    #         if len(source_features) > 0 and len(target_features) > 0:
    #             label = row[self.label_type]
    #             data.append((source_features, target_features, label))
    #     return data
        
    def create_df_to_train(self, output_dir_name = None, type_task = None, create_more_data=False, models_type=None):
        """
        Similarity between all the pairs that have a label, in the tagged file,
        return: tagged pairs with tagged label and similarity dict between the feature 
        """
        
        self.logger.info(f"Start to create DF with {self.features_type}, for train pairs similarity!")
        
        # if models_type[0] == 'siamese_network':
        #     return self.create_sn_dataset()
        
        all_verdicts_similarity = defaultdict(list)
        
        for _, row in self.tagged_pairs.iterrows():
            source_verdict = row['Source_Verdict']  # Get the source key or default to source if not in mapping
            target_verdict = row['Target_Verdict']
            
            similarity_row = self.__add_similarity_vector_to_result(source_verdict,
                                                                    target_verdict,
                                                                    all_verdicts_similarity)
        
            if similarity_row is not None:
                all_verdicts_similarity = similarity_row
                all_verdicts_similarity['label'].append(row['label'])

        feature_similarity_df = pd.DataFrame.from_dict(all_verdicts_similarity, orient='index').transpose()
        if create_more_data:
            num_copies_dict = {1:5,1.5:5, 2:6,2.5:7,4.5:6,3.5:4, 3:6, 4:3, 5:6}
            feature_similarity_df = self.augment_and_analyze_data(feature_similarity_df,
                                                                  num_copies_dict=num_copies_dict)

        if output_dir_name is not None:
            self.__save_df(feature_similarity_df, output_dir_name, type_task)
            
        return feature_similarity_df


    def create_df_to_test_new_case(self, source_verdict: str = None, 
                                   output_dir_name:str = None):
        
        """
        Given a case, the similarity between it and the other cases in the database is calculated
        """
        self.logger.info(f"Start to create DF with {self.features_type}, for test pairs similarity!")
        if source_verdict is None:
            self.logger.error("Source verdict is not provided.")
            return None
        
        all_verdicts_similarity = defaultdict(list) #contain the all similarity feature (as key), vs the source and target cases
        all_verdicts_num = list(self.verdict_emb_dict.keys())

        for target_verdict in all_verdicts_num:
            similarity_row = self.__add_similarity_vector_to_result(source_verdict,
                                                                    target_verdict,
                                                                    all_verdicts_similarity)
        
            if similarity_row is not None:
                all_verdicts_similarity = similarity_row
     
        feature_similarity_df = pd.DataFrame.from_dict(all_verdicts_similarity, orient='index').transpose()
        
        if self.result_path:
            save_path = self.__save_df(feature_similarity_df, output_dir_name)
            self.logger.info(f"Save pairs similiarity to {save_path}")
        return feature_similarity_df

    def __compute_dice_similarity(self, source_verdict, target_verdict):
        pairs_similarity = defaultdict(str)
        source_features = self.verdict_fe_dict[source_verdict]
        target_features = self.verdict_fe_dict[target_verdict]
        if (len(source_features) == 0) or (len(target_features) == 0):
            return None
        for feature in self.feature_name:  
            # have scenario that feature isnt exist in some cases
            try:
                source_feature = source_features[feature]
                target_feature = target_features[feature]       
                intersection = len(source_feature.intersection(target_feature))
                union = len(source_feature) + len(target_feature)
                if union == 0:
                    agreement = 1
                else:    
                    agreement = (2.0 * intersection) / union  # Dice coefficient formula
            except Exception as e:
                print(e)
                agreement = -2
            
            pairs_similarity[feature] = agreement
        return pairs_similarity
    
    def __add_similarity_vector_to_result(self, source_verdict, target_verdict, all_verdicts_similarity, embedding_similarity=False):
        """
        Compute similarity between source and target verdicts and add it to the result.
        Args:
            source_verdict (str): Source verdict identifier.
            target_verdict (str): Target verdict identifier.
            all_verdicts_similarity (dict): Dictionary containing all pair similarities.

        Returns:
        dict or None: Updated dictionary containing all pair similarities or None if similarity computation fails.
        """
        if embedding_similarity:
            pairs_similarity = self.__compute_embedding_similarity(source_verdict, target_verdict)
        else:
            pairs_similarity = self.__compute_dice_similarity(source_verdict, target_verdict)
        if pairs_similarity is None:
            return None
        
        all_verdicts_similarity['source'].append(source_verdict)
        all_verdicts_similarity['target'].append(target_verdict)
        for feature in self.feature_name:
            if feature != 'full_verdict':
                if feature in pairs_similarity.keys():
                    all_verdicts_similarity[feature].append(pairs_similarity[feature])
                else:
                    all_verdicts_similarity[feature].append(-1)
        if embedding_similarity:
            offence_similarity = self.__offence_number_similarity(source_verdict, target_verdict)
            all_verdicts_similarity['OFFENCE_NUMBER'].append(offence_similarity)
        return all_verdicts_similarity         

    def __offence_number_similarity(self, source, target):
        df = self.features_vector_df
        source_offence_number = eval(df.loc[df['Verdict'] == source, "OFFENCE_NUMBER"].iloc[0])
        target_offence_number = eval(df.loc[df['Verdict'] == target, "OFFENCE_NUMBER"].iloc[0])
        source_offence_number = " ".join(source_offence_number)
        target_offence_number = " ".join(target_offence_number)
        
        if source_offence_number == target_offence_number:
            return 1
        
        elif target_offence_number in source_offence_number :
            return 2
        
        elif source_offence_number in target_offence_number:
            return 3
        else:
            return 0
    
        

    def __compute_embedding_similarity(self, source, target):
        try:
            # Get the target key or default to target if not in mapping

            source_vector = self.verdict_emb_dict[source]
            target_vector = self.verdict_emb_dict[target]
            feature_similarity_dict = {}

            for feature in source_vector:
                # TODO: add more similarity except cosin similarity (offence_number, offence type and ex.) 
                if feature != 'full_verdict':  # calculate without the full verdict
                    if source_vector[feature] is None or target_vector[feature] is None:
                        # similarity.append(np.finfo(np.float32).max);
                        feature_similarity_dict[feature] = -1
                    
                    
                    elif type(source_vector[feature]) == int or type(target_vector[feature]) == int:
                        feature_similarity_dict[feature] = -1
                    else:
                        try:
                            source_vector[feature] = eval(str(source_vector[feature]))
                            target_vector[feature] = eval(str(target_vector[feature]))
                        except:
                            print()
                        feature_similarity_dict[feature] = cosine_similarity(source_vector[feature],
                                                                             target_vector[feature])[0][0]

            return feature_similarity_dict

        except KeyError as e:
             self.logger.error("KeyError occurred: %s", e)
             return None
        except SyntaxError as e:
            self.logger.error("SyntaxError occurred: %s", e)
            return None

    def __save_df(self, feature_similarity_df, output_dir_name, type_task):
        """
        Save the DataFrame containing pair similarities to a CSV file.

        """
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d')
        dir_save_path = os.path.join(self.result_path, 'pairs_similarity', 
                                     f'{formatted_datetime}_{self.features_type}_{output_dir_name}')
        os.makedirs(dir_save_path, exist_ok=True)
        save_path = os.path.join(dir_save_path ,f'{type_task}.csv')
        feature_similarity_df.to_csv(save_path)
        self.logger.info(f'Save {type_task}.csv in {save_path}')
        return save_path
    


def main(params):
    logger = setup_logger(save_path=os.path.join(params['result_path'], 'logs'),
                          file_name='pairs_similarity')
    ps_params = params['pairs_similarity']
    handler = SimilarityPairsCreation(db_path=params["db_path"],
                                      tagged_pairs_path=ps_params["tagged_pairs_path"], 
                                      embeddding_features_path=ps_params["embeddding_features_path"],
                                      features_type=ps_params['features_type'],
                                      result_path=params["result_path"],
                                      logger=logger,
                                      label_type=ps_params['label_type']
                                      )
    
    sim_df = handler.create_df_to_train()
    handler.create_df_to_test_new_case(source_verdict='SH-18-07-52337-770',
                                       output_dir_name=params['output_dir_name'])


if __name__ == "__main__": 
   params = config_parser("main_config")
   main(params)