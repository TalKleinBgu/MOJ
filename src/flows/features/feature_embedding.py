from collections import defaultdict
import os
import sys
import numpy as np
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM


current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from scripts.features.feature_embedding_class import FeatureEmbedding
from utils.files import config_parser, save_json, setup_logger, weap_type_extract
from utils.features import aggrigateion_and_union

 
class FeatureEmbeddingFlow():
    def __init__(self, embedding_model_name:str = None, result_path:str = None, 
                 manual_tagging_path:str = None, extraction_methods:str = None,
                 logger:logging.Logger = None, db_path:str = None) -> None:
        
        self.db_path = db_path
        self.model = AutoModelForMaskedLM.from_pretrained(embedding_model_name,
                                                          output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.result_path = result_path
        self.manual_tagging_df = pd.read_csv(manual_tagging_path)
        self.extraction_methods = extraction_methods
        self.logger = logger

        
    def result_folder_process(self, method:str = None):
        """
        this function get method type (regex or Q&A),
        over on each case in result path (in this path we can find the textual 
        feature that extract)
        """
        all_verdict_fe = []
        for case_dir in os.listdir(self.db_path):
            case_dir_path = os.path.join(self.db_path, case_dir)
            
            for file in os.listdir(case_dir_path):
                    if method in file:
                        feature_extraction_path = os.path.join(case_dir_path, file)
                        feature_extraction_df = pd.read_csv(feature_extraction_path)
                        
                        # After extracting the features, there can be several sentences in which the same weapon appeared, 
                        # for example, in this operation we reduce the duplications
                        feature_extraction_df = aggrigateion_and_union(feature_extraction_df)
                        try:
                            OFFENCE_NUMBER = feature_extraction_df[feature_extraction_df['feature_key']=='OFFENCE_NUMBER'].iloc[0]['extraction']
                        except:
                            self.logger.error(f"Case {case_dir} failed to pass embedding!")
                        row_data = {'Verdict': case_dir, 'OFFENCE_NUMBER': OFFENCE_NUMBER[0]}

                        fe = FeatureEmbedding(fe_df_path=feature_extraction_path,
                                              model=self.model,
                                              tokenizer=self.tokenizer)
                        fe_dict = fe.get_embedding_dict()
                        for feature in fe_dict['embd_vector']:
                            row_data[feature] = fe_dict['embd_vector'][feature]
                            
                        for feature in fe_dict['text_vector']:
                            row_data[feature+ '_text'] = fe_dict['text_vector'][feature]
                            
                        all_verdict_fe.append(row_data)
        return pd.DataFrame(all_verdict_fe)
                        

    def manual_tagging_file_process(self):
        """
        thid function over on tagging feature extraction csv, and for each case
        generate his embedded feature vectore. 
        """
        
        all_verdict_fe = []
        fe = FeatureEmbedding(model=self.model, tokenizer=self.tokenizer)
        manual_tagging_df = weap_type_extract(self.manual_tagging_df)

        for _, row in manual_tagging_df.iterrows():
            row_data = {'Verdict': row['Verdict'], 'OFFENCE_ID': row['Offence_number']}
            for feature in fe.feature_association['textual']:
                if feature == 'USE' and row[feature] == 'לא':
                    emb_feature_vector = None
                else:
                    emb_feature_vector = fe.embed_element(row[feature])
                    try:
                        emb_feature_vector = emb_feature_vector[0]
                    except:
                        emb_feature_vector = None
                row_data[feature] = emb_feature_vector
                row_data[feature + '_txt'] = row[feature]
                
            for feature in fe.feature_association['boolian']:
                try:
                    row_data[feature] = row[feature]
                except:
                    row_data[feature] = None
                    
            all_verdict_fe.append(row_data)

        return pd.DataFrame(all_verdict_fe)

    def run_flow(self):
        """
        This function get path to data you want to process (result folder if its (qa/regex) and db), 
        and list of methods that you want to procees 
        Note that if you want to process also qa/regex and manual method - you need to supply the appropriate paths 
        """    
        self.logger.info(f'Feature embedding for {self.extraction_methods} methods is starting!')
        all_verdict_fe = ''
        # embedding the qa/regex feature
        for method in self.extraction_methods:
            if method in ['qa', 'regex']:
                all_verdict_fe = self.result_folder_process(method)
            
            elif method == 'manual':
                all_verdict_fe = self.manual_tagging_file_process()   
            
            if type(all_verdict_fe) != str:
                save_path = os.path.join(self.result_path, 'embedding', 
                                        f"fearture_{method}_emb.xlsx")
                all_verdict_fe.to_excel(save_path)
                self.logger.info(f"save {method} in {save_path}")
            else:
                self.logger.error(f"Faild to embedd feature by {method} method")    
                
                
def main(params):
    logger = setup_logger(save_path=os.path.join(params['result_path'], 'logs'),
                          file_name='predict_similarity_test')
    
    feature_embedding_params = params['feature_embedding']
    
    fef = FeatureEmbeddingFlow(embedding_model_name=feature_embedding_params['embedding_model_name'],
                               manual_tagging_path=feature_embedding_params['manual_tagging_path'],
                               extraction_methods=feature_embedding_params['extraction_methods'],
                               result_path=params['result_path'],
                               db_path=params['db_path'],
                               logger=logger)
    fef.run_flow()


if __name__ == '__main__':
    params = config_parser('main_config')
    main(params)