import os
from pathlib import Path
import sys
import importlib

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..' ))
sys.path.insert(0, pred_sentencing_path)

from scripts.similarity_cases.pairs_similarity_creation import SimilarityPairsCreation
from scripts.similarity_cases.training_case_similarity_model import CaseSimilarity_Model
from utils.files import config_parser, setup_logger

project_path = Path(os.path.dirname(__file__)).parent.parent.parent


def train_similarity_case(params, logger, domain):
   
   #  Embedding extraction!!
   #  handler = SimilarityPairsCreation(db_path=params["db_path"],
   #                                    tagged_pairs_path=sp_params["tagged_pairs_path"], 
   #                                    embeddding_features_path= sp_params["embeddding_features_path"],
   #                                    features_type=sp_params['features_type'],
   #                                    logger=logger,
   #                                    label_type=sp_params['label_type'])
    
   #  Feature extraction!!
   db_path = params["db_path"].format(domain=domain)
   tagged_pairs_path = params["tagged_pairs_path"].format(domain=domain)
   tagged_feature_extraction = params["tagged_feature_extraction"].format(domain=domain)
   handler = SimilarityPairsCreation(db_path=db_path,
                                     tagged_pairs_path=tagged_pairs_path, 
                                     features_type=params['features_type'],
                                     result_path=params['result_path'],
                                     feature_from_tagged=params['feature_from_tagged'],
                                     logger=logger,
                                     label_type=params['label_type'],
                                     project_path=project_path,
                                     tagged_feature_extraction = tagged_feature_extraction,
                                     ignore_column=params['ignore_column']
                                     )
   

      
   case_pairs = handler.create_df_to_train(output_dir_name=params['output_dir_name'],
                                           create_more_data=params['create_more_data'],
                                           type_task=params['type_task'],
                                           models_type=params['models_type'])   
   
   model = CaseSimilarity_Model(case_pairs=case_pairs, 
                                 extraction_type=params['features_type'],
                                 result_path=params['result_path'],
                                 models_type=params['models_type'],
                                 seed = params['seed'],
                                 create_more_data=params['create_more_data'],
                                 output_dir_name = params['output_dir_name'],
                                 logger=logger)
   
   if params['models_type'] == 'LLM':
      model.claude_response(handler, params['api_key'], params['prompt_path'],  params['predict_path'], domain)
         
   else:
      model.train(loocv=params['loocv'])
   # doc_sim.train()
   
    
    
def main():
   main_params = config_parser("", "main_config")    
   logger = setup_logger(save_path=os.path.join(main_params['result_path'], 'logs'),
                         file_name='pairs_similarity')
   
   domain = main_params["domain"]    

   params = config_parser("similarity", domain)
   
   # sp_params = params['pairs_similarity']

   train_similarity_case(params, logger, domain) 


if __name__ == "__main__":
   main()