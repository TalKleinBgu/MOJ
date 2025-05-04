import logging
import os
import sys
import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# Get the directory where the current file is located
current_dir = os.path.abspath(__file__)
# Construct the path to the project root (pred_sentencing_path)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# Insert the project root path into sys.path so that local modules can be imported
sys.path.insert(0, pred_sentencing_path)

# Import custom utility modules and classes
from scripts.features.feature_extraction.qa_extraction import QA_Extractor
from scripts.features.feature_extraction.regex_extraction import RegexExtractor
from utils.features import is_convertible_to_int
from utils.files import config_parser, setup_logger

def feature_extraction_flow(db_path: str = None,
                            qa_extraction_flag: bool = True,
                            regex_extraction_flag: bool = True,
                            model_name: str = None,
                            logger: logging.Logger = None,
                            tagged_file_path: str = None,
                            db_flow: bool = False,
                            experiment_name: str = '',
                            save_path: str = '',
                            domain: str = ''):
    
    qa = None      # Will hold an instance of QA_Extractor if Q&A extraction is enabled
    methods = ''   # A string to describe which extraction methods will be used

    # Q&A extraction block
    if qa_extraction_flag:
        # If the model name includes 'dictalm2' (example usage), we load a specialized model
        if 'dictalm2' in model_name:
            device = 'cuda'
            # Load a pretrained model using a specific torch dtype and device mapping
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                         torch_dtype=torch.bfloat16,
                                                         device_map=device) 
            # Load the corresponding tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Instantiate the QA_Extractor with the chosen model/tokenizer
            qa = QA_Extractor(model, tokenizer, experiment_name, logger)

        # Example usage if the model name includes 'c4ai'
        elif 'c4ai' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            qa = QA_Extractor(model, tokenizer, experiment_name, logger)

        # If the model name includes 'claude', create QA_Extractor specifically for that
        elif 'claude' in model_name:
            # 'params' presumably comes from a config file or global scope
            qa = QA_Extractor(experiment_name=experiment_name,
                              api_key=params['api_key'],
                              logger=logger)

        # If the model name includes 'hebert', create QA_Extractor using a question-answering pipeline
        elif 'hebert' in model_name:
            qa_model = pipeline('question-answering', model=model_name)
            qa = QA_Extractor(qa_model, save_path=save_path)
        
        # Update the methods string to indicate Q&A was enabled
        methods += '# Q&A #'
            
    # Regex extraction block
    if regex_extraction_flag:
        methods += '# Regex #'

    # Log which extraction methods are being used
    logger.info(f"The feature extraction started with the {methods}")
        
    # Check if the DB path is valid
    try:
        os.listdir(db_path)
    except:
        logger.error(f'There is an error in the path - "{db_path}"')
        return
    
    # If a tagged file path is provided and Q&A is enabled, perform Q&A feature extraction on that file
    if tagged_file_path is not None and qa is not None:
        result_df = qa.extract(tagged_file_path,
                               model_name,
                               db_flow=db_flow,
                               import_type=domain)
        logger.info(f"Finish to feature extraction on {tagged_file_path} tagged cases!!")
        return 
    
    # If db_flow is True, it means we want to walk through the directory structure for extraction
    if db_flow:
        for case_dir, dirs, files in os.walk(db_path):
            case_name = os.path.basename(case_dir)
            # Check if the directory name is not numeric -> indicates a case folder, not a year folder
            if not is_convertible_to_int(case_name): 
                if case_name == 'sentence_calssification' or case_name == '10.8_newsetfit':
                    continue
                if 'qa_features.csv' in files:
                    logger.info(f"qa_features.csv already exists in {case_name} folder case")
                    continue
                # Check if the relevant CSV files exist
                if 'sentence_predictions.csv' in files or 'sentence_tagging.csv' in files:
                    try:
                        # Run Q&A extraction if enabled
                        if qa_extraction_flag:
                            qa.extract(case_dir, model_name, db_flow=db_flow, import_type=domain)
                    except Exception as e:
                        continue
                    
                    # Run Regex extraction if enabled
                    if regex_extraction_flag:
                        RE = RegexExtractor(case_dir)
                        RE.extract(save_path, db_flow=db_flow)
                    
                    logger.info(f"Finish to feature extraction {case_name} case!!")
                else:
                    logger.info(f"preprocessing.csv doesn't exist in {case_name} folder case" )
    # If we're not walking the DB structure (db_flow=False) but regex_extraction is enabled, just do regex on a single file
    elif regex_extraction_flag:
        RE = RegexExtractor(tagged_file_path)
        RE.extract(save_path, db_flow=db_flow)
    
    logger.info(f'Finished extracting files, the appropriate files were saved in the cases in {db_path}!')

def main(params, domain):
    """
    Main function that sets up the logger, reads the DB path from params, and 
    initiates the feature extraction flow using the feature_extraction_flow function.
    """
    # Setup the logger for logging messages into a dedicated file
    logger = setup_logger(save_path=os.path.join(params['result_path'], 'logs'),
                          file_name='feature_extraction_test')

    # Construct the database path from the parameters
    db_path = params['db_path'].format(domain=domain)

    # Initiate the feature extraction flow with the configured parameters
    feature_extraction_flow(qa_extraction_flag=params['qa_extraction_flag'],
                            regex_extraction_flag=params['regex_extraction_flag'],
                            model_name=params['model_name'],
                            save_path=params['save_path'],
                            db_flow=params['db_flow'],
                            db_path=db_path,
                            logger=logger,
                            domain=domain)

if __name__ == '__main__':
    # Load main configuration parameters
    main_params = config_parser("", "main_config")
    # Extract the domain from the main configuration
    domain = main_params["domain"]
    
    # Parse the specific feature extraction parameters for the given domain
    params = config_parser("features_extraction", domain)
    
    # Run the main function with the parsed parameters
    main(params, domain)
