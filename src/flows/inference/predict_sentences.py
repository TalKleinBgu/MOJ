import logging
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import torch

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import setup_logger, config_parser, load_all_datasets, save_json
from utils.errors.predict_sentence_cls import *
from utils.sentence_classification import load_classifires

from scripts.sentence_classification.predict_sentence_cls.setfit_postprocessor import SetFitPostProcessor
from scripts.sentence_classification.setfit_version import compute_metrics


def predict_from_datasets(datasets_dict, classifiers,first_level_labels, second_level_labels, logger, save_path, best_threshold = 0.999):
    # empty df with labels as columns
    labels = ['text'] + first_level_labels + second_level_labels
    binary_predictions_df = pd.DataFrame(columns=labels)
    probability_predictions_df = pd.DataFrame(columns=labels)
    #remove duplicates text
    datasets_dict = datasets_dict.drop_duplicates(subset=['text'])
    len_data = len(datasets_dict['text'])
    for sentence in datasets_dict['text']:
        binary_predictions = []
        probability_predictions = []
        binary_predictions.append(sentence)
        probability_predictions.append(sentence)

        for label in first_level_labels:
            model = classifiers[label]
            try:
                probabilities = model[0].predict_proba(sentence)
                probabilities = probabilities[1].detach().cpu().numpy() if isinstance(probabilities[1], torch.Tensor) else probabilities[1]
                binary_predictions.append((probabilities >= model[1]).astype(int))
                probability_predictions.append(probabilities)

            except Exception as e:
                logger.error(f"Error predicting for label {label}: {e}")
                binary_predictions.append(0)
                probability_predictions.append(0)

        for label in second_level_labels:
            model = classifiers[label]
            try:
                probabilities = model[0].predict_proba(sentence)
                probabilities = probabilities[1].detach().cpu().numpy() if isinstance(probabilities[1], torch.Tensor) else probabilities[1]
                binary_predictions.append((probabilities >= model[1]).astype(int))
                probability_predictions.append(probabilities)
            except Exception as e:
                logger.error(f"Error predicting for label {label}: {e}")
                binary_predictions.append(0)
                probability_predictions.append(0)
                

        binary_predictions_df = pd.concat([binary_predictions_df, pd.DataFrame([binary_predictions], columns=labels)], ignore_index=True)
        probability_predictions_df = pd.concat([probability_predictions_df, pd.DataFrame([probability_predictions], columns=labels)], ignore_index=True)

    binary_predictions_path = os.path.join(save_path, 'sentence_predictions.csv')
    probability_predictions_path = os.path.join(save_path, 'sentence_probabilities.csv')

    binary_predictions_df.to_csv(binary_predictions_path, index=False)
    probability_predictions_df.to_csv(probability_predictions_path, index=False)

    logger.info(f"Binary predictions saved to {binary_predictions_path}")
    logger.info(f"Probabilities saved to {probability_predictions_path}")


            
            
            
def predict_2cls_lvl_flow(db_path:str = None, test_set_path:str = None, classifiers:dict = None, classifiers_path:str = None, 
                          logger:logging.Logger = None, first_level_labels:list = None,
                          second_level_labels:list = None, threshold:object = None, tagged_path: str = None,
                          eval_path:str = None, experimant_name:str = ''):

                                                                                    
    """
    In this function, it is possible move on case directory,
    and go through each sentence in the preprocess csv to predict the label (2 level),
    or move on excel tagged file perform an evaluation

    Args:
        case_dir_path: path of the verdict in db
        test_set_path: the output of preprocess flow
        second/first_level_labels: list of labels the the user want to predict 

    Returns:
        str: The path to the saved predicted CSV file.
    """
    
    if classifiers is None:
            classifiers, first_level_labels, second_level_labels = load_classifires(eval_path=eval_path,
                                                                                    classifiers_path= classifiers_path,
                                                                                    first_level_labels=first_level_labels,
                                                                                    second_level_labels=second_level_labels,
                                                                                    logger=logger)
    cases = os.listdir(db_path)
    for case in cases:
        case_dir_path = os.path.join(db_path, case)

        test_set_path = os.path.join(case_dir_path, 'preprocessing.csv')
        save_path = case_dir_path
        test_dataset = pd.read_csv(test_set_path)
        # test_dataset = load_all_datasets(test_set_path, second_level_labels)
        predict_from_datasets(test_dataset, classifiers, first_level_labels, second_level_labels, logger, save_path)
        logger.info(f'Finish predict case {case}')


def main(param, domain):
    logger = setup_logger(save_path=os.path.join(param['result_path'], 'logs'),
                          file_name='predict_sentence_cls_test')
    
    classifiers_path = param['classifiers_path'].format(experiment_name=param['experiment_name'],model_name=param['model_name'])    
    eval_path = param['eval_path'].format(experiment_name=param['experiment_name'],date=datetime.today().strftime("%d.%m"))

    predict_2cls_lvl_flow(db_path= params['db_path'],
                        eval_path=eval_path,
                        classifiers_path=classifiers_path,
                        first_level_labels=param['first_level_labels'],
                        second_level_labels=param['second_level_labels'],
                        logger=logger
                        )        



if __name__ == '__main__':
    main_params = config_parser("", "main_config")    
    domain = main_params["domain"] 
    # Parse the training configuration parameters for the specific task
    params = config_parser("predict", domain)
    
    main(params, domain)