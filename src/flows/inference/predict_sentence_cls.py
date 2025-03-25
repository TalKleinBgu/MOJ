import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import setup_logger, config_parser, load_all_datasets, save_json
from utils.errors.predict_sentence_cls import *
from utils.sentence_classification import load_classifires, convert_dataset_to_dataframe

from scripts.sentence_classification.predict_sentence_cls.setfit_postprocessor import SetFitPostProcessor
from scripts.sentence_classification.setfit_version import compute_metrics

map_lvl = {'RESPO': 'CONFESSION',
           'REGRET': 'CONFESSION',
           'CONFESSION': 'CONFESSION'}

def predict_from_datasets(datasets_dict, classifiers, logger, save_path,model_name, with_clf, best_threshold = 0.999):
    for label in datasets_dict.keys():
        model = classifiers[label]
        probabilities = []
        # if label in map_lvl.keys():
        #     first_lvl_model = classifiers['CONFESSION']
        # else: 
        #     first_lvl_model = classifiers['CIRCUM_OFFENSE']
        first_lvl_model = classifiers['CIRCUM_OFFENSE']
        # reject_model = classifiers['reject']
        for sentence in datasets_dict[label]['text']:
            try:
                # probabilities.append(model[0].predict_proba(sentence))
                # if int(reject_model.predict(sentence)) == 0:
                if with_clf:
                    first_lvl_pred = first_lvl_model[0].predict(sentence)
                    if int(first_lvl_pred) == 1:
                        probabilities.append(model[0].predict_proba(sentence))
                    else:
                        probabilities.append((0,0))
                else:
                    probabilities.append(model[0].predict_proba(sentence))

                # else:
                    # probabilities.append((0,0))
            except :
                probabilities.append((0,0))
            
        second_probabilities = [
                                prob[1].detach().cpu().numpy() if isinstance(prob[1], torch.Tensor) else prob[1] 
                                for prob in probabilities
                                 ]
        predictions = (np.array(second_probabilities) >= model[1]).astype(int)
        metrics = compute_metrics(predictions=predictions,
                        labels=datasets_dict[label]['label'],
                        probabilities=second_probabilities
                        )
        logger.info(f'Metrics to {label} is: {metrics}')
        save_label_path = os.path.join(save_path, label, model_name)
        os.makedirs(save_label_path, exist_ok=True)
        save_json(metrics, os.path.join(save_label_path, 'metrics.json'))
            
            
def predict_2cls_lvl_flow(case_dir_path:str = None, test_set_path:str = None, classifiers:dict = None, classifiers_path:str = None, 
                          logger:logging.Logger = None, first_level_labels:list = None, first_lvl_cls_path:str = None,
                          second_level_labels:list = None, threshold:object = None, tagged_path: str = None,
                          eval_path:str = None, experimant_name:str = '',model_name:str = '', with_clf:bool = False):

                                                                                    
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
    
    # if (case_dir_path is None) and (tagged_path is None):
        # case_dir_path_error(logger)
        
    if case_dir_path:
        if test_set_path is None:
            test_set_path = os.path.join(case_dir_path, 'preprocessing.csv')
        save_path = case_dir_path
    elif with_clf:
        save_path = os.path.join(eval_path, 'predictions')
    else:
        save_path = os.path.join(eval_path, 'predictions_no_clf')
        # save_path = eval_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
            
    if classifiers is None:
            classifiers, first_level_labels, second_level_labels = load_classifires(eval_path=eval_path,
                                                                                    classifiers_path= classifiers_path,
                                                                                    first_level_labels=first_level_labels,
                                                                                    second_level_labels=second_level_labels,
                                                                                    logger=logger,
                                                                                    model_name=model_name)
    test_dataset = load_all_datasets(test_set_path, second_level_labels)
    predict_from_datasets(test_dataset, classifiers, logger, save_path, model_name, with_clf)

    



def main(param, domain):
    logger = setup_logger(save_path=os.path.join(param['result_path'], 'logs'),
                          file_name='predict_sentence_cls_test')
    
    
    classifiers_path = param['classifiers_path'].format(experiment_name=param['experiment_name'])    
    eval_path = param['eval_path'].format(experiment_name=param['experiment_name'])
    model_name = param['model_name']
    with_clf = param['with_clf']
    predict_2cls_lvl_flow(eval_path=eval_path,
                          classifiers_path=classifiers_path,
                          first_level_labels=param['first_level_labels'],
                          second_level_labels=param['second_level_labels'],
                          test_set_path=param['test_set_path'],
                          logger=logger,
                          model_name=model_name,
                          with_clf=with_clf
                          )


if __name__ == '__main__':
    main_params = config_parser("", "main_config")    
    domain = main_params["domain"] 
    # Parse the training configuration parameters for the specific task
    params = config_parser("predict", domain)
    
    main(params, domain)