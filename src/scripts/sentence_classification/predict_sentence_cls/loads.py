import logging
import os
import sys

from utils.files import load_json
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, pred_sentencing_path)

from model_handler import Classifier

def models_name_extraction(classifiers_path:str = None, level:str = None):
    models_name = []
    # classifiers_path = os.path.join(classifiers_path, f'{level}_level')
    for model_name in os.listdir(classifiers_path):
        models_name.append(model_name)
    return models_name

def load_all_classifies(eval_path, classifiers_path:str = None, first_level_labels:list = None,
                        second_level_labels:list = None, logger:logging.Logger = None,model_name:str = ''):
    """
    Loads all classifiers (both first-level and second-level) from the specified model paths.
    Returns:
        dict: A dictionary containing all loaded classifiers.
        ** if one of the label list (first/second) is None -> load from folder models the models name and 
           return this lists
    """
    
    classifiers = {}
    return_labels = False
    first_level_path = os.path.join(classifiers_path, "first_labels")
    second_level_path = os.path.join(classifiers_path, "second_labels")

    eval_path_first = os.path.join(eval_path, 'first_labels')
    eval_path_second = os.path.join(eval_path, 'second_labels')
        
    # regect_path = os.path.join(first_level_path, "reject")
    # # metrics = load_json(os.path.join(regect_path, 'metric.json'))
    # regect_path = os.path.join(regect_path, load_models_name(regect_path))
    # classifiers['reject'] = Classifier(regect_path, "reject").load_model(logger)
    
    # if first_level_labels is None:
    #     first_level_labels = models_name_extraction(classifiers_path=first_level_path)
    #     return_labels = True
        
    for label in first_level_labels:
        if label.lower() == 'yaml' in label :
            continue
        label_path = os.path.join(first_level_path, f"{label}")
        model_path = os.path.join(label_path, model_name)
        model_path = os.path.join(model_path, 'model')
        try:
            label_path = os.path.join(eval_path_first, f"{label}")
            eval_path = os.path.join(label_path, model_name)
            metrics = load_json(os.path.join(eval_path, 'metric.json'))
            classifiers[label] = (Classifier(model_path, label).load_model(logger), metrics['best_threshold'])
        except:
            classifiers[label] = (Classifier(model_path, label).load_model(logger), 0.95)

    for label in second_level_labels:
        if label.lower() == 'yaml' in label :
            continue
        label_path = os.path.join(second_level_path, f"{label}")
        model_path = os.path.join(label_path, model_name)
        model_path = os.path.join(model_path, 'model')
        try:
            label_path = os.path.join(eval_path_second, f"{label}")
            eval_path = os.path.join(label_path, model_name)
            metrics = load_json(os.path.join(eval_path, 'metric.json'))
            classifiers[label] = (Classifier(model_path, label).load_model(logger), metrics['best_threshold'])
        except:
            classifiers[label] = (Classifier(model_path, label).load_model(logger), 0.95)
    # if second_level_labels is None:
    #     second_level_labels = models_name_extraction(classifiers_path=classifiers_path,
    #                                                  level='second')
    #     return_labels = True
        
    if return_labels:
        return classifiers, first_level_labels, second_level_labels
    return classifiers


def load_models_name(path):
    """
    Helper function to get the name of the model from a given directory.
    Args:
        path (str): The directory path where the models are stored.
    Returns:
        str: The name of the model.
    """

    list_dir1 = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    path = os.path.join(path, list_dir1[0])
    # list_dir2 = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return f'{list_dir1[0]}'
