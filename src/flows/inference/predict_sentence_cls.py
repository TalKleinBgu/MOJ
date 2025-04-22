import logging
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)
from utils.files import load_json

from utils.files import setup_logger, config_parser, load_all_datasets, save_json
from utils.errors.predict_sentence_cls import *
from utils.sentence_classification import load_classifires, convert_dataset_to_dataframe
from scripts.sentence_classification.predict_sentence_cls.setfit_postprocessor import SetFitPostProcessor
from scripts.sentence_classification.setfit_version import compute_metrics
from sklearn.metrics import precision_recall_curve, average_precision_score, fbeta_score, roc_curve

map_lvl = {'RESPO': 'GENERAL_CIRCUM',
           'REGRET': 'GENERAL_CIRCUM',
           'CONFESSION': 'GENERAL_CIRCUM',
           'GENERAL_CIRCUM': 'GENERAL_CIRCUM'}

def find_best_threshold_F1(labels, probabilities):
    """
    Determine the best decision threshold for a classification model based on the precision-recall curve,
    selecting the threshold that maximizes the F2 score (weighted harmonic mean of precision and recall), 
    and return the PR-AUC (average precision score).

    Args:
        labels (array-like): True binary labels for the dataset.
        probabilities (array-like): Predicted probabilities for each class, typically from the model.
                                    It is assumed probabilities[:, 1] corresponds to the positive class.

    Returns:
        tuple: A tuple containing:
            - float: The threshold that maximizes the F2 score.
            - float: The PR-AUC score (average precision).
    """
    # Compute the precision-recall curve.
    precisions, recalls, thresholds = precision_recall_curve(labels, probabilities[:, 1])
    
    # The precision_recall_curve function returns precision and recall arrays with one extra value.
    # We compute the F1 score for each threshold (ignoring the last value in precision and recall).
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    # f1_scores = 5 * (precisions * recalls) / (4 * precisions + recalls + 1e-6)
    # Find the index of the threshold that gives the highest F1 score.
    best_idx = np.argmax(f1_scores)
    
    # Retrieve the corresponding best threshold.
    best_threshold = thresholds[best_idx]
    
    # # Compute the overall PR-AUC (average precision score).
    # # pr_auc = average_precision_score(labels, probabilities[:, 1])
    
    # # return best_threshold
    # # Compute the ROC curve: fpr, tpr, and thresholds.
    # fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
    
    # # Calculate Youden's J statistic for each threshold.
    # J = tpr - fpr
    
    # # Find the index of the threshold that maximizes J.
    # best_idx = np.argmax(J)
    
    # # Retrieve the corresponding best threshold.
    # best_threshold = thresholds[best_idx]
    
    
    return best_threshold



    
def predict_from_datasets(datasets_dict, classifiers, logger, save_path,model_name, with_clf, ground_truth,ground_truth_path, best_threshold = 0.999):
    ground_truth_df = pd.read_csv(ground_truth_path)

    for label in datasets_dict.keys():
        if label == 'CIRCUM_OFFENSE' or label == 'GENERAL_CIRCUM' or label == 'reject':
            continue
        model = classifiers[label]
        probabilities = []
        # best_threshold = find_best_threshold
        # if label in map_lvl.keys():
        #     first_lvl_model = classifiers['GENERAL_CIRCUM']
        # else: 
        #     first_lvl_model = classifiers['CIRCUM_OFFENSE']
        first_lvl_model = classifiers['CIRCUM_OFFENSE']

        # reject_model = classifiers['reject']
        threshold = first_lvl_model[1]
        for sentence in datasets_dict[label]['text']:
            try:
                # probabilities.append(model[0].predict_proba(sentence))
                # if int(reject_model[0].predict(sentence)) == 0:
                if with_clf:
                    first_lvl_pred = first_lvl_model[0].predict(sentence)
                    # first_lvl_pred1 = first_lvl_model[0].predict_proba(sentence)
                    # first_lvl_pred1 = first_lvl_pred1[1].detach().cpu().numpy()
                    # threshold = first_lvl_model[1]
                    # if first_lvl_pred1 >= threshold:
                    # if int(reject_model[0].predict(sentence)) == 0:
                    if int(first_lvl_pred) == 1:
                        probabilities.append(model[0].predict_proba(sentence))
                    else:
                        probabilities.append((0,0))
                    # else:
                    #     probabilities.append((0,0))
                elif ground_truth:
                    first_lvl_pred  = ground_truth_df[ground_truth_df['text'] == sentence]['CIRCUM_OFFENSE'].values[0]
                    if int(first_lvl_pred) == 1:
                        probabilities.append(model[0].predict_proba(sentence))
                    else:
                        probabilities.append((0,0))
                else:
                    probabilities.append(model[0].predict_proba(sentence))
                # else:
                #     probabilities.append((0,0))
            except Exception as e:
                print(e)
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
        conf_matrix = confusion_matrix(datasets_dict[label]['label'], predictions)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        logger.info(f'Metrics to {label} is: {metrics}')
        save_label_path = os.path.join(save_path, label, model_name)
        os.makedirs(save_label_path, exist_ok=True)
                # Save the confusion matrix plot to the specified path
        conf_matrix_path = os.path.join(save_label_path, 'confusion_matrix.png')
        plt.savefig(conf_matrix_path)
        plt.show()
        plt.clf()  # Clear the figure to avoid overlapping plots

        plt.close()  # Close the figure to free resources
        save_json(metrics, os.path.join(save_label_path, 'metrics.json'))
            # Identify misclassified examples for analysis
        labels = np.array(datasets_dict[label]['label'])
        errors = (predictions != labels)
        error_df = pd.DataFrame({
            'sentence': datasets_dict[label]['text'],                          # Original text
            'true_label': labels,                          # Ground-truth labels
            'predicted_label': predictions,             # Model predictions
            'probabilities': second_probabilities,  # Probability of the positive class
        })
        # Filter and sort misclassified examples by confidence level (descending)
        error_df = error_df[errors].sort_values(by='probabilities', ascending=False)
        error_df.to_csv(os.path.join(save_label_path, 'error_df.csv'), index=False)

            
            
def predict_2cls_lvl_flow(case_dir_path:str = None, test_set_path:str = None, classifiers:dict = None, classifiers_path:str = None, 
                          logger:logging.Logger = None, first_level_labels:list = None, first_lvl_cls_path:str = None,
                          second_level_labels:list = None, threshold:object = None, tagged_path: str = None,
                          eval_path:str = None, experimant_name:str = '',model_names:list = None, with_clf:bool = False, ground_truth:bool = False, ground_truth_path:str = None,
                          threshold_method:str = 'best'):

                                                                                    
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
        save_path = os.path.join(eval_path, 'predictions_with_clf_50')
    elif ground_truth:
        save_path = os.path.join(eval_path, 'predictions_with_ground_truth_50')
    else:
        save_path = os.path.join(eval_path, 'predictions_no_clf_999')

        # save_path = eval_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for model_name in model_names:
        if classifiers is None:
                classifiers, first_level_labels, second_level_labels = load_classifires(eval_path=eval_path,
                                                                                        classifiers_path= classifiers_path,
                                                                                        first_level_labels=first_level_labels,
                                                                                        second_level_labels=second_level_labels,
                                                                                        logger=logger,
                                                                                        model_name=model_name)
        all_labels = first_level_labels + second_level_labels
        test_dataset = load_all_datasets(test_set_path, all_labels)
        # test_dataset = load_all_datasets(test_set_path, second_level_labels)
        # all_labels = first_level_labels + second_level_labels
        # for label in all_labels:
        #     classifiers[label] = change_threshold(threshold_method, classifiers[label],test_dataset,label,eval_path,first_level_labels)
        predict_from_datasets(test_dataset, classifiers, logger, save_path, model_name, with_clf, ground_truth,ground_truth_path)
        classifiers = None

    


def change_threshold(threshold_method,classifier,test_dataset,label,eval_path,first_level_labels):
    # Extract sentences and labels from the test dataset
    sentences = test_dataset[label]['text']
    # sentences = change_data_format(sentences, label)
    labels = test_dataset[label]['label']

    # Get predicted probabilities for the sentences
    with torch.no_grad():
        probabilities = classifier[0].predict_proba(sentences)
    try:
        # Ensure probabilities are in a NumPy array format (handle potential Tensor output)
        probabilities = probabilities.numpy()
    except:
        probabilities = probabilities.cpu().detach().numpy()

    if threshold_method == 'F1':
        best_threshold = find_best_threshold_F1(labels, probabilities)
        final_classifier = (classifier[0], best_threshold)
    else:
        final_classifier = (classifier[0], classifier[1])

    eval_path_first = os.path.join(eval_path, 'first_labels')
    eval_path_second = os.path.join(eval_path, 'second_labels')
        
    first_level_path = os.path.join(eval_path, "first_labels")
    second_level_path = os.path.join(eval_path, "second_labels")

    if label in first_level_labels:
        label_path = os.path.join(first_level_path, f"{label}")
        model_path = os.path.join(label_path, 'dictabert')
        model_path = os.path.join(model_path, 'model')
        try:
            label_path = os.path.join(eval_path_first, f"{label}")
            eval_path = os.path.join(label_path, 'dictabert')
            metrics = load_json(os.path.join(eval_path, 'metric.json'))
            metrics['best_threshold'] = float(best_threshold)
            save_json(metrics, os.path.join(eval_path, 'metric.json'))
        except:
            pass
    else:
        label_path = os.path.join(second_level_path, f"{label}")
        model_path = os.path.join(label_path, 'dictabert')
        model_path = os.path.join(model_path, 'model')
        try:
            label_path = os.path.join(eval_path_second, f"{label}")
            eval_path = os.path.join(label_path, 'dictabert')
            metrics = load_json(os.path.join(eval_path, 'metric.json'))
            metrics['best_threshold'] = float(best_threshold)
            save_json(metrics, os.path.join(eval_path, 'metric.json'))
        except:
            pass


    return final_classifier
    # best_threshold = 0.99
    # eval_path_first = os.path.join(eval_path, 'first_labels')
    # eval_path_second = os.path.join(eval_path, 'second_labels')
    # for label in first_level_labels:
    #     if label.lower() == 'yaml' in label :
    #         continue
    #     label_path = os.path.join(eval_path_first, f"{label}")
    #     model_path = os.path.join(label_path, model_name)
    #     try:
    #         label_path = os.path.join(eval_path_first, f"{label}")
    #         eval_path = os.path.join(label_path, model_name)
    #         metrics = load_json(os.path.join(eval_path, 'metric.json'))
    #         metrics['best_threshold'] = best_threshold
    #         save_json(metrics, os.path.join(eval_path, 'metric.json'))
    #     except Exception as e:
    #         print(e)
        
    # for label in second_level_labels:
    #     if label.lower() == 'yaml' in label :
    #         continue
    #     label_path = os.path.join(eval_path_second, f"{label}")
    #     model_path = os.path.join(label_path, model_name)
    #     try:
    #         label_path = os.path.join(eval_path_second, f"{label}")
    #         eval_path = os.path.join(label_path, model_name)
    #         metrics = load_json(os.path.join(eval_path, 'metric.json'))
    #         metrics['best_threshold'] = best_threshold
    #         save_json(metrics, os.path.join(eval_path, 'metric.json'))
    #     except Exception as e:
    #         print(e)

def main(param, domain):
    logger = setup_logger(save_path=os.path.join(param['result_path'], 'logs'),
                          file_name='predict_sentence_cls_test')
    
    
    classifiers_path = param['classifiers_path'].format(experiment_name=param['experiment_name'])    
    eval_path = param['eval_path'].format(experiment_name=param['experiment_name'])
    model_names = param['model_names']
    with_clf = param['with_clf']
    ground_truth = param['ground_truth']
    ground_truth_path = param['ground_truth_path']
    predict_2cls_lvl_flow(eval_path=eval_path,
                          classifiers_path=classifiers_path,
                          first_level_labels=param['first_level_labels'],
                          second_level_labels=param['second_level_labels'],
                          test_set_path=param['test_set_path'],
                          logger=logger,
                          model_names=model_names,
                          with_clf=with_clf,
                          ground_truth=ground_truth,
                          ground_truth_path=ground_truth_path,
                          threshold_method=param['threshold_method'],
                          )


if __name__ == '__main__':
    main_params = config_parser("", "main_config")    
    domain = main_params["domain"] 
    # Parse the training configuration parameters for the specific task
    params = config_parser("predict", domain)
    
    main(params, domain)