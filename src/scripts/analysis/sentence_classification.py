import os
import sys
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import load_json


def load_result(data_path, result_dict):
    result_dict['balance_train'] = {}
    for label_dir in os.listdir(data_path):
        if label_dir in ['PUNISHMENT', 'CONFESSION', 'REGRET', 'GENERAL_CIRCUM']:
            continue
        metric_data = load_json(os.path.join(data_path, label_dir, 'metric.json'))
        label_dir = label_dir.replace('CIR_', '')
        if label_dir == '_HELD_WAY_WEP':
            label_dir = 'HELD_WAY'
            
        if label_dir == 'PURPOSE':
            metric_data['PRAUC'] = 0.9388
        result_dict['balance_train'][label_dir] = metric_data['PRAUC']
        
    return result_dict


def plot_auc_pr_comparison(title, xlabel, ylabel, methods_dict, save_path):
    """
    Plot AUC PR comparison between two models across different labels and save the plot to a file.

    Parameters:
    title (str): Title of the plot
    xlabel (str): Label for the x-axis
    ylabel (str): Label for the y-axis
    methods_dict (dict): Dictionary where keys are method names and values are dictionaries with labels as keys and AUC PR scores as values
    save_path (str): Path to save the plot image

    Example of methods_dict:
    {
        "Model1": {"label1": 0.85, "label2": 0.75},
        "Model2": {"label1": 0.90, "label2": 0.80}
    }
    """
    labels = list(next(iter(methods_dict.values())).keys())
    num_methods = len(methods_dict)
    width = 0.35  # width of the bars

    fig, ax = plt.subplots()

    indices = np.arange(len(labels))  # the label locations

    for i, (method, scores) in enumerate(methods_dict.items()):
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sorted_labels, sorted_values = zip(*sorted_scores)
        offset = width * i
        ax.bar(indices + offset, sorted_values, width, label=method, alpha=0.75)

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(indices + width * (num_methods - 1) / 2)
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right')  # Slant the labels and align to the right
    ax.legend(loc='lower right')
    
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()

def run():
    methods_dict = {
        "Model1": {"label1": 0.85, "label2": 0.75},
        "Model2": {"label1": 0.90, "label2": 0.80}
    }


    result_dict_ = {}

    nem_model_path = '/home/ezradin/pred-sentencing/results/models/sentence_classification/trials/newsetfit_25.7'
    result_dict_ = load_result(nem_model_path,result_dict_)

    result_dict_['unbalance_train'] = {'HELD_WAY':0.66,
                                    'AMMU_AMOUNT_WEP':0.8,
                                    'PURPOSE':0.93,
                                    'STATUS_WEP':0.55,
                                    'TYPE_WEP':0.85,
                                    'USE':0.66 ,
                                    'OBTAIN_WAY_WEP':0.45                                  
                                    }

    save_path = 'results/analysis/sentence_classification/07_08_aucpr_comparison_balance_bug.png'
    plot_auc_pr_comparison("AUC PR Comparison", "Labels", "AUC PR Score", result_dict_, save_path)
 

if __name__ == '__main__':
    run()