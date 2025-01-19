from datetime import datetime
import os
import sys
import pandas as pd

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from flows.inference.predict_sentence_cls import predict_2cls_lvl_flow
from utils.files import config_parser, setup_logger
from utils.sentence_classification import evaluate

import numpy as np
import os
import json
from datetime import datetime
import pandas as pd

def evaluation(params, logger, dir_path, tagged_path, eval_path, save_for_each_file=False):
    """
    This function goes through the tagging file, a legal contract and performs an evaluation
    """

    # Initialize accumulators for metrics
    metric_sums = {}
    file_counts = {}
    sentence_counts = {}
    case_count = []
    amount_values = {}
    # _, _, classifiers = predict_2cls_lvl_flow(tagged_path=tagged_path,
    #                                           eval_path=eval_path,
    #                                           classifiers_path=params['classifiers_path'],
    #                                           result_path=params['result_path'],
    #                                           logger=logger,
    #                                           )
    today = datetime.today()
    formatted_date = today.strftime("%d.%m")
    save_path = os.path.join(eval_path, formatted_date)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tagged_df = pd.read_csv(tagged_path)

    for file in os.listdir(dir_path):
        if 'ME' in file or 'SH' in file:
            df_predictions = pd.read_csv(os.path.join(dir_path, file,'sentence_predictions.csv'))
            df_probabilities = pd.read_csv(os.path.join(dir_path, file,'sentence_probabilities.csv'))
            save_path = os.path.join(save_path, f"{file}.json")
            tagged_df_1 = tagged_df[tagged_df['verdict'] == file]
            all_metrics = {}
            for column in tagged_df.columns:
                if column == 'Unnamed: 0' or column == 'text':
                    continue
                try:
                    column_upper = column.upper()
                    # if column == 'CONFESSION':
                    #     column_upper = 'CONFESSION_LVL2'
                    if column == 'verdict':
                        case_count.extend(list(tagged_df_1[column].values))
                        continue
                    amount_value = tagged_df_1[column].sum()
                    sentence_count = df_predictions[column].sum()
                    
                    

                    precision, recall, f1, auc_pr = evaluate(tagged_df=tagged_df,
                                                            df_predictions=df_predictions,
                                                            df_probabilities=df_probabilities,
                                                            column=column,
                                                            file=file,
                                                            label=column, 
                                                            logger=logger)
                    # Initialize metrics accumulators for this column if not already initialized
                    if column_upper not in metric_sums:
                        metric_sums[column_upper] = {'precision': 0, 'recall': 0, 'f1': 0, 'auc_pr': 0}
                        file_counts[column_upper] = 0
                        sentence_counts[column_upper] = 0
                        amount_values[column_upper] = 0

                    # Accumulate metrics
                    metric_sums[column_upper]['precision'] += precision
                    metric_sums[column_upper]['recall'] += recall
                    metric_sums[column_upper]['f1'] += f1
                    metric_sums[column_upper]['auc_pr'] += auc_pr
                    file_counts[column_upper] += 1
                    sentence_counts[column_upper] += sentence_count
                    amount_values[column_upper] += amount_value

                    metrics_data = {
                        "Precision": round(precision, 3),
                        "Recall": round(recall, 3),
                        "F1 Score": round(f1, 3),
                        "AUC-PR": round(auc_pr, 3),
                        "Amount True Labels": amount_value,
                        "Sentence Count": sentence_count
                    }
                    all_metrics[column] = metrics_data

                except Exception as e:
                    logger.error(f"Error evaluating column {column}: {e} in file {file}")

                    continue
            if save_for_each_file:
                with open(save_path, 'w') as file:
                    json.dump(all_metrics, file, indent=4)

                logger.info(f"All metrics saved to {save_path}")


    # Compute average metrics per column
    avg_metrics = {}
    for column_upper, metrics in metric_sums.items():
        if file_counts[column_upper] > 0:
            avg_metrics[column_upper] = {
                "Precision": round(metrics['precision'] / file_counts[column_upper], 3),
                "Recall": round(metrics['recall'] / file_counts[column_upper], 3),
                "F1 Score": round(metrics['f1'] / file_counts[column_upper], 3),
                "AUC-PR": round(metrics['auc_pr'] / file_counts[column_upper], 3),
                "Cases Count": file_counts[column_upper],
                "Amount True Labels": amount_values[column_upper],
                "Sentence Count": sentence_counts[column_upper]
            }

    # Save metrics for each label as separate JSON objects in a single file
    json_save_path = os.path.join(eval_path, formatted_date, "average_metrics.json")

    with open(json_save_path, 'w') as json_file:
        for label, metrics in avg_metrics.items():
            # Convert numpy types to native Python types
            def convert_to_python_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()  # Convert numpy int/float to native Python int/float
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()  # Convert numpy array to list
                else:
                    return obj

            # Convert metrics to Python types
            metrics = {key: convert_to_python_types(value) for key, value in metrics.items()}

            # Write the JSON object for this label
            json.dump({label: metrics}, json_file)
            json_file.write('\n')  # Add a newline between JSON objects

    logger.info(f"Metrics saved to {json_save_path}")




def main(params):
    logger = setup_logger(save_path=os.path.join(params['result_path'], 'logs'),
                          file_name='predict_sentence_cls_test')

    dir_path = params['dir_path'].format(domain=domain)
    tagged_path = params['tagged_path'].format(domain=domain)
    eval_path = params['eval_path'].format(domain=domain)

    evaluation(params,
               logger,
               dir_path = dir_path,
               tagged_path = tagged_path,
               eval_path = eval_path,
               save_for_each_file=params['save_for_each_file']
               )



if __name__ == '__main__':
    param = config_parser("", "main_config")
    domain = param["domain"]
    params = config_parser("evaluation_sentences", domain)
    main(params)