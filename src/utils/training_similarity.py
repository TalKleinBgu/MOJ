
import re
import joblib
import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import json

def save_model(model, model_type, result_path, logger):
    model_file = f"{model_type}_model.pkl"
    save_path = os.path.join(result_path, model_file)
    joblib.dump(model, save_path)
    logger.info(f"Save {model_type} model in {save_path}!")
            
def plot_roc_curve(y_test, y_probs, output_path, extraction_type):
        # Plot ROC curve
        auc_score = roc_auc_score(y_test, y_probs[:, 1])

        fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {extraction_type}')
        plt.legend(loc="lower right")
        file_dave_path = os.path.join(output_path, f"{extraction_type}_roc_curve.png")
        plt.savefig(file_dave_path)

        plt.show()


# def save_report(output_path, model_type, extraction_type, evaluation_metrics, confusion_matrix, auc_score):
#     """
#     Save evaluation report to a text file.
#     :param output_path: Path where the report will be saved.
#     :param model_type: Type of the model used.
#     :param extraction_type: Type of feature extraction used.
#     :param evaluation_metrics: Dictionary containing evaluation metrics.
#     :param confusion_matrix: Confusion matrix.
#     :param auc_score: AUC score.
#     """
#     # Create a report string
#     report = f"Model Evaluation Report\n\n"
#     report += f"Model Type: {model_type}\n"
#     report += f"Extraction Type: {extraction_type}\n"
#     report += f"=========================\n"
#     for metric, value in evaluation_metrics.items():
#         report += f"{metric}: {value}\n"
#     report += f"=========================\n"
#     report += f"Confusion Matrix:\n{confusion_matrix}\n"
#     report += f"AUC Score: {auc_score}\n"

#     # Save report to file
#     save_path = os.path.join(output_path, model_type)
#     os.makedirs(save_path, exist_ok=True)

#     report_file_path = os.path.join(save_path, 'evaluation_report.txt')
#     with open(report_file_path, 'w') as report_file:
#         report_file.write(report)

#     print("Evaluation report saved to:", report_file_path)

def save_report(output_path, model_type, extraction_type, evaluation_metrics, confusion_matrix, auc_score):
    """
    Save evaluation report to a JSON file.
    :param output_path: Path where the report will be saved.
    :param model_type: Type of the model used.
    :param extraction_type: Type of feature extraction used.
    :param evaluation_metrics: Dictionary containing evaluation metrics.
    :param confusion_matrix: Confusion matrix.
    :param auc_score: AUC score.
    """
    # Create a dictionary for the report
    report = {
        "Model Type": model_type,
        "Extraction Type": extraction_type,
        "Evaluation Metrics": evaluation_metrics,
        "Confusion Matrix": confusion_matrix.tolist(),  # Convert NumPy array to list
        "AUC Score": auc_score
    }

    # Save the report to a JSON file
    save_path = output_path
    os.makedirs(save_path, exist_ok=True)

    report_file_path = os.path.join(save_path, 'evaluation_report.json')
    with open(report_file_path, 'w') as report_file:
        json.dump(report, report_file)

    print("Evaluation report saved to:", report_file_path)

    
    
# Example function to read the prompt and substitute the placeholders
def read_and_fill_prompt(file_path, case1_properties, case2_properties):
    with open(file_path, 'r') as file:
        prompt = file.read()

    # Replace placeholders with actual case vectors
    prompt_filled = prompt.replace('{{CASE1_PROPERTIES}}', str(case1_properties))
    prompt_filled = prompt_filled.replace('{{CASE2_PROPERTIES}}', str(case2_properties))
    
    return prompt_filled

def similarity_parser(text):
    match = re.search(r'<similarity_score>(.*?)</similarity_score>', text, re.DOTALL)
    if match:
        # Extract the number (group 1) and convert it to an integer
        number = int(match.group(1))
        return number
