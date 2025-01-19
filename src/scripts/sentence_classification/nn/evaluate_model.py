from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import torch

from utils.files import save_json


def evaluate_model(val_loader, model, device, num_labels, label_mapping, save_path, classification_type):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs['logits']
        if classification_type == "multi-class":
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        else:  # binary classification
            probs = torch.sigmoid(logits).squeeze()
            predictions = (probs > 0.5).long()

        all_labels.extend(batch["labels"].cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    if classification_type == "multi-class":
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, labels=range(num_labels))
        auc_pr = [average_precision_score([1 if l == i else 0 for l in all_labels], [p[i] for p in all_probs]) for i in range(num_labels)]
        label_names = {v: k for k, v in label_mapping.items()}

        metrics = {
            label_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'auc_pr': auc_pr[i]
            } for i in range(num_labels)
        }
    else:  # binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        auc_pr = average_precision_score(all_labels, all_probs)
        label_names = {1: 'positive', 0: 'negative'}

        metrics = {
            'binary_classification': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_pr': auc_pr
            }
        }
        
    results_save_path = f"{save_path}/evaluation_metrics.json"
    save_json(metrics, results_save_path)
    
    return metrics, label_names