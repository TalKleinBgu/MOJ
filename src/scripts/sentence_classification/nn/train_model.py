import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
from tqdm.auto import tqdm

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from data_hendler import DataHandler
from utils.files import config_parser
from utils.models import save_model
from scripts.sentence_classification.nn.evaluate_model import evaluate_model

import torch.nn as nn
from transformers import AutoModel


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels, use_conv_layers):
        super(CustomModel, self).__init__()
        if model_name == 'dicta-il/dictalm-7b':
            self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)  # Regularization with Dropout
        self.use_conv_layers = use_conv_layers
        self.embedding_dim = self.bert.config.hidden_size

        if self.use_conv_layers:
            self.conv1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.classifier = nn.Linear(64, num_labels)
        else:
            self.classifier = nn.Linear(self.embedding_dim, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        if self.use_conv_layers:
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
            x = last_hidden_state.transpose(1, 2)  # (batch_size, hidden_size, seq_length)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x).squeeze(-1) 
        else:
            x = outputs[1]
            
        x = self.dropout(x)
        logits = self.classifier(x)
        
        if labels is not None:
            return {"logits": logits, "labels": labels}
        return {"logits": logits}
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        model_to_save = self.bert
        model_to_save.save_pretrained(save_directory)
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, 'classifier.pt'))


class ModelTrainer:
    def __init__(self, params):
        self.params = params
        self.data_handler = DataHandler(params['file_paths'], params['model_name'], params['batch_size'], classification_type=params['classification_type'])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def train_model(self, train_loader, val_loader, model, optimizer, lr_scheduler, num_epochs, patience, classification_type, label_name):
        model.train()
        progress_bar = tqdm(range(num_epochs * len(train_loader)))

        best_loss = float('inf')
        epochs_no_improve = 0

        if classification_type == "multi-class":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs['logits']

                if classification_type == "multi-class":
                    loss = criterion(logits, batch['labels'])
                else:
                    loss = criterion(logits.view(-1), batch['labels'].float())
                loss.backward()
                total_loss += loss.item()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

            val_loss = self.validate_model(val_loader, model, criterion, classification_type)
            print(f"Validation Loss: {val_loss:.4f}")
            # Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                # Save the best model
                model_save_path = save_model(model, self.params, label_name)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return model_save_path

    def validate_model(self, val_loader, model, criterion, classification_type):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs['logits']
                if classification_type == "multi-class":
                    loss = criterion(logits, batch['labels'])
                else:
                    loss = criterion(logits.view(-1), batch['labels'].float())
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        model.train()
        return avg_loss

    def train_and_evaluate_model(self, train_loader, val_loader, label_mapping, label_name, perform_dropout):
        if perform_dropout:
            model = CustomModel(self.params['model_name'], self.params['num_labels'],
                                self.params['use_conv_layers'])
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.params['model_name'], num_labels=self.params['num_labels'], ignore_mismatched_sizes=True)
            model.classifier = torch.nn.Linear(model.config.hidden_size, self.params['num_labels'])

        optimizer = AdamW(model.parameters(), lr=float(self.params['learning_rate']))
        num_training_steps = self.params['num_epochs'] * len(train_loader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        model.to(self.device)

        save_path = self.train_model(train_loader, val_loader, model, optimizer, lr_scheduler, self.params['num_epochs'], self.params['patience'], self.params['classification_type'], label_name)
        metrics, label_names = evaluate_model(val_loader, model, self.device, len(label_mapping), label_mapping, save_path, self.params['classification_type'])

        if self.params['classification_type'] == "multi-class":
            for i in range(len(label_mapping)):
                print(f"Label {i} ({label_names.get(i, 'Unknown')}):")
                print(f"  Precision: {metrics['precision'][i]:.4f}")
                print(f"  Recall: {metrics['recall'][i]:.4f}")
                print(f"  F1 Score: {metrics['f1'][i]:.4f}")
                print(f"  AUC-PR: {metrics['auc_pr'][i]:.4f}")
        else:
            binary_metrics = metrics['binary_classification']
            print(f"Binary Classification Metrics of {label_name}:")
            print(f"  Precision: {binary_metrics['precision']:.4f}")
            print(f"  Recall: {binary_metrics['recall']:.4f}")
            print(f"  F1 Score: {binary_metrics['f1']:.4f}")
            print(f"  AUC-PR: {binary_metrics['auc_pr']:.4f}")

    def run(self):
        if self.params['classification_type'] == "multi-class":
            combined_df, label_mapping = self.data_handler.load_and_combine_csvs(self.params['file_paths'])
            self.data_handler.combined_df = combined_df
            self.data_handler.label_mapping = label_mapping
            train_loader, val_loader = self.data_handler.create_dataloaders(combined_df)
            self.train_and_evaluate_model(train_loader, val_loader, self.data_handler.label_mapping, 
                                          "multi_class", self.params['perform_dropout'])
        else:
            label_dfs, label_mapping = self.data_handler.load_and_combine_csvs(self.params['file_paths'])
            for label, df in label_dfs.items():
                print(f"\nTraining model for label: {label}")
                self.data_handler.combined_df = df
                self.data_handler.label_mapping = label_mapping
                train_loader, val_loader = self.data_handler.create_dataloaders(df, label=label)
                # self.train_and_evaluate_model(train_loader, val_loader, label_mapping, label, self.params['perform_dropout'])


if __name__ == '__main__':
    params = config_parser("nn_config")
    trainer = ModelTrainer(params)
    trainer.run()
