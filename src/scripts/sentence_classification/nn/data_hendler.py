import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class DataHandler:
    def __init__(self, file_paths, model_name="distilbert-base-uncased",
                 batch_size=16, seed=42, classification_type="multi-label"):
        self.file_paths = file_paths
        self.model_name = model_name
        self.batch_size = batch_size
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classification_type = classification_type
        self.combined_df, self.label_mapping = self.load_and_combine_csvs(file_paths)
        self.max_length = self.find_percentile_length()


    def load_and_combine_csvs(self, file_paths):
        dataframes = []
        labels = []
        for file in os.listdir(file_paths):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(file_paths, file))
                label = file.replace('_generated_sentences.csv', '')
                df['label'] = label
                dataframes.append(df)
                labels.append(label)

        label_mapping = {label: idx for idx, label in enumerate(labels)}
        if self.classification_type == "multi-class":
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['label'] = combined_df['label'].map(label_mapping)
            return combined_df, label_mapping
        else:
            label_dfs = self.create_binary_classification_data(dataframes, labels)
            return label_dfs, label_mapping


    def create_binary_classification_data(self, dataframes, labels):
        label_dfs = {}
        
        for i, df in enumerate(dataframes):
            label = labels[i]
            df['label'] = 1  # Positive examples
            positive_df = df
            other_dfs = [dataframes[j] for j in range(len(dataframes)) if j != i]
            negative_sample = pd.concat(other_dfs, ignore_index=True)
            negative_sample = negative_sample[negative_sample['Type'] == 'original']
            
            if len(negative_sample) < len(positive_df):
                sampled_negative = negative_sample.sample(len(negative_sample), random_state=self.seed)
            else:
                sampled_negative = negative_sample.sample(len(positive_df), random_state=self.seed)
            
            sampled_negative['label'] = 0
            combined_df = pd.concat([positive_df, sampled_negative], ignore_index=True)
            combined_df['label'] = combined_df['label'].astype(int)
            label_dfs[label] = combined_df
            
        return label_dfs

    
    def find_percentile_length(self):
        lengths = []
        if isinstance(self.combined_df, dict):
            # Iterate through each DataFrame in the dictionary for binary classification
            for df in self.combined_df.values():
                for sentence in df['Sentence']:
                    tokenized = self.tokenizer(sentence, truncation=True, return_tensors="pt")
                    lengths.append(tokenized.input_ids.shape[1])
        else:
            # For multi-class classification, use the combined DataFrame
            for sentence in self.combined_df['Sentence']:
                tokenized = self.tokenizer(sentence, truncation=True, return_tensors="pt")
                lengths.append(tokenized.input_ids.shape[1])
        return int(np.percentile(lengths, 95))


    class CustomDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length, classification_type):
            self.data = dataframe
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.classification_type = classification_type
   
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data.iloc[idx]
            encoding = self.tokenizer(item['Sentence'], padding='max_length', truncation=True,
                                      max_length=self.max_length, return_tensors="pt")
            encoding = {key: val.squeeze(0) for key, val in encoding.items()}
            if self.classification_type == "multi-label":
                encoding['labels'] = torch.tensor(item['label'], dtype=torch.float)
            else:
                encoding['labels'] = torch.tensor(item['label'], dtype=torch.long)
            return encoding

    def collate_fn(self, batch):
        input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
        labels = torch.stack([item['labels'] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    def create_dataloaders(self, df, label, save_df=True):
        # Filter original sentences for validation
        original_df = df[df['Type'] == 'original']
        generated_df = df[df['Type'] != 'original']

        # Split original data into train and validation sets
        train_size = int(0.8 * len(original_df))
        val_size = len(original_df) - train_size

        train_original, val_original = torch.utils.data.random_split(
            original_df,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Combine training original data with generated data for training set
        train_combined = pd.concat([train_original.dataset.iloc[train_original.indices], generated_df], ignore_index=True)
        val_combined = val_original.dataset.iloc[val_original.indices]

        if save_df:
            path = f'/home/ezradin/pred-sentencing/results/sentence_generation/combained/dfs/{label}'
            os.mkdir(path)
            save_path = os.path.join(path, label+'_train.csv')
            train_combined.to_csv(save_path)
            save_path = os.path.join(path, label+'_val.csv')
            val_combined.to_csv(save_path)
            print(f'save {label} on {save_path}')
            
        train_dataset = self.CustomDataset(train_combined, self.tokenizer, self.max_length, self.classification_type)
        val_dataset = self.CustomDataset(val_combined, self.tokenizer, self.max_length, self.classification_type)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return train_loader, val_loader
