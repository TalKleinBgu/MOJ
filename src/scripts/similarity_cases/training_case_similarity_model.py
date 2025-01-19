import logging
import sys
from datetime import date
import os
import numpy as np
import pandas as pd
import shap
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
import pickle
import importlib

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, pred_sentencing_path)

# from scripts.features.feature_extraction.prompts.claude import ClaudeSonnet
from pairs_similarity_creation import SimilarityPairsCreation
from utils.training_similarity import plot_roc_curve, save_model, save_report
from utils.files import config_parser, load_pkl, setup_logger
from utils.features import train_format_conversion
from utils.training_similarity import read_and_fill_prompt, similarity_parser

def dynamic_import(import_type):
    # Construct the base module path using the import_type
    module_base = f"scripts.features.feature_extraction.prompts.{import_type}"

    # Dynamically import the three classes from their respective modules
    ClaudeSonnet = importlib.import_module(f"{module_base}.claude").ClaudeSonnet
    
    # Return the imported classes as a tuple
    return ClaudeSonnet


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x).squeeze()

def contrastive_loss(output, label, margin=1.0):
    euclidean_distance = output.squeeze()
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

class SiameseDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        x = self.vectors[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def create_pairs(X, y):
    pairs = []
    labels = []
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        indices = np.where(y == label)[0]
        pairs.extend([(i, j) for i in indices for j in indices if i != j])
        labels.extend([1] * len([(i, j) for i in indices for j in indices if i != j]))
    
    return np.array(pairs), np.array(labels)

def train_siamese_network(X_train, y_train, X_test, y_test):
    # change to input 1 input 2 and not similarity vector
    # pairs_train, labels_train = create_pairs(X_train, y_train)
    # pairs_test, labels_test = create_pairs(X_test, y_test)
    
    train_dataset = SiameseDataset(X_train, y_train)
    test_dataset = SiameseDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    input_dim = len(X_train[0])
    model = SiameseNetwork(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for x, label in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, label)
            print(loss.item())
            loss.backward()
            optimizer.step()
    
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for x, label in test_loader:
            
            output = model(x)
            all_outputs.append(output)
            all_labels.append(label)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    predictions = (all_outputs > 0.5).int().tolist()
    return predictions, all_labels

class CaseSimilarity_Model:
    def __init__(self, case_pairs: str = None, models_type: list = None,
                 result_path:str = None, extraction_type: str = None,
                 output_dir_name:str = None,
                 create_more_data:bool = False,
                 error_analisys: bool=False,
                 seed:int = 42, logger:logging.Logger = None
                 ):

        """

        :param df_path:
        :param model_type:
        :param extraction_type:
        :param weights:
        :return:
        """
        self.case_pairs_df = case_pairs
        self.create_more_data = create_more_data
        self.extraction_type = extraction_type
        self.models_type = models_type
        self.error_analisys = error_analisys
        self.seed = seed
        if output_dir_name:
            result_folder_name = date.today().strftime("%Y-%m-%d") + '_' + output_dir_name
        self.result_path = os.path.join(result_path,
                                        'evaluations', 'case_similarity',
                                        result_folder_name
                                        )
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.logger = logger
        
    def get_copies(self, row, data):
        source = row['source']
        target = row['target']
        copies = data[(data['source'].str.startswith(source)) & (data['target'].str.startswith(target))]
        return copies  
    
    def claude_response(self, handler, api_key, prompt_path, predict_path, domain):
    
        predicts = []
        ClaudeSonnet = dynamic_import(domain)

        claude = ClaudeSonnet(api_key)
        if not predict_path:
            for _, row in tqdm(self.case_pairs_df.iterrows(), total=self.case_pairs_df.shape[0]):
                source_vec = handler.verdict_fe_dict[row.source]
                target_vec = handler.verdict_fe_dict[row.target]
                prompt = read_and_fill_prompt(prompt_path, target_vec, source_vec)
                claude_answer = claude.generate_answer(prompt)
                predict_similarity = similarity_parser(claude_answer)
                predicts.append(predict_similarity)
                
            with open('predicts_similarity_cases_on_clude extracted.pkl', 'wb') as f:
                pickle.dump(predicts, f)
        else:
            predicts = load_pkl(predict_path)
            
            
        true_label = self.case_pairs_df['label'].values.tolist()
        true_label_binary = [1 if label >= 3 else 0 for label in true_label]
        self.__evaluations(true_label_binary, predicts)
        print()
        
    def train(self, binary_label: bool=True, loocv: bool=False) -> None:
        """
        :param binary_label: Whether to treat labels as binary (default: True)
        :param loocv: Whether to use Leave-One-Out Cross-Validation (default: False)
        :return: None
        """
        all_y_test = []
        all_y_pred = []
        all_y_probs = []
        all_target = []
        all_source = []
        case_pairs_df_ = train_format_conversion(self.case_pairs_df, binary_label)
        case_pairs_df_ = case_pairs_df_.dropna(subset=['similarity_vector', 'label'])
        similarity_vector = list(case_pairs_df_['similarity_vector'].values)
        labels = list(case_pairs_df_['label'].values)

        if loocv:

            
            loo = LeaveOneOut()
            for model_type in self.models_type:
                self.logger.info(f"Start to train {model_type} case similarity model using LOOCV!")
                model = self._create_model(model_type)
                
                for train_index, test_index in loo.split(similarity_vector):
                    X_train, y_train = [], []
                    X_test, y_test = [], []
                    
                    test_row = case_pairs_df_.iloc[test_index[0]]
                    if 'copy' in test_row['source']:
                        continue
                    copies = self.get_copies(test_row, case_pairs_df_)
                    unique_test = copies[~copies['source'].str.contains('copy')]
                    X_test = unique_test['similarity_vector'].values.tolist()
                    y_test = unique_test['label'].values.tolist()                    


                    # X_train, X_test = [similarity_vector[i] for i in train_index], [similarity_vector[i] for i in test_index]
                    # y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]
                    
                    # We will download all the copies of the selected test case from the 
                    # training set to prevent a situation where the model sees a very
                    # similar mole in the training set
                    train_df = case_pairs_df_.drop(copies.index)    
                    for i in train_index:
                        if i not in copies.index:
                            X_train.append(similarity_vector[i])
                            y_train.append(labels[i]) 
          
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    y_probs = model.predict_proba(X_test)
                    source_case = case_pairs_df_.iloc[test_index].source.values[0]
                    target_case = case_pairs_df_.iloc[test_index].target.values[0]
                    
                    all_y_test.append(y_test)
                    all_y_pred.append(y_pred)
                    all_y_probs.append(y_probs)
                    all_source.append(source_case)
                    all_target.append(target_case)
                    


                self.__evaluations(all_y_test, all_y_pred, all_y_probs, model_type, all_source, all_target)  

        else:

            X_train, X_test, y_train, y_test = train_test_split(
                similarity_vector, labels, test_size=0.25, random_state=self.seed, stratify=labels
            )
            
            for model_type in self.models_type:
                self.logger.info(f"Start to train {model_type} case similarity model!")
                if model_type == 'siamese_network':
                    y_pred, y_probs = train_siamese_network(X_train, y_train, X_test, y_test)  
                else:
                    model = self._create_model(model_type)
                    model.fit(X_train, y_train)
                    save_model(model, model_type, self.result_path, self.logger)            
                
                    y_pred = model.predict(X_test)
                    y_probs = model.predict_proba(X_test)
                self.__evaluations(y_test, y_pred, y_probs, model_type)  

    def error_model_check():
        pass
    
    def _create_model(self, model_type):
        if model_type == 'logistic_regression':
            return LogisticRegression(random_state=self.seed)
        elif model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.seed)
        elif model_type == 'decision_tree':
            return DecisionTreeClassifier(random_state=self.seed)
        elif model_type == 'svm':
            return SVC(probability=True, random_state=self.seed)
        # elif model_type == 'siamese_network':
        #     return SiameseNetwork()
        else:
            raise ValueError(f"Invalid model type: {model_type}")
                


    def __evaluations(self, y_test, y_pred, y_probs=None, model_type=None, all_source=None, all_target=None, ):

        """

        :param y_test:
        :return:
        """
        auc_pr = ''
        
        def find_optimal_threshold(y_true, y_probs):
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_probs)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            optimal_threshold_index = np.argmax(f1_scores)
            optimal_threshold = thresholds_pr[optimal_threshold_index]
            return optimal_threshold, f1_scores[optimal_threshold_index]
        
        # if y_probs is not None:
        #     y_probs = np.concatenate(y_probs)
        
        #  if the y_test is list of lists so concatente it 
        if not isinstance(y_test[0], int) and not isinstance(y_test[0],np.int64):
            y_test = np.concatenate(y_test)
            y_pred = np.concatenate(y_pred)
        if y_probs is not None:
            optimal_threshold, f1 = find_optimal_threshold(y_test, y_probs[:, 1])
            # optimal_threshold = 0.63
            y_pred = (y_probs[:, 1]>= optimal_threshold).astype(int)

            if torch.is_tensor(y_pred):
                y_pred = y_pred.cpu().numpy()

        if torch.is_tensor(y_probs):
            y_probs = y_probs.cpu().numpy()
            y_probs = np.concatenate([probs_array.squeeze() for probs_array in y_probs])
        # else:
        #     y_probs_1 = [prob[0][1] for prob in y_probs]
        
        

        accuracy = round(accuracy_score(y_test, y_pred), 3)
        self.logger.info(f"Accuracy: {accuracy}")
        
        precision = round(precision_score(y_test, y_pred, average='weighted'), 3)
        self.logger.info(f"Precision: {precision}")
        
        recall = round(recall_score(y_test, y_pred, average='weighted'), 3)
        self.logger.info(f"Recall: {recall}")
        
        f1 = round(f1_score(y_test, y_pred, average='weighted'), 3)
        self.logger.info(f"F1: {f1}")
        
        
        # auc_ = round(roc_auc_score(y_test,y_probs), 3)
        # plot_roc_curve(y_test, y_probs, self.result_path, self.extraction_type)
        # self.logger.info(f"AUC: {auc_}")
        if y_probs is not None:
            try:
                precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_probs[:, 1])
                # precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_probs_1)
                
            except:
                # y_probs_array = np.array([probs_array[:, 1] for probs_array in y_probs])
                precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_probs)

            # Calculate PR AUC
            auc_pr = round(auc(recall_pr, precision_pr), 3)
            self.logger.info(f"PR AUC: {auc_pr}")
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        self.logger.info(f"Confusion Matrix:\n{conf_matrix}")
        evaluation_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'PR AUC': auc_pr
            }

        save_report(self.result_path, model_type, self.extraction_type, evaluation_metrics, conf_matrix, auc)
        # if all_source is not None:
        #     error_list = []
        #     for i, y_true in enumerate(y_test):
        #         try:
        #             if y_pred[i][0] != y_true[0]:
        #                 error_list.append({'source':all_source[i],
        #                                     'target':all_target[i],
        #                                     'pred': y_pred[i][0],
        #                                     'prob':y_probs[i]})
        #         except:
        #             if y_pred[i] != y_true[0]:
        #                 error_list.append({'source':all_source[i],
        #                                     'target':all_target[i],
        #                                     'pred': y_pred[i][0],
        #                                     'prob':y_probs[i]})
        #     df_errors = pd.DataFrame(error_list)
        #     print()
        

    def __sample_explainer(self, X_test: list = None, sample_id: int = 0, model=None):
        """

        :return:
        """
        explainer = shap.TreeExplainer(model)
        X_test = X_test.values
        chosen_instance = X_test[[sample_id]]
        shap_values = explainer.shap_values(chosen_instance)
        
        shap.initjs()
        expected_value = explainer.expected_value[1]
        x = shap_values[sample_id][:, 1]
        shap.force_plot(expected_value, x, chosen_instance, matplotlib=matplotlib)


def main(params):
    sp_params = params['pairs_similarity']
    logger = setup_logger(save_path=os.path.join(params['result_path'], 'logs'),
                          file_name='pairs_similarity')
    
    handler = SimilarityPairsCreation(db_path=params["db_path"],
                                      tagged_pairs_path=sp_params["tagged_pairs_path"], 
                                      embeddding_features_path= sp_params["embeddding_features_path"],
                                      features_type=sp_params['features_type'],
                                      logger=logger,
                                      label_type=sp_params['label_type'])
    
    case_pairs = handler.create_df_to_train(output_dir_name='din_testss',
                                            type_task='train_case_sim')   
    
    doc_sim = CaseSimilarity_Model(case_pairs=case_pairs, 
                                   extraction_type=sp_params['features_type'],
                                   result_path=params['result_path'],
                                   models_type=params['models_type'],
                                   seed = params['seed'],
                                   output_dir_name = params['output_dir_name'],
                                   logger=logger)

    doc_sim.train(params['binary_label'])


if __name__ == "__main__":
   params = config_parser("main_config")
   main(params)