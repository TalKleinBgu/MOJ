import os
import numpy as np
import pandas as pd
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import config_parser, yaml_load
feature_association_path = 'resources/appendices/feature_association.yaml'

class FeatureEmbedding():
    """
    This class generates an embedding representation given the - single - case folder and the extraction type
    """
    def __init__(self, model, tokenizer, fe_df_path:str = None) -> None:
        """
        
        Args: 
            fe_csv_path - path to csv thatn cotain feature extraction
            
        """
        if fe_df_path is not None:
            self.feature_df = pd.read_csv(fe_df_path)
        self.model = model
        self.tokenizer = tokenizer
        self.feature_association = yaml_load(feature_association_path)
    
        
    def embed_element(self, element, max_token_limit=512):
        if isinstance(element, bool):
            return 1 if element else 0

        if isinstance(element, str):
            if element == "":
                return -1
            tokens = self.tokenizer(element, return_tensors='pt', truncation=True)
            num_tokens = tokens['input_ids'].size(1)
            if num_tokens > max_token_limit:
                print(f"Warning: Input exceeds the maximum token limit of {max_token_limit}.")
            with torch.no_grad():
                embeddings = self.model(**tokens).hidden_states[-1]
            avg_embeddings = torch.mean(embeddings, dim=1).numpy()
            return avg_embeddings.tolist()

        if isinstance(element, list):
            embedded_list = [self.embed_element(sub_element) for sub_element in element]
            return embedded_list


    def get_embedding_dict(self):
        """
        Calculate the average vector (first each feature is considered separately 
        and its embedding representation), for each of the features
        Returns:
            dictionery of embedding vector by features (as key)
        """

        embd_vectors_dict, text_vectors_dict = {}, {}
        feature_df = self.feature_df
        
        for feature in self.feature_association['textual']:
            if feature == "full_verdict":
                continue
            else:
                feature_text_vector_series = feature_df[feature_df['feature_key'] == feature]['extraction']
                feature_text_vector_lst = list(feature_text_vector_series.values)
                if len(feature_text_vector_lst) == 0:
                    feature_embd_vector = -1
                else: 
                    feature_embd_vector = self.embed_element(feature_text_vector_lst)
                    
                    # Stack the arrays along a new axis to create a 3D array
                    stacked_feature_embd = np.stack(feature_embd_vector, axis=0)
                    #Calculate the mean along the first axis to get the mean by location
                    feature_embd_vector = np.mean(stacked_feature_embd, axis=0).tolist()

                embd_vectors_dict[feature] = feature_embd_vector
                text_vectors_dict[feature] = feature_text_vector_lst


        data = {
            'embd_vector': embd_vectors_dict,
            'text_vector': text_vectors_dict
        }
        return data


def main(params):
    feature_sim_params = params['feature_similarity']
    model = AutoModelForMaskedLM.from_pretrained(feature_sim_params['embedding_model_name'], output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(feature_sim_params['embedding_model_name'])
    femb = FeatureEmbedding(fe_df_path=feature_sim_params['fe_df_path'],
                            embedding_model=model,
                            tokenizer=tokenizer,
                            feature_association_path=feature_sim_params['feature_association_path'])
    data = femb.get_embedding_dict()


if __name__ == '__main__':
    params = config_parser('main_config')
    main(params)
