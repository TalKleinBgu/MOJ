import logging
import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import config_parser, load_json, load_pkl, setup_logger


class PredictSimilarityFlow():
    
    def __init__(self, model_path:str = None, db_path:str = None,
                 pairs_similarity_path:str = None, logger:logging.Logger = None):
        
        self.db_path = db_path
        self.logger = logger
        
        try:
            self.model = load_pkl(model_path)
            self.pairs_similarity_df = pd.read_csv(pairs_similarity_path)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error occurred during initialization: {e}")
            else:
                print(f"Error occurred during initialization: {e}")
        
        self.similarity_prediction_df = self.predict_similarity_and_probabilities()
            
        
        
    def predict(self, model, similarity_vector):
        """
        Predicts the label of a similarity vector using a trained model.
        Makes sure that the data is in the right format for the model to run
        Args:
            model to test
            similarity_vector to test

        Returns:
            label if te cases are similar or not
        """
        if isinstance(similarity_vector, list):
            similarity_vector = np.array(similarity_vector)
        if similarity_vector.ndim == 1:
            similarity_vector = [similarity_vector]
        prediction = model.predict(similarity_vector)
        if hasattr(model, "predict_proba"):
            predicted_probabilities = model.predict_proba(similarity_vector)
            return predicted_probabilities
        return prediction

    def __convert_column_to_vector(self, row):
        similarity_lst = []
        for column_name in row.keys():
            if column_name not in ['source', 'target', 'Unnamed: 0']:
                similarity_lst.append(row[column_name])
        return similarity_lst
    
    def predict_similarity_and_probabilities(self):
        """
        Predict similarity scores and probabilities for pairs of cases.

        Returns:
            pandas.DataFrame: DataFrame containing pairs similarity scores and probabilities.
        """
        similarity_scores = []
        for _, row in self.pairs_similarity_df.iterrows():
            similarity_vector = self.__convert_column_to_vector(row)
            similarity_probabilities = self.predict(self.model, similarity_vector)

            similarity_scores.append(similarity_probabilities[0][1])

        self.pairs_similarity_df['probability_score'] = similarity_scores
        similarity_df_sorted = self.pairs_similarity_df.sort_values(by='probability_score', ascending=False)
        return similarity_df_sorted
    
    def get_similarity_prediction(self):
        """
        Retrieve prediction results with labels for the cases.

        Returns:
            pandas.DataFrame: DataFrame containing prediction results with punishment labels.
        """
        labels = []
        for _, row in self.similarity_prediction_df.iterrows():
            label_path = os.path.join(self.db_path, row['target'], 'label.json')
            label = load_json(label_path)
            labels.append(label)
        self.similarity_prediction_df['punishment_range'] = labels

        return self.similarity_prediction_df
    
    
def main(params):
    logger = setup_logger(save_path=os.path.join(param['result_path'], 'logs'),
                          file_name='predict_similarity_test')
    
    psf = PredictSimilarityFlow(model_path=params['model_path'],
                                db_path=params['db_path'],
                                pairs_similarity_path=params['pairs_similarity_path'],
                                logger=logger)
    
    result_df = psf.get_similarity_prediction()
    print()
    
if __name__ == '__main__':
    param = config_parser('main_config')
    main(param)