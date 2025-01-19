from setfit import SetFitModel
import os
import pandas as pd
import torch
import sys
import numpy as np

sys.path.append('../../../')

from scripts.sentence_classification.predict_sentence_cls.loads import models_name_extraction
device = "cuda" if torch.cuda.is_available() else "cpu"


class SetFitPostProcessor:
    """
    A class for postprocessing predictions with models trained using SetFit.

    Attributes:
        second_level_labels (list): A list of second-level labels from the configuration.
        first_level_labels (list): A list of first-level labels from the configuration.
        init_model (str): The initial path where the models are stored.
        classifiers (dict): A dictionary holding the loaded SetFitModel for each label.
        save_path (str): Path to save the final predictions.
        thresholds (dict): A dictionary of thresholds for decision-making for each label.
        logger: Logger instance to log information during processing.
        df (DataFrame): Pandas DataFrame containing the dataset for prediction.
        data (dict): Dictionary to store prediction results.
        labels_map (dict): Mapping of second-level labels to corresponding first-level labels.

    Methods:
        __init__(self, logger, df_path=None, model_path=None, save_path=None): Initializes the postprocess instance.
        find_key(self, val): Helper method to find the corresponding first-level label for a given second-level label.
        predict_reject(self): Runs prediction to determine if an instance should be rejected.
        predict_first_level(self): Runs first-level predictions for all non-rejected instances.
        predict_second_level(self): Runs second-level predictions based on first-level predictions.
        predict(self): Orchestrates the running of all predictions and returns the final DataFrame with results.
    """

    def __init__(self, logger, classifiers, second_level_labels:list = None, first_level_labels:list = None, threshold:object = None,
                 preprocess_file_path: str = None, save_path: str = None, evaluation: bool = False):
        
        self.classifiers = classifiers
        self.second_level_labels = second_level_labels
        self.first_level_labels = first_level_labels
        self.save_path = save_path
        self.threshold = threshold

        self.logger = logger
        self.evaluation = evaluation            
        # if evaluation:
        #     self.df = pd.read_excel(preprocess_file_path)
        # else:
        self.df = pd.read_csv(preprocess_file_path)
        self.data = pd.DataFrame(columns=['case', 'text', 'reject'])
        self.tagged_df = pd.DataFrame(columns=['case', 'text', 'reject'])
        if 'verdict' in self.df.columns:
            self.df.rename(columns={'verdict': 'case'}, inplace=True)
        else:
            self.data['case'] = self.df['Case']
        self.tagged_df['case'] = self.df['case']
        self.data['text'] = self.df['text']
        self.tagged_df['text'] = self.df['text']
        self.labels_map = {
            'CONFESSION': ['REGRET', 'RESPO','CONFESSION_LVL2'],
            'CIRCUM_OFFENSE': ['CIR_OBTAIN_WAY_WEP', 'CIR_AMMU_AMOUNT_WEP', 'CIR_TYPE_WEP', 'CIR_HELD_WAY_WEP',
                                'PURPOSE', 'CIR_STATUS_WEP', 'PLANNING', 'CIR_BUYER_ID_WEP', 'CIR_MONEY_PAID_WEP',
                                'CIR_PLANNING'
                                'CIR_PURPOSE', 'CIR_USE', 'CIR_PLANNING', 'PLANING', 'CIR_PLANING', 'CIR_PURPOSE']
        }


    def find_key(self, val):
        """
        Find the first-level key (label) for a given second-level value (label).

        :param val: The second-level label for which to find the corresponding first-level label.
        :return: The first-level label if found, otherwise None.
        """

        for key, values_list in self.labels_map.items():
            for label in values_list:
                if val in label:
                    return key
        return None


    def predict_reject(self):
        """
        Predict the 'reject' label for all instances in the dataset.

        :return: None. Updates the 'reject' field in the class's DataFrame attribute.
        """
        self.logger.info("predict label reject start")

        # If 'reject' column exists and is numeric, rename it
        if 'reject' in self.df.columns:
            self.df.rename(columns={'reject': 'true-reject'}, inplace=True)

        self.df['reject'] = False
        self.data["reject-predicted"] = np.nan
        self.tagged_df["reject"] = np.nan


        for idx, row in self.df.iterrows():
            try:
                proba_reject = self.classifiers['REJECT'].predict_proba([row['text']]).numpy()[0][1]
            except:
                continue
            if type(self.threshold) is not float:
                threshold = next(item['reject'] for item in self.threshold if 'reject' in item)
            else:
                threshold = 0.5
            reject = proba_reject > threshold
            self.data.loc[idx, 'reject-predicted'] = proba_reject
            self.tagged_df.loc[idx, 'reject'] = int(reject)

            self.df.at[idx, 'reject'] = reject


    def predict_first_level(self):
        """
        Predict first-level labels for all non-rejected instances in the dataset.

        :return: None. Updates the DataFrame with predictions for first-level labels.
        """

        if self.first_level_labels is None:
            self.logger.errors('First level labels is None!')
            
        for first_level_label in self.first_level_labels:
            if first_level_label.lower() == "reject":
                continue
            self.data[f"{first_level_label}-predicted"] =np.nan
            self.tagged_df[first_level_label] = np.nan


        for idx, row in self.df[self.df['reject'] == False].iterrows():
            for first_level_label in self.first_level_labels:
                if first_level_label.lower() == "reject":
                    continue
                try:
                    proba = self.classifiers[first_level_label].predict_proba([row['text']]).numpy()[0][1]
                except:
                    continue
                self.data.loc[idx,f"{first_level_label}-predicted"] = proba
                self.df.at[idx, f"{first_level_label}-predicted"] = proba
                if type(self.threshold) is not float:
                    label_threshold = next(item[first_level_label] for item in self.thresholds if first_level_label in item)
                else: 
                    label_threshold = 0.5
                label = proba > label_threshold
                self.tagged_df.loc[idx, first_level_label] = int(label)



    def predict_second_level(self):
        """
        Predict second-level labels for all instances in the dataset, based on the predictions of the first-level labels.

        :return: None. Updates the DataFrame with predictions for second-level labels.
        """

        for second_level_label in self.second_level_labels:
            self.data[f"{second_level_label}-predicted"] =np.nan
            self.tagged_df[second_level_label] =np.nan


        for idx, row in self.df[self.df['reject'] == 0].iterrows():
            for second_level_label in self.second_level_labels:
                first_label = self.find_key(second_level_label)
                if type(self.threshold) is not float:
                    label_threshold = next(item[first_label] for item in self.thresholds if first_label in item)
                else:
                    label_threshold = 0.5
                try:
                    if row[f"{first_label}-predicted"] > label_threshold:
                        
                            proba = self.classifiers[second_level_label][0].predict_proba([row['text']]).numpy()[0][1]
                            self.data.loc[idx,f"{second_level_label}-predicted"] = proba

                            # # Append true value if the column exists
                            # if f"{second_level_label}" in self.df.columns:
                            #     true_value = row[second_level_label]
                            #     self.data.loc[idx,f"true-{second_level_label}"] = true_value

                            if type(self.threshold) is not float:
                                label_threshold = next(item[second_level_label] for item in self.thresholds if second_level_label in item)
                            else:
                                label_threshold = self.classifiers[second_level_label][1]
                            label = proba > label_threshold
                            self.tagged_df.loc[idx, second_level_label] = int(label)
                            
                except:
                        self.tagged_df.loc[idx, second_level_label] = 0

    def predict(self):
        """
        Run all prediction methods and compile results into a final DataFrame.
        :return: A pandas DataFrame with all prediction results.
        """
        if not self.evaluation:
            dir_name = os.path.basename(os.path.normpath(self.save_path))
            
        self.predict_reject()
        self.predict_first_level()
        self.predict_second_level()
        
        if not self.evaluation:
            self.logger.info(f"End to predict {dir_name} verdict")
        else:
            self.logger.info(f"End to predict tagged file!")
            
        self.tagged_df.fillna(0, inplace=True)
        return self.data, self.tagged_df
    


