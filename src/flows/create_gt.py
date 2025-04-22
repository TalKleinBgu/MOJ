import os
import sys
import logging
import glob
import pandas as pd
from setfit import SetFitModel, SetFitTrainer
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime
import numpy as np
import shutil

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-cased")

from utils.files import config_parser, setup_logger, write_yaml
from utils.sentence_classification import evaluate, save_model, create_setfit_logger, load_datasets
from scripts.sentence_classification.setfit_handler import SetfitTrainerWrapper
from scripts.sentence_classification.setfit_din_version import SentenceTaggingModel
from flows.inference.predict_sentence_cls import predict_2cls_lvl_flow

def combine_csv_files(params, logger=None):
    main_folder_path  = params['db_path']
        # List to hold DataFrames
    dataframes = []

    # Loop through each subfolder in the main folder
    for subfolder in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Find CSV files in the current subfolder
            csv_files = glob.glob(subfolder_path + "/*.csv")
            
            # Read each CSV file in the subfolder
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                dataframes.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    output_file_path = os.path.join(main_folder_path, "combined_file.csv")
    combined_df.to_csv(output_file_path, index=False)
    logger.info(f"Combined file saved to {output_file_path}")

def add_csv_files_to_tagging(params, logger=None):
    file_path = os.path.join(params['db_path'], "preprocessing.csv")
    if not os.path.exists(file_path):
        df_cleaned = params['combine_files_path']
        df_cleaned = pd.read_csv(df_cleaned)
        random_verdicts = np.random.choice(df_cleaned['verdict'].unique(), size=100, replace=False)
        new_df = df_cleaned[df_cleaned['verdict'].isin(random_verdicts)]

        sentences_tagging_df = '/home/tak/pred-sentencing/resources/data/tagging/gt/sentence_tagging_1.csv'
        df = pd.read_csv(sentences_tagging_df)
        unique_verdicts = df['verdict'].unique()
        unique_verdicts_list = unique_verdicts.tolist()

        for verdict in new_df['verdict'].unique():
            verdict_tagged = verdict+'_tagged'
            if verdict_tagged in unique_verdicts_list:
                new_df=new_df[new_df['verdict'] != verdict]

        output_file_path = os.path.join(params['db_path'], "preprocessing.csv")
        new_df.to_csv(output_file_path, index=False)

    predict_2cls_lvl_flow(eval_path='results/evaluations/sentence_calssification',
                        first_lvl_cls_path=params['first_level_classifiers_path'],
                        classifiers_path=params['classifiers_path'],
                        experimant_name=params['experimant_name'],
                        test_set_path=params['test_set_path'],
                        case_dir_path=params['db_path'],
                        logger=logger,
                        )
    
    new_df = pd.read_csv(file_path)

    pred_path = os.path.join(params['db_path'], "sentences_predictions.csv")
    pred_df = pd.read_csv(pred_path)

    verdict_dict = new_df.set_index('text')['verdict'].to_dict()
    pred_df['verdict'] = pred_df['sentence'].map(verdict_dict)
    if 'sentence' in pred_df.columns:
        pred_df.rename(columns={'sentence': 'text'}, inplace = True)  
    pred_df.to_csv(pred_path, index=False)


def copy_gt(params, logger=None):
    source_path  = '/home/tak/pred-sentencing/results/db/sentences_predictions.csv'
    destination_path  = '/home/tak/pred-sentencing/resources/data/tagging/gt/sentence_tagging.csv'

    # Copy the file
    shutil.copy(source_path, destination_path)

    # Check if the file was copied
    if os.path.exists(destination_path):
        logger.info(f"File copied successfully to {destination_path}")
    else:
        logger.info("Failed to copy the file.")    



def main(params):
    logger = setup_logger(save_path=os.path.join(params['result_path'], 'logs'),
                          file_name='unbelance_setnence_classification')
    
    # combine_csv_files(params=params,logger=logger)
    add_csv_files_to_tagging(params=params, logger=logger)
    # copy_gt(params=params, logger=logger)

    

if __name__ == "__main__":
    params = config_parser("main_config")
    main(params)
    