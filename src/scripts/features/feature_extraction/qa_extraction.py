import sys
import pandas as pd
import os
from collections import OrderedDict
import re
from transformers import pipeline
from datetime import datetime
import importlib
import numpy as np
# Get the directory where this file is located
current_dir = os.path.abspath(__file__)
# Calculate the path to the root of the project
pred_sentencing_path = os.path.abspath(os.path.join(current_dir,  '..', '..', '..', '..'))
# Add the project root to sys.path so that local modules can be imported
sys.path.insert(0, pred_sentencing_path)

# from scripts.features.feature_extraction.prompts.weapon.dicta import DicatePrompts
# from scripts.features.feature_extraction.prompts.weapon.dicta2 import Dicta2Prompts
# from scripts.features.feature_extraction.prompts.weapon.claude import ClaudeSonnet
from utils.files import get_date, reformat_sentence_tagged_file

def dynamic_import(import_type):
    # Construct the base module path using the import_type
    module_base = f"scripts.features.feature_extraction.prompts.{import_type}"

    # Dynamically import the three classes from their respective modules
    DicatePrompts = importlib.import_module(f"{module_base}.dicta").DicatePrompts
    Dicta2Prompts = importlib.import_module(f"{module_base}.dicta2").Dicta2Prompts
    ClaudeSonnet = importlib.import_module(f"{module_base}.claude").ClaudeSonnet
    
    # Return the imported classes as a tuple
    return DicatePrompts, Dicta2Prompts, ClaudeSonnet

class QA_Extractor:
    """
    A class responsible for extracting question-answer-based features from text data. 
    It utilizes various prompt classes (DicatePrompts, Dicta2Prompts, ClaudeSonnet) 
    based on the model and tokenizer provided, or uses an API key (e.g., for Claude).

    Attributes:
        model (transformers pipeline/model): The Transformer model used for inference.
        tokenizer (transformers tokenizer): The tokenizer corresponding to the model.
        experiment_name (str): An optional experiment name used for logging or saving.
        logger (logging.Logger): A logger instance used to record process flow information.
        save_path (str): Path where the extracted feature CSV will be saved.
        api_key (str): An API key, e.g. for Claude or other external LLM services.
    """

    def __init__(self, model=None, tokenizer=None, experiment_name='', save_path=None, logger=None, api_key=''):
        """
        Initialize the QA_Extractor class.

        Args:
            model: A model instance or pipeline used for Q&A (optional).
            tokenizer: The tokenizer matching the model (optional).
            experiment_name: An optional string naming the experiment.
            save_path: Where resulting CSV files should be stored.
            logger: Logger instance for logging events and errors.
            api_key: API key if an external call is needed (e.g., Claude).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.experiment_name = experiment_name
        self.logger = logger
        self.save_path = save_path
        self.api_key = api_key
        self.feature_dict = {}

    def extract(self, data_path, model_name="", db_flow=False, error_analysis=False, label='CIR_STATUS_WEP', import_type='weapon'):
        """
        Runs the feature-extraction flow on a CSV file located at data_path, 
        using either a local model/tokenizer pipeline or an external API call class.

        Args:
            data_path (str): The path to the directory containing sentence_prediction.csv.
            model_name (str): The name of the model; used in the output filename.
            db_flow (bool): Whether we are processing a DB directory (unused here but can be expanded).
            error_analysis (bool): If True, might store additional information for debugging (unused in this snippet).
            label (str): An optional label name used for reference.
            import_type (str): Determines which prompt classes to import dynamically.

        Returns:
            pd.DataFrame: A DataFrame containing all extracted features appended in new columns.
        """
        # Dynamically import prompt classes based on the given import_type
        DicatePrompts, Dicta2Prompts, ClaudeSonnet = dynamic_import(import_type)

        # Construct full path to the CSV file with sentence predictions
        # data_path_csv = os.path.join(data_path, 'sentence_predictions.csv')
        data_path_csv = os.path.join(data_path, 'sentence_tagging.csv')

        # Instantiate the prompt class depending on self.model / self.tokenizer / self.api_key
        if self.model is None:
            # If no local model is provided, we default to a ClaudeSonnet instance
            dp = ClaudeSonnet(self.api_key)
        elif self.tokenizer is None:
            # If no tokenizer is provided, we assume an older DicatePrompts usage
            dp = DicatePrompts(self.model, [])
        else:
            # Otherwise, we assume a Dicta2Prompts usage (model + tokenizer)
            dp = Dicta2Prompts(model=self.model, tokenizer=self.tokenizer)

        data = []  # Will collect dictionaries of extracted features, row by row

        # Read the sentence_prediction CSV file
        df = pd.read_csv(data_path_csv)
        
        # Iterate over each row in the CSV to extract features
        for _, row in df.iterrows():
            # Skip rows where 'reject' == 1 (indicating a rejected row)
            if row['reject'] == 1:
                continue
            
            # Initialize a dictionary to collect features for this row
            dict_lables = {}
            
            # The raw text is assumed to be under 'text'
            text = row['text']
            dict_lables['text'] = text

            # Iterate over all columns in the row
            for column, value in row.items():
                # Skip certain columns or column names
                if column.lower() in ['verdict'] or 'Unnamed' in column:
                    continue
                elif column.lower() in ['text']:
                    continue
                elif value == 0:
                    dict_lables[column] = ""
                elif column.lower() == 'reject':
                    continue
                else:
                    # If column is 'PUNISHMENT', handle multiple sub-features
                    if column == 'PUNISHMENT':
                        options = ['ACTUAL', 'SUSPENDED', 'FINE']
                        for option in options:
                            func_name = f'ask2extract_{column}_{option}'
                            # Check if the dynamic prompts class has a method for this sub-feature
                            if hasattr(dp, func_name):
                                func = getattr(dp, func_name)
                                answers = func(text)
                                features_set = []
                                # Convert answers to a list if not already
                                if isinstance(answers, list):
                                    for feature in answers:
                                        if feature not in features_set:
                                            features_set.append(feature)
                                elif isinstance(answers, str):
                                    features_set.append(answers)
                                # Store extracted sub-feature in a column name like PUNISHMENT_ACTUAL
                                name_column = f'{column}_{option}'
                                dict_lables[name_column] = features_set
                    else:
                        # Otherwise, form a function name for the column
                        func_name = f'ask2extract_{column}'
                        # If the dynamic prompts class has this method, call it
                        if hasattr(dp, func_name):
                            func = getattr(dp, func_name)
                            answers = func(text)
                            features_set = []
                            # Convert answers to a list if not already
                            if isinstance(answers, list):
                                for feature in answers:
                                    if feature not in features_set:
                                        features_set.append(feature)
                            elif isinstance(answers, str):
                                features_set.append(answers)
                            dict_lables[column] = features_set
            
            # Append the dictionary with extracted features to our data list
            data.append(dict_lables)
        
        # Convert our feature dictionaries to a DataFrame
        results_df = pd.DataFrame(data)

        # The model name is trimmed for the output filename (e.g., "dicta-il/dictabert-heq" -> "dictabert-heq")
        model_name = model_name.split('/')[-1]
        save_path = os.path.join(data_path, f'features_extraction.csv')
        results_df.to_csv(save_path, index=False)

        # Build the save path with the model name and current date
        save_path = os.path.join(data_path, f'qa_features.csv')
        # Save the results as a CSV
        results_df = results_df.melt(
            id_vars='text',        # Columns to keep as identifiers (unchanged)
            var_name='feature_key',      # Name for the 'old' column headers
            value_name='extraction'     # Name for the values from those columns
        )
        # Replace empty strings, empty lists, and other placeholders with NaN
        results_df['extraction'] = results_df['extraction'].apply(
            lambda x: np.nan if x in ["", [], None] else x
        )       
        results_df = results_df.dropna(subset=['extraction'])
        results_df = results_df[['feature_key', 'text', 'extraction']]
        # results_df = self.extract_offense_info(results_df, data_path)
        results_df = self.extract_offense_info_drugs(results_df, data_path)

        results_df.to_csv(save_path, index=False)
        print(f'Save extract DF to {save_path}')
        
        return results_df

    def extract_offense_info_drugs(self, results_df, data_path):
        """
        (Unused in the main flow above, but shown as an example of further processing.)
        
        Extracts information about offenses from a verdict using regex patterns to match 
        specific legal terminology and categorizes them accordingly. Updates and returns 
        the provided results_df with new columns/features.

        Args:
            results_df (pd.DataFrame): DataFrame to which the offense information will be added.
            data_path (str): Path where additional input CSV (e.g., preprocessing.csv) might be stored.

        Returns:
            pd.DataFrame: Updated DataFrame with offense information added.
        """
        # Load an external CSV that presumably has a "text" column
        preprocessing_df = pd.read_csv(os.path.join(data_path, "preprocessing.csv"))
        pattern = r'(21\s*\(\s*[אב]\s*\))|(?:\+|ו-)(\(\s*[אב]\s*\))|(7\s*\(\s*[אבג]\s*\))|(?:\+|ו-)(\(\s*[אבג]\s*\))|(?:סעיף|סעיפים)\s*(6(?!\d)|13(?!\d)|14(?!\d)|22(?!\d))|(19א)'
        # Patterns for matching specific legal references in text
        verdict_number_pattern = re.compile(pattern)

        # Initialize dictionaries to hold offense features (dict not declared in __init__, example usage)
        self.feature_dict["OFFENCE_NUMBER"] = []
        self.feature_dict["OFFENCE_TYPE"] = []

        # For each row in preprocessing_df, check for offense patterns
        for _, row in preprocessing_df.iterrows():
            text = row.get('text', '')
            
            # Look for legal references and offense types in the text
            verdict_numbers = verdict_number_pattern.findall(text)
            verdict_numbers = [match for group in verdict_numbers for match in group if match]

            # If any verdict numbers are found, add them to feature_dict
            if verdict_numbers:
                self.feature_dict["OFFENCE_NUMBER"].extend([''.join(num) for num in verdict_numbers])
                offences_numbers_output = []
                for offence_number in list(set(verdict_numbers)):
                    if "א" in str(offence_number):
                        if "21" in str(offence_number):
                            offences_numbers_output.append("סעיף 21א")
                        elif "7" in str(offence_number):
                            offences_numbers_output.append("סעיף 7א")
                        elif "19" in str(offence_number):
                            offences_numbers_output.append("סעיף 19א")
                    elif "ב" in str(offence_number):
                        if "21" in str(offence_number):
                            offences_numbers_output.append("סעיף 21ב")
                        elif "7" in str(offence_number):
                            offences_numbers_output.append("סעיף 7ב")
                    elif "ג" in str(offence_number):
                        offences_numbers_output.append("סעיף 7ג")
                    else:    
                        offences_numbers_output.append(f'סעיף {offence_number}')
                # Append a row to results_df for each found offense number
                results_df = pd.concat([results_df,
                                        pd.DataFrame({
                                            'feature_key': "OFFENCE_NUMBER",
                                            'text': text,
                                            'extraction': [offences_numbers_output]
                                        })],
                                       ignore_index=True)

            # Extend the OFFENCE_NUMBER list with all found verdict numbers for this row
            self.feature_dict["OFFENCE_NUMBER"].extend([''.join(num) for num in verdict_numbers])

        # Remove duplicates and standardize the OFFENCE_NUMBER list
        self.feature_dict["OFFENCE_NUMBER"] = list(set(self.feature_dict["OFFENCE_NUMBER"]))
        # offences_numbers_output = []
        # for offence_number in list(set(self.feature_dict["OFFENCE_NUMBER"])):
        #     if "א" in offence_number:
        #         offences_numbers_output.append("144 א")
        #     elif "ב" in offence_number:
        #         offences_numbers_output.append("144 ב")
        # self.feature_dict["OFFENCE_NUMBER"] = offences_numbers_output

        
        return results_df
    

    def extract_offense_info(self, results_df, data_path):
        """
        (Unused in the main flow above, but shown as an example of further processing.)
        
        Extracts information about offenses from a verdict using regex patterns to match 
        specific legal terminology and categorizes them accordingly. Updates and returns 
        the provided results_df with new columns/features.

        Args:
            results_df (pd.DataFrame): DataFrame to which the offense information will be added.
            data_path (str): Path where additional input CSV (e.g., preprocessing.csv) might be stored.

        Returns:
            pd.DataFrame: Updated DataFrame with offense information added.
        """
        # Regex patterns to identify offenses
        offense_patterns = [
            "רכיש[הת]",
            "חזק[הת]",
            "נשיא[הת]",
            "החזק[הת]",
            "הובל[הת]",
            "עסק[הת]",
            "סחר[הת]",
            "ירי[יהות]",
            "ירי"
        ]
        # Mapping from partial string to a more descriptive offense name
        offense_mapping = {
            "רכיש": "סחר בנשק",
            "חזק": "החזקה נשק",
            "נשיא": "נשיאת נשק",
            "הובל": "הובלת נשק",
            "עסק": "סחר בנשק",
            "סחר": "סחר בנשק"
        }

        # Load an external CSV that presumably has a "text" column
        preprocessing_df = pd.read_csv(os.path.join(data_path, "preprocessing.csv"))
        
        # Patterns for matching specific legal references in text
        verdict_number_pattern = re.compile(r'(144\s*\(?\s*[אב]\d*\s*\)?)|(340א)')
        offense_combined_pattern = '|'.join(offense_patterns)
        offense_combined_with_qualifiers_pattern = f'({offense_combined_pattern})( ו{offense_combined_pattern})*'
        pattern_ = f'({offense_combined_with_qualifiers_pattern})( של נשק| של תחמושת| נשק| תחמושת| אביזר נשק לתחמושת| נשק ותחמושת| אביזר נשק או תחמושת)?'
        offense_full_pattern = pattern_

        # Initialize dictionaries to hold offense features (dict not declared in __init__, example usage)
        self.feature_dict["OFFENCE_NUMBER"] = []
        self.feature_dict["OFFENCE_TYPE"] = []

        # For each row in preprocessing_df, check for offense patterns
        for _, row in preprocessing_df.iterrows():
            text = row.get('text', '')
            
            # Look for legal references and offense types in the text
            verdict_numbers = verdict_number_pattern.findall(text)
            self.compile = re.compile(offense_full_pattern)
            verdict_types = self.compile.findall(text)
            
            # If any verdict numbers are found, add them to feature_dict
            if verdict_numbers:
                self.feature_dict["OFFENCE_NUMBER"].extend([''.join(num) for num in verdict_numbers])
                offences_numbers_output = []
                for offence_number in list(set(verdict_numbers)):
                    if "א" in str(offence_number):
                        offences_numbers_output.append("144 א")
                    elif "ב" in str(offence_number):
                        offences_numbers_output.append("144 ב")
                # Append a row to results_df for each found offense number
                results_df = pd.concat([results_df,
                                        pd.DataFrame({
                                            'feature_key': "OFFENCE_NUMBER",
                                            'text': text,
                                            'extraction': offences_numbers_output
                                        })],
                                       ignore_index=True)

                # For each matched offense type, standardize and log it
                for match in verdict_types:
                    non_empty_elements = [elem for elem in match if elem and len(elem) > 1]
                    stripped_elements = [elem.strip() for elem in non_empty_elements]
                    unique_elements = list(OrderedDict.fromkeys(stripped_elements))
                    concatenated_match = ' '.join(unique_elements).strip()
                    self.feature_dict["OFFENCE_TYPE"].append(concatenated_match)

                    regex_extraction = []
                    # Map the offense short tag to a descriptive name
                    for offences_type in offense_mapping.keys():
                        if offences_type in concatenated_match:
                            regex_extraction.append(offense_mapping[offences_type])

                    # Append a row to results_df for each found offense type
                    results_df = pd.concat([results_df,
                                            pd.DataFrame({
                                                'feature_key': "OFFENCE_TYPE",
                                                'text': text,
                                                'extraction': regex_extraction
                                            })],
                                           ignore_index=True)

            # Extend the OFFENCE_NUMBER list with all found verdict numbers for this row
            self.feature_dict["OFFENCE_NUMBER"].extend([''.join(num) for num in verdict_numbers])

        # Remove duplicates and standardize the OFFENCE_NUMBER list
        self.feature_dict["OFFENCE_NUMBER"] = list(set(self.feature_dict["OFFENCE_NUMBER"]))
        offences_numbers_output = []
        for offence_number in list(set(self.feature_dict["OFFENCE_NUMBER"])):
            if "א" in offence_number:
                offences_numbers_output.append("144 א")
            elif "ב" in offence_number:
                offences_numbers_output.append("144 ב")
        self.feature_dict["OFFENCE_NUMBER"] = offences_numbers_output

        # Build the OFFENCE_TYPE list with proper names based on offense_mapping
        offences_output = []
        for offence in set(self.feature_dict["OFFENCE_TYPE"]):
            for offences_type in offense_mapping.keys():
                if offences_type in offence:
                    offences_output.append(offense_mapping[offences_type])
        self.feature_dict["OFFENCE_TYPE"] = offences_output
        
        return results_df

if __name__ == '__main__':
    # Example usage of the QA_Extractor class:

    # Instantiate a question-answering pipeline (Dicta example)
    model = pipeline('question-answering', model='dicta-il/dictabert-heq')
    model_name = 'cluade'
    # A sample directory that contains a sentence_prediction.csv
    example_case = 'results/db/2017/SH-16-08-7996-293/'

    # Create an instance of QA_Extractor without specifying model/tokenizer
    qa = QA_Extractor()
    # Run extraction on the example case directory
    qa.extract(example_case)
