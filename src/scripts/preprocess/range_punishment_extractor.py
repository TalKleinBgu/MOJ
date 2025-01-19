import os
import re
import pandas as pd
import sys
from pathlib import Path

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import save_json, config_parser
from resources.number_mapping import number_dict
from utils.punishment_extractor import verdict_punishmet_dict, check_numbers_match, replace_words_with_numbers, punishment_range_extract
pattern = "מתח. ה?עו?ני?ש"


# TODO:  split it to extract the right length punishment, and to evaluate


class Punishment_range_extractor:
    def __init__(self, directory_path: str = None):
        self.directory_path = directory_path
        # TODO change the terminology tagged is after classifier and its ambigo (_Tagged)

        self.prediction_path = os.path.join(directory_path, 'sentence_tagging.csv')  # Use the callsifier file.
        self.sentence_punishment_range = self.extract_sentence_punishment_range()
        if self.sentence_punishment_range:
            self.punishment_range = punishment_range_extract(self.sentence_punishment_range)
        else:
            self.punishment_range = 'None'

    def find_sentence_contain_punishment(self):
        """
            Find the punishment, Itrate over _Tagged file of the verdict,
            use the regex pattern and return the last match.

            Returns:
            last regex we catch
        """
        prediction_df = pd.read_csv(self.prediction_path)
        punishment_rows = prediction_df[prediction_df['PUNISHMENT'] == 1]

        # Initialize variables to store the last match and associated text
        last_match = None
        associated_text = None

        # Iterate over rows and apply regex on the 'text' column
        for index, row in punishment_rows.iterrows():
            text = row['text']
            matches = re.findall(pattern, text)
            # matches = get_boundry_punishment(text,pattern)
            if matches:
                last_match = matches[-1]
                associated_text = text

        # Check if there was a match
        if last_match:
            return associated_text
        else:
            return None

    def extract_sentence_punishment_range(self):
        punishment_regex = self.find_sentence_contain_punishment()
        if punishment_regex:
            return replace_words_with_numbers(punishment_regex, number_dict)


def extract_punishment_range_tagging_db(corpus_path):
    df = pd.read_csv('/home/tak/pred-sentencing/results/db/weapon/weapon_range.csv')
    #lst of verdict file column
    lst = df['Verdict File'].tolist()
    for dir_name in os.listdir(corpus_path):
        if dir_name not in lst:
            continue
        directory_path = os.path.join(corpus_path, dir_name)
        directory_path_jason = os.path.join(corpus_path, dir_name, 'punishment_range.json')
        if os.path.isdir(directory_path):
            if os.path.exists(os.path.join(directory_path, 'sentence_tagging.csv')):
                pre = Punishment_range_extractor(directory_path)
                save_json(pre.punishment_range, directory_path_jason)
            else:
                save_json('tagged file de`snt exist', directory_path_jason)


# def evaluate():
#     # output_csv = 'output.csv'  # The name of the output CSV
#     headers = ['Verdict File', 'Verdict Number', 'Tagged Punishment', 'Extracted Punishment', 'Is Match']
#     # create_csv(output_csv, headers)

#     verdict_punishment_dict = verdict_punishmet_dict(manual_feature_extraction_path, mapping)
#     for dir_name in os.listdir(corpus_path):
#         directory_path = os.path.join(corpus_path, dir_name)
#         if os.path.isdir(directory_path):
#             pre = Punishment_range_extractor(directory_path)
#             try:
#                 real_punishment_string = verdict_punishment_dict[dir_name][1]
#             except Exception as e1:
#                 continue
#             try:
#                 is_match = check_numbers_match(real_punishment_string, pre.punishment_range)
#             except Exception as e:
#                 is_match = None
#             data_row = [dir_name, verdict_punishment_dict[dir_name][0], real_punishment_string, pre.punishment_range,
#                         is_match]
            # append_to_csv(output_csv, data_row)


if __name__ == "__main__":
#     params = load_complete_config()  # TODO update this function
#     project_path = Path(os.path.dirname(__file__)).parent.parent.parent
#     corpus_path = params["db_path"]
#     mapping = os.path.join(project_path, 'resources/data/database/2017/mapping.csv')
#     manual_feature_extraction_path = os.path.join(project_path, 'resources/data/tagging/feature_extraction/2017.csv')
    
    extract_punishment_range_tagging_db('/home/tak/pred-sentencing/results/db/weapon')
    # TODO add logger print that finish
