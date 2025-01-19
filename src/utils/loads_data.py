import os
import json
import pandas as pd

def load(json_file):
    data = []
    # open the file and load its contents
    with open(json_file, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def jsonTag2dict(tag_file_path):
    """
    Parse JSON tagged data files and convert them into a Python dictionary.

    Parameters:
        tag_file_path (str): The file path for the directory containing the JSON tagged data files.

    Returns:
        tuple:
            choise_sentences (dict): A dict mapping each label to a list of sentences.
            multy_label_dict (dict): A dict mapping each sentence to a dictionary with 'labels' and 'file'.
                                     E.g. {
                                         "Some sentence text": {
                                             "labels": ["LABEL1", "LABEL2"],
                                             "file": "source_filename.json"
                                         },
                                         ...
                                     }
    """
    sentences_choice = {}
    choise_sentences = {'reject': []}

    def read_file_with_encoding(file_path):
        """Read file content with the correct encoding."""
        for encoding in ['utf-16', 'utf-8']:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.readlines(), encoding
            except UnicodeError:
                continue
        raise ValueError(f"Unable to decode file: {file_path}")

    for filename in os.listdir(tag_file_path):
        if filename == '.ipynb_checkpoints':
            continue
        file_name = filename.split('_')[0]
        file_path = os.path.join(tag_file_path, filename)
        try:
            # Try reading the file with an appropriate encoding
            lines, encoding_used = read_file_with_encoding(file_path)
            print(f"Successfully read {filename} with encoding {encoding_used}")

            for line in lines:
                # Load the JSON data from the line
                line_data = json.loads(line)
                text = line_data['text']
                
                # Initialize entry if not present
                if text not in sentences_choice:
                    sentences_choice[text] = {
                        'labels': [],
                        'verdict': file_name  # store the entire filename, or just file_name if you prefer
                    }

                if line_data['answer'] == 'reject':
                    if 'reject' not in sentences_choice[text]['labels']:
                        sentences_choice[text]['labels'].append('reject')
                    choise_sentences['reject'].append(text)

                for choice in line_data.get('accept', []):
                    if choice not in sentences_choice[text]['labels']:
                        sentences_choice[text]['labels'].append(choice)
                    
                    if choice not in choise_sentences:
                        choise_sentences[choice] = [text]
                    else:
                        if text not in choise_sentences[choice]:
                            choise_sentences[choice].append(text)

        except ValueError as e:
            print(f"Error processing file {filename}: {e}")
    
    multy_label_dict = sentences_choice
    return choise_sentences, multy_label_dict
