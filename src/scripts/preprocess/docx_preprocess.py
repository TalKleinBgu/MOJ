import os
import re
import pandas as pd
import stanza
from docx import Document
import sys

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)

# sys.path.append('../../') 
# stanza.download('he')
nlp = stanza.Pipeline('he')


from utils.preprocess import (is_block_bold, iterate_block_items,
                   extract_part_after_number_or_hebrew_letter)

def doc_to_csv(doc_path: str = None, result_path: str = None):
    """
    Converts a DOCX document to a CSV format by extracting relevant parts of the document 
    based on specified conditions like block boldness or specific patterns.

    Parameters:
    - doc_path (str, optional): The path to the DOCX document. Defaults to None.

    Steps:
    1. Initialize data dictionary to hold extracted content.
    2. Open and iterate through the provided DOCX document.
    3. Filter out unnecessary blocks.
    4. Determine if the current block is a title or content.
    5. If it's content, tokenize it using the Stanza library.
    6. Add the extracted content to the data dictionary.
    7. Convert the data dictionary to a Pandas DataFrame.

    Returns:
    - DataFrame: A Pandas DataFrame containing the extracted text from the DOCX document with columns 'text' and 'part'.
    """

    data = {'verdict': [],'text': [], 'part': []}
    data['verdict']=os.path.splitext(os.path.basename(doc_path))[0]
    doc_path += '.docx'
    doc = Document(doc_path)
    part = 'nothing'  

    for block in iterate_block_items(doc): # Updated usage
        flag = False

        if len(block.text) <= 1 or 'ע"פ' in block.text or 'ת"פ' in block.text or 'עפ"ג' in block.text:
            continue

        if is_block_bold(block) and len(block.text.split(' ')) < 10 and not re.match(r'^\d', block.text) and not re.match(r'[\u0590-\u05FF][^.)*]*[.)]', block.text):
            part = block.text
        else:
            extracted_part_text = extract_part_after_number_or_hebrew_letter(block.text)
            sentences = nlp(extracted_part_text)

            for sentence in sentences.sentences:
                text = sentence.text
                if text.startswith('"'):
                    flag = True
                    continue
                if text.endswith('".') or text.endswith('"'):
                    flag = False
                    continue
                if flag:
                    continue
                if text == part:
                    continue
                if len(block.text.split(' ')) > 3:
                    data['text'].append(text)
                    data['part'].append(part)

    sentence_doc_df = pd.DataFrame(data)
    #remove duplicates text
    sentence_doc_df = sentence_doc_df.drop_duplicates(subset=['text'])
    #remove last 2 rows
    sentence_doc_df = sentence_doc_df[:-2]
    result_path = os.path.join(result_path, 'preprocessing.csv')
    sentence_doc_df.to_csv(result_path)
    return

def run():
    """Main execution function to process all .docx files in a directory."""
    root_directory = 'C:\\Users\\liork\\pred-sentencing\\resources\\files\\Tag'

    for root, directories, files in os.walk(root_directory):
        print("Current directory:", root)
        for file in files:
            print('--------------')
            path = os.path.join(root, file)
            print(path)
            df = doc_to_csv(doc_path=path) 
            path = os.path.join('C:\\Users\\liork\\pred-sentencing\\resources\\files\\json_csvToTag', file.split('_')[0] + '.csv')
            df.to_csv(path)
    print('--------------')


if __name__ == "__main__":
    run()