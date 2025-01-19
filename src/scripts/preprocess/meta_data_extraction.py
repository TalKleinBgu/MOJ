import os
import re
import pandas as pd
import stanza
import sys
from docx import Document

sys.path.append('../../')
stanza.download('he')
nlp = stanza.Pipeline('he')

from utils.preprocess import (
    iterate_block_items,
    extract_violations,
    count_consecutive_blocks_starting_with_number,
    extract_name_after_word
)


def process_document(doc_path: str) -> pd.DataFrame:
    """
    Process a given Word document to extract specific legal information.
    
    Steps:
    1. Initialize data structures for capturing extracted data and load the Word document.
    2. Define flags to ensure each section (like court info or judge name) is processed only once.
    3. Iterate over blocks of text in the document.
       a. If a block contains court information and hasn't been processed, extract and store it.
       b. If a block indicates the number of accused and hasn't been processed, determine the count and store it.
       c. If a block contains a date pattern and hasn't been processed, extract the date and store it.
       d. If a block has judge information and hasn't been processed, extract and store the judge's name.
       e. If a block cites legislation and hasn't been processed, extract the subsequent block as the legislation reference.
       f. Continuously check for and extract violation details from blocks until they have all been processed.
    4. Convert the extracted data into a pandas DataFrame and return it.

    Args:
    - doc_path (str): Path to the Word document to be processed.

    Returns:
    - pd.DataFrame: DataFrame containing extracted textual snippets and their associated labels.
    """

    data = {'text': [], 'part': []}
    doc = Document(doc_path)
    
    court_processed = False
    num_accused_processed = False
    date_processed = False
    judge_processed = False
    legislation_processed = False
    violations_processed = False
    violations_block=0;
    blocks = list(iterate_block_items(doc)) 

    for idx, block in enumerate(blocks):

        if ('בית המשפט' in block.text or 'בית משפט' in block.text) and not court_processed:
            data['text'].append(block.text)
            data['part'].append('בית המשפט')
            court_processed = True

        if 'נגד' in block.text and not num_accused_processed:
            count = count_consecutive_blocks_starting_with_number(blocks[idx+1:])
            data['text'].append(count)
            data['part'].append('מספר הנאשמים')
            num_accused_processed = True

        date_pattern = r'ת"פ (\d{5}|\d{4})-(\d{2}-\d{2})'
        date_match = re.search(date_pattern, block.text)
        if date_match and not date_processed:
            data['text'].append(date_match.group(2))
            data['part'].append('תאריך')
            date_processed = True

        judge_info = extract_name_after_word(block.text, "השופטת?")
        if judge_info and not judge_processed:
            data['text'].append(judge_info)
            data['part'].append('שופט')
            judge_processed = True

        if 'חקיקה שאוזכרה' in block.text and not legislation_processed:
            data['text'].append(blocks[idx+1].text)
            data['part'].append('חקיקה שאוזכרה')
            legislation_processed = True


        if(not violations_processed or idx-3<=violations_block):
            violations=extract_violations(block.text)
            for violation in violations:
                data['text'].append(violation)
                data['part'].append('עבירות')
            if len(violations)>0:
                violations_processed=True
                violations_block=idx

    return pd.DataFrame(data)

def main():
    """Main function to process all Word documents in a specified directory."""
    root_directory = 'C:\\Users\\liork\\pred-sentencing\\resources\\data\\DocxCases'
    output_directory = 'C:\\Users\\liork\\pred-sentencing\\resources\\data\\csv2'

    for root, _, files in os.walk(root_directory):
        print(f"Processing directory: {root}")

        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_directory, file.split('_')[0] + '.csv')

            print(f"Converting {input_path} to {output_path}")
            
            df = process_document(doc_path=input_path)
            df.to_csv(output_path)
    
    print("Conversion completed.")

if __name__ == "__main__":
    main()
