import os
import pandas as pd
import csv
import anthropic
import re
import time
from tqdm.auto import tqdm

# 'CONFESSION': ('CONFESSION',0)

explain_label_dict = {
    'CIR_TYPE_WEP': ('type of weapon', 3),
    '_HELD_WAY_WEP': ('way the weapon is held received, wich is one of the following:  At home, in the car, on his body, removed, near his house', 7),
    'CIR_AMMU_AMOUNT_WEP': ('amount of ammunition', 7),
    'CIR_PURPOSE': ('purpose of the offense', 10),
    'CIR_STATUS_WEP': ('weapon status, wich is one of the following: loaded, primed, malfunctioning, disassembled', 5),
    'CIR_PLANNING': ('planning that preceded the commission of the crime', 14),
    'CIR_OBTAIN_WAY_WEP': ('wich is one of the following: found, stole, purchased, produced', 10),
    'CIR_USE': ('use made of the weapon wich is one of the following: Shooting, attempting to shoot, unsuccessfully attempting to shoot, throwing a grenade, activating a charge', 8),
    'RESPO': ('taking responsibility', 7),
    'PUNISHMENT': ('punishment of the accused can be a final punishment or a punishment range', 10)
}

def read_excel(input_file):
    return pd.read_excel(input_file)

def initialize_client(api_key):
    return anthropic.Anthropic(api_key=api_key)

def ll_response(client, original_sentence, temperature, label):
    
    if label not in explain_label_dict.keys():
        return None
    
    label_explenation, num_sentences = explain_label_dict[label]
    message = f"""
                You are an expert in the field of law, specializing in cases involving weapons.
                Your task is to create additional sentences similar to a given original sentence,
                These sentences will be used to grow a dataset for training a classification model.

                Your task is to create {num_sentences} new sentences that are variations of the original sentence,
                that will be different from each other.
                When creating the sentence focus on the category of {label_explenation}.

                Display the generated sentences in the following format:
                <generated_senses>
                1. [first sentence created]
                2. [second sentence created]
                ...
                {num_sentences}. [last generated sentence]
                </generated_senses>
                
                
                Here is the original sentence:
                <original_sentence>
                {original_sentence}
                </original_sentence>
                """
                
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        top_k=3,
        top_p=0.9,
        temperature=temperature,
        messages=[{"role": "user", "content": message}]
    )
    generated_text = message.content[0].text
    pattern = re.compile(r'\d+\.\s+(.+?)(?=\n\d+\.|\n</generated_senses>)', re.DOTALL)
    sentences = [sentence.strip() for sentence in pattern.findall(generated_text)]
    return sentences

def __generate_sentences(output_dir, sentence_tagging_df, client, num_sentences, temperature):
    label_columns = sentence_tagging_df.columns[4:]
    max_calls_per_minute = 49
    delay_between_calls = 61 / max_calls_per_minute
    api_calls = 0
    label_counts = {label: 0 for label in label_columns}
    
    progress_bar = tqdm(total=len(sentence_tagging_df))

    for idx, row in sentence_tagging_df.iterrows():
        original_sentence = row['text']
        for label in label_columns:
            if row[label] == 1:
                if api_calls >= max_calls_per_minute:
                    time.sleep(60)
                    api_calls = 0
                generated_sentences = ll_response(client, original_sentence, temperature, label)
                  
                if generated_sentences is None:
                    continue
                
                label_file = os.path.join(output_dir, f"{label}_generated_sentences.csv")
                file_exists = os.path.isfile(label_file)
                with open(label_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(["Sentence", "Type"])
                    writer.writerow([original_sentence, "original"])
                    for sentence in generated_sentences:
                        writer.writerow([sentence, "generated"])
                    label_counts[label] += len(generated_sentences)
                api_calls += 1
                time.sleep(delay_between_calls)

        progress_bar.update(1)

    print("\nSummary of generated sentences per label:")
    for label, count in label_counts.items():
        print(f"Label: {label}, Sentences generated:ש {count}")

def run():
    input_file = 'resources/data/tagging/gt/sentence_tagging.xlsx'
    output_dir = 'results/sentence_generation/T0.9_K3_P0.9_v2'
    ANTHROPIC_API_KEY = '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'
    NUM_SENTENCES = 10
    temperature = 0.9
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sentence_tagging_df = read_excel(input_file)
    print("Loaded Excel file.")
    client = initialize_client(ANTHROPIC_API_KEY)
    print("Initialized Anthropics client.")
    __generate_sentences(output_dir, sentence_tagging_df, client, NUM_SENTENCES, temperature)

if __name__ == '__main__':
    run()
