import os
import sys
import logging
import pandas as pd
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import config_parser, setup_logger
from utils.loads_data import jsonTag2dict

def preprocessing_flow(path, SEED=42, csv=None, similarity_sampling=False, under_sampling=False, load_xlsx=False,
                       labels: list = None):
    if load_xlsx:
        df = pd.read_csv(path)
        multy_label_dict = {}
    else:
        choise_sentences, multy_label_dict = jsonTag2dict(path)
        # df = dict2df(choise_sentences, SEED)
    # if under_sampling:
        # df = under_sampling(df)
    # dfs = convert_all_labels2binary(df, labels=labels)
    # print(dfs['GENERAL_CIRCUM'])
    # return dfs, df, multy_label_dict
    return multy_label_dict


def create_multilabel_df(multi_label_dict: dict = None, labels: list = []):
    """

    :param multi_label_dict:
    :param labels:
    :return:
    """

    data = {'text': [], 'verdict': []}
    for label in labels:
        data[label] = []
    
    for text, info in multi_label_dict.items():
        t = info['labels']
        data['verdict'].append(info['verdict'])
        data['text'].append(text)
        for label in labels:
            data[label].append(1 if label in info['labels'] else 0)

    df = pd.DataFrame(data)
    # Reorder columns so that 'file' is first, then 'text'
    df = df[['verdict', 'text'] + labels]
    df.insert(0, 'Unnamed: 0', range(1, len(df) + 1))

    return df


def run(labels, source_path, output_path):
    multy_label_dict = preprocessing_flow(source_path)
    i = 0
    for item in multy_label_dict:
        if len(multy_label_dict[item]) > 1:
            print(i, multy_label_dict[item], item)

        i += 1

    df = create_multilabel_df(multi_label_dict=multy_label_dict, labels=labels)

    # add case number base on 
    #to csv - resources/data/tagging/drugs/gt
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main_params = config_parser("", "main_config")
    domain = main_params["domain"]
    params = config_parser("preprocessing", domain)
    
    source_path = params['source_path'].format(domain=domain)
    output_path = params['output_path'].format(domain=domain)
    run(params['labels'], source_path, output_path)
