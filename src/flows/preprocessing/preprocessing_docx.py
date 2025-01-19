import os
import sys
import logging
import pandas as pd
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import config_parser, setup_logger
from scripts.preprocess import docx_preprocess, tagging_preprocess


class Preprocessing_flow():

    def __init__(self, source_path:str = None, result_path:str = None, 
                 prepare_to_tagging:bool = True, logger:logging.Logger = None,
                 domain:str = None) -> None:
        
        """
        -----------
        source_path : 
            Path to the directory containing input data files or a single data file.
        prepare_to_tagging : 
            Indicates whether the preprocessing step should prepare the data for tagging.
        """
        
        self.logger = logger
        self.source_path = source_path
        self.result_path = result_path
        self.prepare_to_tagging = prepare_to_tagging
        self.domain = domain
        
        if self.logger:
            if not self.result_path:
                self.logger.warning("Output path is not provided.")
                
        # Check if source_path is provided
        if not self.source_path:
            error_message = "Data path is mandatory. Please provide a valid data path."
            if self.logger:
                self.logger.error(error_message)
            raise ValueError(error_message)
        
        self.logger.info("Start Preprocessing flow!")
        
    def convert_csv_to_json(self, csv_path, output_tagging_path):
        """
        Convert a CSV file to JSON format (for prodigi input (tagging)).

        Args:
            csv_path (str): The path to the CSV file.

        Returns:
            str: The path to the saved JSON file.
        """

        csv_preprocess = tagging_preprocess.Csv2json_conversion(csv_path, output_tagging_path)
        ready2tag_json = csv_preprocess.convert_csv_to_json()
        return ready2tag_json
    
    def directory_preprocess(self):
        """
        This function over all docx directory in data path and preform preprocess,
        than save the .csv output (text|part) to output path
        """
        
        for file_name in os.listdir(self.source_path):
            # if year.startswith('.'):
            #     continue
            # year_path = os.path.join(self.source_path, year)
            # for file_name in os.listdir(year_path):
            if file_name.endswith('.docx'):
                case_name = file_name.replace('.docx', '')
                case_path = os.path.join(self.source_path, case_name)
                output_csv_path = os.path.join(self.result_path, 'db', self.domain, case_name)
                if not os.path.exists(output_csv_path):
                    os.makedirs(output_csv_path)
                    docx_preprocess.doc_to_csv(case_path, output_csv_path)
                else:
                    preprocessing_csv_path = os.path.join(output_csv_path, 'preprocessing.csv')
                    tagged_file = pd.read_csv('/home/tak/pred-sentencing/resources/data/tagging/drugs/gt/sentence_tagging.csv')
                    tagged_file = tagged_file[tagged_file['verdict'] == case_name]
                    #for text row in preprocessing_csv if not in tagged_file remove it
                    if not tagged_file.empty:
                        tagged_file = tagged_file['text']
                        preprocessing_csv = pd.read_csv(preprocessing_csv_path)
                        preprocessing_csv = preprocessing_csv[preprocessing_csv['text'].isin(tagged_file)]
                        if len(preprocessing_csv) != len(tagged_file):
                            self.logger.warning(f" rows from {case_name} case")
                        preprocessing_csv.to_csv(preprocessing_csv_path, index=False)
                    if not os.path.exists(preprocessing_csv_path):
                        docx_preprocess.doc_to_csv(case_path, output_csv_path)
                
                if self.prepare_to_tagging:
                    json_name = case_name + ".json"
                    output_tagging_path = os.path.join(self.result_path, 'doc2tag', json_name)
                    
                    self.convert_csv_to_json(output_csv_path, output_tagging_path)
                
                self.logger.info(f"Finish to process {case_name} case")

                                
    def preprocessing_docx(self):    
        #case one: source_path is directory and we over on each case
        if os.path.isdir(self.source_path): 
            self.directory_preprocess()
            
        # case two: we want to process one doc
        else:
            
            verdict_num = os.path.splitext(os.path.basename(self.source_path))[0] 
            output_csv_path = os.path.join(self.result_path, 'db', self.domain,
                                           verdict_num, 'preprocessing.csv')
            docx_preprocess.doc_to_csv(self.source_path, 
                                       output_csv_path)
            if self.prepare_to_tagging:
                json_name = verdict_num + ".json"
                output_tagging_path = os.path.join(self.result_path, 'doc2tag', json_name)
                self.convert_csv_to_json(output_csv_path, output_tagging_path)
            
            self.logger.info(f"\nSave in {output_csv_path}")


    
    

def main(source_path, result_path, prepare_to_tagging, domain):
    logger = setup_logger(save_path=os.path.join(result_path, 'logs'),
                          file_name='preprocess_test')
    source_path = source_path + "/" + domain
    pp = Preprocessing_flow(source_path=source_path,
                            result_path=result_path,
                            prepare_to_tagging=prepare_to_tagging,
                            domain=domain,
                            logger=logger)
    
    pp.preprocessing_docx()

if __name__ == '__main__':
    params = config_parser("preprocessing")
    main(params['source_path_docx'], params['result_path'],
         params['prepare_to_tagging'], params['domain'])
    


