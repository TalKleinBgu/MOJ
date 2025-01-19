import os
import sys
import pandas as pd
import re
from collections import OrderedDict

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir,  '..', '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import get_date, reformat_sentence_tagged_file


class RegexExtractor:
    """
    Class for extracting features from verdict documents.
    """

    def __init__(self, data_path=None):
        """
        Initialize the FeatureExtractor with the path to the verdict document.

        Parameters:
        - verdict_path (str): Path to the verdict document folder.
        """
        self.data_path = data_path
        self.feature_dict = {"SALE2AGENT": False,"USE": False,"CONFESSION": False, "PUNISHMENT": [], "AMMO_AMOUNT": [], "HELD_WAY": [],
                             "TYPE_WEP": [],"MONEY_PAID":[],"OBTAIN_WAY":[],"STATUS_WEP":[],"OFFENCE_TYPE":[],"OFFENCE_NUMBER":[]}


    def extract(self, save_path, db_flow):
        """
        Main method to extract features from the verdict document. It processes tagged data to identify and extract various legal features based on predefined patterns.

        Returns:
            pd.DataFrame: A DataFrame with columns ['verdict', 'feature_key', 'text', 'regex_extraction'] containing the extracted feature information.
        """
        feature_keyes = []
        sentences = []
        extractions = []
        data = reformat_sentence_tagged_file(self.data_path)
        results_df = pd.DataFrame(columns=['feature_key', 'text', 'extraction'])
        for _, row in data.iterrows():
            sentence = row['text']
            label = row['label']
            func_name = f'extract_{label}'

            if hasattr(self, func_name):
                func = getattr(self, func_name)
                FK = func(sentence)
                try:
                    features=self.feature_dict[FK]
                except:
                    continue

                if isinstance(features,list):
                    features=list(set(features))
                
                feature_keyes.append(FK)
                sentences.append(sentence)
                extractions.append(features)
                
        results_df =  pd.DataFrame({
            'feature_key':feature_keyes,
            'text': sentences,
            'extraction': extractions
        })
        if db_flow:
            results_df = self.extract_offense_info(results_df)
        
        save_path = os.path.join(save_path, f'{get_date()}_regex_feartures.csv')
        results_df.to_csv(save_path)
        print(f'save fe regex to {save_path}')
        return results_df

    def extract_offense_info(self,results_df):
        """
        Extracts information about offenses from the verdict document using regex patterns to match specific legal terminology and categorizes them accordingly.

        Args:
            results_df (pd.DataFrame): DataFrame to which the offense information will be added.

        Returns:
            pd.DataFrame: Updated DataFrame with offense information added.
        """

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
        offense_mapping = {
            "רכיש": "סחר בנשק",
            "חזק": "החזקה נשק",
            "נשיא": "נשיאת נשק",
            "הובל": "הובלת נשק",
            "עסק": "סחר בנשק",
            "סחר": "סחר בנשק"
        }

        preprocessing_df = pd.read_csv(os.path.join(self.data_path, "preprocessing.csv"))
        verdict_number_pattern = re.compile(r'(144\s*\(?\s*[אב]\d*\s*\)?)|(340א)')
        offense_combined_pattern = '|'.join(offense_patterns)
        offense_combined_with_qualifiers_pattern = f'({offense_combined_pattern})( ו{offense_combined_pattern})*'
        pattern_ = f'({offense_combined_with_qualifiers_pattern})( של נשק| של תחמושת| נשק| תחמושת| אביזר נשק לתחמושת| נשק ותחמושת| אביזר נשק או תחמושת)?'
        offense_full_pattern = pattern_

        self.feature_dict["OFFENCE_NUMBER"] = []
        self.feature_dict["OFFENCE_TYPE"] = []

        for _, row in preprocessing_df.iterrows():
            text = row.get('text', '')

            verdict_numbers = verdict_number_pattern.findall(text)
            self.compile = re.compile(offense_full_pattern)
            verdict_types = self.compile.findall(text)
            if verdict_numbers:
                self.feature_dict["OFFENCE_NUMBER"].extend([''.join(num) for num in verdict_numbers])
                offences_numbers_output = []
                for offence_number in list(set(verdict_numbers)):
                    if "א" in str(offence_number):
                        offences_numbers_output.append("144 א")
                    elif "ב" in str(offence_number):
                        offences_numbers_output.append("144 ב")
                results_df = results_df.append({
                    'feature_key': "OFFENCE_NUMBER",
                    'text': text,
                    'extraction': offences_numbers_output
                }, ignore_index=True)

                # Process each match to concatenate non-empty elements
                for match in verdict_types:
                    non_empty_elements = [elem for elem in match if elem and len(elem) > 1]
                    stripped_elements = [elem.strip() for elem in non_empty_elements]
                    unique_elements = list(OrderedDict.fromkeys(stripped_elements))
                    concatenated_match = ' '.join(unique_elements).strip()
                    self.feature_dict["OFFENCE_TYPE"].append(concatenated_match)

                    regex_extraction=[]
                    for offences_type in offense_mapping.keys():
                        if offences_type in concatenated_match:
                            regex_extraction.append(offense_mapping[offences_type])

                    results_df = results_df.append({
                        'feature_key': "OFFENCE_TYPE",
                        'text': text,
                        'extraction': regex_extraction
                    }, ignore_index=True)

            self.feature_dict["OFFENCE_NUMBER"].extend([''.join(num) for num in verdict_numbers])

        self.feature_dict["OFFENCE_NUMBER"] = list(set(self.feature_dict["OFFENCE_NUMBER"]))

        offences_numbers_output = []
        for offence_number in list(set(self.feature_dict["OFFENCE_NUMBER"])):
            if "א" in offence_number:
                offences_numbers_output.append("144 א")
            elif "ב" in offence_number:
                offences_numbers_output.append("144 ב")

        self.feature_dict["OFFENCE_NUMBER"] = offences_numbers_output

        offences_output = []


        for offence in set(self.feature_dict["OFFENCE_TYPE"]):
            for offences_type in offense_mapping.keys():
                if offences_type in offence:
                    offences_output.append(offense_mapping[offences_type])

        self.feature_dict["OFFENCE_TYPE"] = offences_output

        if self.feature_dict["USE"] is False:
            if "ירי" in self.feature_dict["OFFENCE_TYPE"] or "340א" in self.feature_dict["OFFENCE_NUMBER"]:
                self.feature_dict["USE"] = True
                results_df = results_df.append({
                    'feature_key': "USE",
                    'text': text,
                    'extraction': True
                }, ignore_index=True)
        return results_df

    def extract_CONFESSION(self, text):
        confession = [
            "הודאתו",
            "הודה הנאשם",
            "הודעתו",
            "הודה",
            'הודייתו'
        ]

        pattern = '|'.join(map(re.escape, confession))
        regex = re.compile(pattern, re.IGNORECASE)
        matches = regex.findall(text)


        if self.feature_dict["CONFESSION"] is False:
            if matches:
                self.feature_dict["CONFESSION"] = 'כן'
        return "CONFESSION"


    def extract_CONFESSION_LVL2(self, text):
        return self.extract_CONFESSION(text)

    # def extract_PUNISHMENT(self, text):
    #     format1 = re.search(r'(\d+(\.\d+)?)[ ]*([שבנ]נ?ים|חודשי?)[ ]*מאסר', text)
    #     format2 = re.search(r'בין[ ]*(\d+)[ ]*ל[ ]*(\d+)[ ]*([שבנ]נ?ים|חודשי?)[ ]*מאסר', text)
    #     format3 = re.search(r'בין[ ]*([\d.]+)[ ]*ל[ ]*([\d.]+)[ ]*שנות[ ]*מאסר', text)
    #     format4 = re.search(r'(\S+)[ ]*חודשי', text)
    #     if format1:
    #         result1 = format1.group(0)
    #         # print(result1)
    #         self.feature_dict["PUNISHMENT"].append(result1)
    #     if format2:
    #         result2 = format2.group(0)
    #         # print(result2)
    #         self.feature_dict["PUNISHMENT"].append(result2)
    #     if format3:
    #         result3 = format3.group(0)
    #         # print(result3)
    #         self.feature_dict["PUNISHMENT"].append(result3)
    #     if format4:
    #         length_of_punishment = format4.group(1)
    #         # print(f"Length of punishment: {length_of_punishment}")
    #         self.feature_dict["PUNISHMENT"].append(length_of_punishment)
    #     return "PUNISHMENT"

    def extract_CIR_AMMU_AMOUNT_WEP(self, text):
        format1 = re.search(r'(\S+)[ ]*(מחסניות)', text)
        if format1:
            quantity = format1.group(1)
            unit = format1.group(2)
            # print(f"{quantity} {unit}")
            self.feature_dict["AMMO_AMOUNT"].append(f"{quantity} {unit}")
        if re.search(r'מחסנית ריקה', text):
            self.feature_dict["AMMO_AMOUNT"].append("מחסנית ריקה")
        if re.search(r'מחסנית(?! ריקה)', text):
            self.feature_dict["AMMO_AMOUNT"].append("מחסנית")
        match = re.search(r'(\S+)[ ]*כדורים', text)
        if match:
            preceding_word = match.group(1)
            # print(f"{preceding_word} כדורים ")
            self.feature_dict["AMMO_AMOUNT"].append(f"{preceding_word} כדורים ")
        match2 = re.search(r'(\S+)[ ]*קליעים', text)
        if match:
            preceding_word = match.group(1)
            # print(f"{preceding_word} כדורים ")
            self.feature_dict["AMMO_AMOUNT"].append(f"{preceding_word} קליעים ")
        self.feature_dict["AMMO_AMOUNT"] = list(set(self.feature_dict["AMMO_AMOUNT"]))

        return "AMMO_AMOUNT"

    def extract_CIR_BUYER_ID_WEP(self, text):
        if self.feature_dict["SALE2AGENT"] is False:
            regex_pattern = r"\bסוכן\b"
            match = re.search(regex_pattern, text)
            if match:
                self.feature_dict["SALE2AGENT"] = True
        return "SALE2AGENT"

    def extract_CIR_HELD_WAY_WEP(self, text):
        held_way = [
            "רכב",
            "בית",
            "תא מטען",
            "גופו",
            "נשא",
            "אופנוע",
            "קטנוע",
           "מחסן",
            "מקלט",

        ]

        pattern = '|'.join(map(re.escape, held_way))
        regex = re.compile(pattern, re.IGNORECASE)
        matches = regex.findall(text)
        self.feature_dict["HELD_WAY"].extend(matches)
        self.feature_dict["HELD_WAY"] = list(set(self.feature_dict["HELD_WAY"]))

        return "HELD_WAY"

    def extract_CIR_MONEY_PAID_WEP(self, text):
        pattern = "(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s₪)"
        regex = re.compile(pattern, re.IGNORECASE)
        matches = regex.findall(text)
        matches_with_currency = [match.replace("₪", "שקלים") for match in matches]
        self.feature_dict["MONEY_PAID"].extend(matches_with_currency)
        self.feature_dict["MONEY_PAID"] = list(set(self.feature_dict["MONEY_PAID"]))

        return "MONEY_PAID"

    def extract_CIR_OBTAIN_WAY_WEP(self, text):
        obtain_way = [
            "רכש",
            "קנה"
            "גנב",
            "קיבל",
            "מצא",
            "ייצר"
        ]

        pattern = '|'.join(map(re.escape, obtain_way))
        regex = re.compile(pattern, re.IGNORECASE)
        matches = regex.findall(text)
        self.feature_dict["OBTAIN_WAY"].extend(matches)
        self.feature_dict["OBTAIN_WAY"] = list(set(self.feature_dict["OBTAIN_WAY"]))

        return "OBTAIN_WAY"
    
    def extract_CIR_PLANNING(self, text):
        # ... your feature extraction logic for PUR_CIR ...
        pass

    def extract_CIR_PURPOSE(self, text):
        # ... your feature extraction logic for PUR_CIR ...
        pass

    def extract_CIR_STATUS_WEP(self, text):
        status = [
            "מפורק","טעון במחסנית","מחסנית בהכנס",
            "כדור בקנה","דרוך","טעון",
            "ובתוכו מחסנית","תקול",
            "ירי","ירייה",
            "בצד מחסנית"
        ]
        status_mapping = {
            "מפורק": "נשק מפורק",
            "טעון במחסנית": "נשק עם מחסנית בהכנס",
            "מחסנית בהכנס": "נשק עם מחסנית בהכנס",
            "כדור בקנה": "נשק עם כדור בקנה",
            "דרוך": "נשק עם כדור בקנה",
            "טעון": "נשק עם מחסנית בהכנס",
            "ובתוכו מחסנית": "נשק עם מחסנית בהכנס",
            "ירי": "נשק עם כדור בקנה",
            "ירייה": "נשק עם כדור בקנה",
            "תקול": "תקול",
            "בצד מחסנית": "נשק מופרד מתחמושת"
        }
        pattern = '|'.join(map(re.escape, status))
        regex = re.compile(pattern, re.IGNORECASE)
        matches = regex.findall(text)
        descriptive_statuses = [status_mapping[match] for match in matches]
        # if not descriptive_statuses: descriptive_statuses = ["תקין"]
        self.feature_dict["STATUS_WEP"].extend(descriptive_statuses)
        self.feature_dict["STATUS_WEP"] = list(set(self.feature_dict["STATUS_WEP"]))

        return "STATUS_WEP"

    def extract_CIR_TYPE_WEP(self, text):
        weapon_types = [
            "תת מקלע מאולתר","קרל גוסטב","רובה מאולתר","קרלו",
            "תת-מקלע","תתי מקלע","אקדח", "ברטה",
            "יריחו","סטאר","M16", "FN",
            "רימון רסס","רימון גז", "רימון", "רימוני",
            "עוזי","CZ","cz","טיל לאו","טילי לאו",
            "רובה",
            "חלוואני",
            "סירקיס",
            "לבנת חבלה",
            "פאראבלום",
            "ROHM",
            "זיג זאוור",
            "קולט 45",
            "גליל",
            "מיקרו תבור",
            "STYER",
             "צייד"]
        pattern = '|'.join(map(re.escape, weapon_types))
        regex = re.compile(pattern)

        matches = regex.findall(text)

        self.feature_dict["TYPE_WEP"].extend(matches)
        self.feature_dict["TYPE_WEP"] = list(set(self.feature_dict["TYPE_WEP"]))
        return "TYPE_WEP"

    def extract_CIR_USE(self, text):
        if self.feature_dict["USE"] is False:
            regex_pattern = r"\bירי\b"
            match = re.search(regex_pattern, text)
            if match:
                self.feature_dict["USE"] = True
        return "USE"

    def extract_offense_info_no_cls(self,text):
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
        offense_mapping = {
            "רכיש": "סחר בנשק",
            "חזק": "החזקה נשק",
            "נשיא": "נשיאת נשק",
            "הובל": "הובלת נשק",
            "עסק": "סחר בנשק",
            "סחר": "סחר בנשק"
        }
        verdict_number_pattern = re.compile(r'(144\s*\(?\s*[אב]\d*\s*\)?)|(340א)')
        offense_combined_pattern = '|'.join(offense_patterns)
        offense_combined_with_qualifiers_pattern = f'({offense_combined_pattern})( ו{offense_combined_pattern})*'
        pattern_ = f'({offense_combined_with_qualifiers_pattern})( של נשק| של תחמושת| נשק| תחמושת| אביזר נשק לתחמושת| נשק ותחמושת| אביזר נשק או תחמושת)?'
        offense_full_pattern = pattern_

        verdict_numbers = verdict_number_pattern.findall(text)
        self.compile = re.compile(offense_full_pattern)
        verdict_types = self.compile.findall(text)
        if verdict_numbers:
            self.feature_dict["OFFENCE_NUMBER"].extend([''.join(num) for num in verdict_numbers])
            offences_numbers_output = []
            for offence_number in list(set(verdict_numbers)):
                if "א" in str(offence_number):
                    offences_numbers_output.append("144 א")
                elif "ב" in str(offence_number):
                    offences_numbers_output.append("144 ב")

            # Process each match to concatenate non-empty elements
            for match in verdict_types:
                non_empty_elements = [elem for elem in match if elem and len(elem) > 1]
                stripped_elements = [elem.strip() for elem in non_empty_elements]
                unique_elements = list(OrderedDict.fromkeys(stripped_elements))
                concatenated_match = ' '.join(unique_elements).strip()
                self.feature_dict["OFFENCE_TYPE"].append(concatenated_match)

        offences_numbers_output = []
        for offence_number in list(set(self.feature_dict["OFFENCE_NUMBER"])):
            if "א" in offence_number:
                offences_numbers_output.append("144 א")
            elif "ב" in offence_number:
                offences_numbers_output.append("144 ב")

        self.feature_dict["OFFENCE_NUMBER"] = offences_numbers_output

        offences_output = []


        for offence in set(self.feature_dict["OFFENCE_TYPE"]):
            for offences_type in offense_mapping.keys():
                if offences_type in offence:
                    offences_output.append(offense_mapping[offences_type])

        self.feature_dict["OFFENCE_TYPE"] = offences_output
        return "OFFENCE_TYPE","OFFENCE_NUMBER"

    def extract_without_label(self,sentence_df):
        for i, row in sentence_df.iterrows():
            self.extract_CONFESSION(row.text)
            self.extract_CIR_USE(row.text)
            self.extract_CIR_TYPE_WEP(row.text)
            self.extract_CIR_STATUS_WEP(row.text)
            self.extract_CIR_OBTAIN_WAY_WEP(row.text)
            self.extract_CIR_MONEY_PAID_WEP(row.text)
            self.extract_CIR_HELD_WAY_WEP(row.text)
            self.extract_CIR_BUYER_ID_WEP(row.text)
            self.extract_CIR_AMMU_AMOUNT_WEP(row.text)
            self.extract_PUNISHMENT(row.text)
            self.extract_CONFESSION_LVL2(row.text)
            self.extract_offense_info_no_cls(row.text)

    # def extract_manual_db(self,row):
    #         for feature in features_list:
    #             if feature == "full_verdict": continue
    #             csv = translation_dict[feature]
    #             text = row[csv]
    #             if pd.isnull(text):  # Skip empty cells
    #                 continue
    #             else:
    #                 if feature=="TYPE_WEP":
    #                     self.extract_CIR_TYPE_WEP(text)
    #                 elif feature=="OBTAIN_WAY":
    #                     self.extract_CIR_OBTAIN_WAY_WEP(text)
    #                 elif feature=="MONEY_PAID":
    #                     self.extract_CIR_MONEY_PAID_WEP(text)
    #                 elif feature=="AMMO_AMOUNT":
    #                     self.extract_CIR_AMMU_AMOUNT_WEP(text)
    #                 elif feature=="HELD_WAY":
    #                     self.extract_CIR_HELD_WAY_WEP(text)
    #                 else:
    #                     self.feature_dict[feature]=text


def read_docx2csv(path):
    docs = []
    for filename in os.listdir(path):
        if os.path.isdir(os.path.join(path, filename)) and filename not in  ['outputs', 'embedding']:
            sentence_df = pd.read_csv(os.path.join(path, filename, 'docx2csv.csv'))
            docs.append((filename, sentence_df))

    return docs


if __name__ == '__main__':
    """
    Main execution block for processing documents and extracting features.
    feature extraction for data after cls

    """
    example_case = 'results/db/2017/SH-16-08-7996-293/'
    RE = RegexExtractor(example_case)
    RE.extract()
    dict_result = RE.feature_dict

    