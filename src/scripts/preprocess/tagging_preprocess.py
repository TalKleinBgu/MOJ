import csv
import json

class Csv2json_conversion:
    """
    A utility class to convert a CSV file into a JSON format.
    
    Attributes:
        csv_file_path (str): Path to the input CSV file.
        json_file_path (str): Path to save the resulting JSON file.
    """

    def __init__(self, csv_path: str = None, save_path: str = None):
        self.csv_file_path = csv_path
        self.json_file_path = save_path

    def convert_csv_to_json(self):
        """
        Converts the content of the CSV file to JSON format and saves it to the specified location.
        """
        data = []

        with open(self.csv_file_path, "r", encoding="utf-8") as csv_file:  # Specify the encoding
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # Only process rows with a "text" column
                if 'text' in row:
                    data.append({"text": row["text"]})

        with open(self.json_file_path, "w", encoding="utf-8") as json_file:  # Specify the encoding
            json.dump(data, json_file, indent=4, ensure_ascii=False)  # Ensure ASCII characters are not escaped

def run():
    csv_converter = Csv2json_conversion(
        csv_path='C:\\Users\\liork\\pred-sentencing\\resources\\data\\json_csvToTag\\ME-20-12-52077-379.docx.csv',
        save_path='C:\\Users\\liork\\pred-sentencing\\resources\\data\\json_csvToTag\\ME-20-12-52077-379.json'
    )
    csv_converter.convert_csv_to_json()

if __name__ == '__main__':
    run()
