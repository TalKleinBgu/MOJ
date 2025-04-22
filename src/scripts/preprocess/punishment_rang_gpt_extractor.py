import openai
import os
import pandas as pd

import os
import re
import pandas as pd
import sys
from pathlib import Path
pattern = "מתח. ה?עו?ני?ש"

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)
import json 

def extract_punishment_range_tagging_db(directory_path: str = None):
    for folder in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                all_punishment_ranges = []
                if file.startswith('pre') and file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)
                    for index, row in df.iterrows():
                        text = row['text']
                        matches = re.findall(pattern, text)
                        if matches:

                            messege_judge = f"""במשפט הבא תענה לי במילה אחת מי אמר אותו - המאשימה/מטעם הנאשם/השופט: 
                            
                            
                            {row['text']}
                            """ 
                            response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "user", "content": messege_judge}
                                    ]   
                            )
                            answer = response.choices[0].message.content  
                            if 'שופט' in answer:
                                messege = """אני אביא לך משפט ואתה צריך להביא את המתחם ענישה שרשום.
                                תענה בפורמט הבא :
                                {מספר} חודשים - {מספר} חודשים.
                                אם אין מספר מובהק תרשום 0.
                                אם אין מתחם ענישה תחזיר רק את המילה לא.

                                המשפט הוא : 

                                """
                                messege += row['text']
                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "user", "content": messege}
                                    ]
                                )
                                answer = response.choices[0].message.content
                                if 'לא' not in answer and len(answer) > 4:
                                    all_punishment_ranges.append(answer)
                    if all_punishment_ranges:
                        # Save the results to a JSON file
                        output_path = os.path.join(folder_path, 'punishment_range_gpt.json')
                        with open(output_path, 'w') as output_file:
                            json.dump(all_punishment_ranges, output_file, ensure_ascii=False)
                        print(f"Punishment ranges saved to {output_path}")

extract_punishment_range_tagging_db('/home/tak/MOJ/results/db/drugs')
