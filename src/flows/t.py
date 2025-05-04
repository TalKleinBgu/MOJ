path = '/home/tak/MOJ/results/db/drugs_docx'
count = 0
count_2 =0
import os
for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            if file.endswith('.json'):
                count += 1
        count_2 +=1
print(count)
print(count_2)