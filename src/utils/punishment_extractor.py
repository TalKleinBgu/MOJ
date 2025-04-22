import pandas as pd
import re

pattern = "מתח. ה?עו?ני?שה?"


def extract_two_numbers(string):
    words = string.split()  # Split the string into words
    numbers = []
    for i in range(len(words)):
        if words[i][0] =='ל':
            #check if ther is words[i][1]
            if len(words[i]) > 1:
                #check if words[i][1] is digit
                n = words[i][1]
                if n.isdigit():
                    words[i] = n
        if len(words[i]) > 1 and '-' in words[i]:
            numbers_ = words[i].split('-')
            # delete '' from the list
            numbers_ = list(filter(lambda x: x != '', numbers_))
            #check if there is number
            numbers_ = list(filter(lambda x: x.isdigit(), numbers_))
            if len(numbers_) == 2:
                if words[i+1] == 'שנות':
                    numbers.append(int(numbers_[0])*12)
                    numbers.append(int(numbers_[1])*12)
                else:
                    numbers.append(int(numbers_[0]))
                    numbers.append(int(numbers_[1]))
                continue
            elif len(numbers_) == 0:
                pass
            else:
                if i+1 < len(words):
                    if words[i+1] =='שנות':
                        if numbers != []:
                            numbers[0] = int(numbers[0])*12
                            numbers.append(int(numbers_[0])*12)
                        else:
                            numbers.append(int(numbers_[0])*12)
                # check if there is word[i+2]
                elif i+2 < len(words):
                    if words[i+2] == 'שנות' or words[i+2] == 'שנים':
                        if numbers != []:
                            numbers[0] = int(numbers[0])*12
                            numbers.append(int(numbers_[0])*12)
                        else:
                            numbers.append(int(numbers_[0])*12)
                else:
                    numbers.append(int(numbers_[0]))
        if words[i].isdigit():
            number = int(words[i])
            try:
                if words[i-1] == 'בן' or 'סעיף' in words[i-1] or words[i-1] == 'תיקון' or words[i+1] == 'מהצדדים':
                    continue
            except:
                number = int(words[i])
            
            next_words = " ".join(words[i+1:i+3])  # Get the next two words after the number
            if next_words.split()[0].isdigit():
                number = float(f'{number}.{next_words.split()[0]}')
                next_words = " ".join(words[i+2:i+4])
                #dekete the next word
                words[i+1] = ''
            if 'שנות מאסר' in next_words or 'שנים' in next_words or 'שנה' in next_words:
                number *= 12
            
            else:
                cleaned_words = [word.strip() for word in words if word.strip()]

                next_words_2 = " ".join(cleaned_words[i+1:i+4])
                next_word = next_words_2.split()[0]
                # print(type(next_word))
                if next_word.strip() in ['עד', 'בין', '-'] or not next_word.strip().isalpha():
                    if len(next_words_2.split()) > 1 and next_words_2.split()[1] == 'שנות':
                        number *= 12
                    elif len(next_words_2.split()) > 2 and next_words_2.split()[2] == 'שנות':
                        number *= 12
                elif 'חודשי מאסר' not in next_words:
                    # If none of the options are found, it's multiplied by 1
                    pass
                
            numbers.append(int(number))
    if len(numbers) == 2:
        # Ensure that numbers[0] represents the lower number
        if numbers[0] > numbers[1]:
            numbers[0], numbers[1] = numbers[1], numbers[0]
        return {"lower": str(numbers[0]), "top": str(numbers[1])}
    elif len(numbers) > 2:
        dict_numbers = {}
        #check if numbers list is had even length
        length = len(numbers)
        if length % 2 != 0:
            numbers = numbers[:-1]
        for i in range(0,len(numbers), 2):
            dict_numbers[f'numbers_{i}'] = f'{numbers[i]} - {numbers[i+1]}'
        return dict_numbers

    else:
        return None

def get_indexs(pattern, str):
    """
    Input:  patern: patren to compile in regex format.
            str: string to find in doc.
    Output: list of tupls - (start_str,end_str).
    """
    pattern = re.compile(pattern)
    try:
        return [(m.start(0), m.end(0)) for m in re.finditer(pattern, str)]
    except:
        print()


def punishment_range_extract(txt):
    # TODO improve regex
    # Role 1: if have 2 number in the sentence so its the range
    punishment_range = extract_two_numbers(txt)
    if punishment_range:
        return punishment_range
    else:
        return None
    # Role 2: else
    indexes = get_indexs(pattern, txt)

    boundry_punishment = []
    between_L_flag = False
    for ind, index in enumerate(indexes):

        candidate_sentence = txt[index[0]:].split(".")[0]
        # if in candidat sentence have more than 1 mitham' split that and check the first sen that contain mitham, for avoid duplication.
        check_duplication = get_indexs(pattern, candidate_sentence)
        if len(check_duplication) > 1:
            try:

                candidate_sentence = candidate_sentence.split(pattern)[1]
            except:
                # TODO check here with breakpoint
                candidate_sentence = candidate_sentence

        # print(candidat_sen)

        punishment_range_indexes = get_indexs('בין', candidate_sentence)
        # print(mi tham_indexs)

        for punishment_range_index in punishment_range_indexes:

            candidate_sentence = txt[punishment_range_index[1] + index[0]:].split(".")[0].split(';')[0]
            # print(candidat_sen_mitham)
            # remove double ' '
            candidate_sentence_ = re.sub(' +', ' ', candidate_sentence)
            index_month = get_indexs("חודשי", candidate_sentence_)
            tokens = candidate_sentence_.split(" ")
            # print(words)
            for i, token in enumerate(tokens):
                if i != 0:

                    if 'ל-' in token: between_L_flag = True
                    if (i == 1) or (tokens[i - 1] == '–') or ('–' in token) or ('ל-' in token) or between_L_flag:
                        try:
                            # print(word)
                            token = token.replace('ל-', '')
                            token = token.replace('-', ' ').split(" ")

                            if len(token) > 1:
                                for word_ in token:
                                    boundry_punishment.append(str(int(word_)))
                            else:
                                boundry_punishment.append(str(int(token[0])))

                        except:
                            continue

                between_L_flag = False
            if boundry_punishment != 0:
                if len(index_month) > 0:
                    boundry_punishment.append("חודשי מאסר")
                else:
                    boundry_punishment.append("שנות מאסר")
    return str(boundry_punishment)

def case_mapping_dict(mapping_path):
    # Load mapping CSV file
    mapping_df = pd.read_csv(mapping_path, encoding='utf-8')
    # Assuming the mapping CSV has columns 'directory' and 'name'
    directory_to_verdict_map = dict(zip(mapping_df['directory'], mapping_df['name']))
    return directory_to_verdict_map

def verdict_punishmet_dict(extracted_features_path, mapping_path):
    """
    Create a dictionary mapping verdict files to their corresponding  verdict number and punishment.

    Parameters:
    extracted_features_path (str): Path to the CSV file containing tagged features.
    mapping_path (str): Path to the CSV file containing the mapping between verdict's file name and verdict number.

    Returns:
    dict: A dictionary mapping verdict file names to tuples of verdict number and punishment.
    example: {Verdict File Name: (Verdict num, punishment)}
    example2:{ME-1234-56-78: (56-78-12, 4 - 5 שנים )}
    """

    directory_to_verdict_map = case_mapping_dict(mapping_path)
    # Load tagged features csv
    tagged_features = pd.read_csv(extracted_features_path, encoding='utf-8')
    # print(tagged_features.columns)
    for verdict_file, verdict_num in directory_to_verdict_map.items():
        verdict_row = tagged_features[tagged_features['מספר תיק '] == verdict_num]
        if not verdict_row.empty:
            punishment = verdict_row['מתחם ענישה - שופט '].iloc[0]
            directory_to_verdict_map[verdict_file] = (verdict_num, punishment)
        else:
            directory_to_verdict_map[verdict_file] = (verdict_num, None)

    return directory_to_verdict_map


def check_numbers_match(string1, string2):  # TODO fix it, doesnt work well!
    """
         checks if 2 numbers are the same in both string, naive way to compare if the extraction is good

         Returns:
         Boolean
         """
    if not string1 or not string2:
        return None

    # Function to extract numbers from a string
    def extract_numbers(string):
        import re
        # Regular expression pattern to match digits
        pattern = r'\d+'
        # Extract numbers from the string
        numbers = re.findall(pattern, string)
        return numbers

    # Extract numbers from both strings
    numbers1 = extract_numbers(string1)
    numbers2 = extract_numbers(string2)

    # Check if both lists contain at least two numbers
    if len(numbers1) == 2 and len(numbers2) == 2:
        # Check if both numbers from string1 match any permutation of the numbers from string2
        if set(numbers1) == set(numbers2):
            return True
        # If the numbers are not the same
        return False
    else:
        # If either string doesn't contain exactly two numbers
        return False


# def replace_words_with_numbers(input_string,
#                                number_dict):  # TODO fix it, doesnt work so well, Think about the right split!
#     """
#        replacing words with thier corresponding number, with the dict at the top of the file here
#         """
#     words = re.split(r'\s|,|\.|ל', input_string)
#     # words = input_string.split()
#     for i, word in enumerate(words):
#         if word in number_dict:
#             words[i] = str(number_dict[word])
#     return ' '.join(words)
def replace_words_with_numbers(input_string, number_dict):
    """
    Replace words representing numbers (including floats) with their corresponding numeric values using a dictionary.

    Args:
        input_string (str): The input string containing words to be replaced.
        number_dict (dict): A dictionary mapping number words to their numeric values.

    Returns:
        str: A string with words replaced by their numeric values.
    """
    # Pattern to match words or phrases representing numbers, including "point" for floats.
    pattern = r'(?:' + '|'.join(map(re.escape, number_dict.keys())) + r')(?:\s+point\s+(?:' + '|'.join(map(re.escape, number_dict.keys())) + '))?'

    def replace_match(match):
        """Replace matched number words with their corresponding numeric value."""
        parts = match.group(0).split()
        if 'point' in parts:
            # Handle floats (e.g., "ten point five").
            integer_part = number_dict.get(parts[0], parts[0])
            decimal_part = number_dict.get(parts[2], parts[2])
            return str(float(f"{integer_part}.{decimal_part}"))
        else:
            # Handle whole numbers.
            return str(number_dict.get(match.group(0), match.group(0)))

    # Use regex to find and replace matches in the input string.
    return re.sub(pattern, replace_match, input_string)
