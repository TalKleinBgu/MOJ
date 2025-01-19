class Dicta2Prompts:
    """
    This class serves as a prompt interface for extracting information from text 
    using a given model and tokenizer. It provides a series of 'ask2extract' methods
    for specific pieces of information (e.g., confession, weapon status) by constructing
    prompt messages and parsing the model's output.
    """

    def __init__(self, model, tokenizer):
        """
        Initializes the Dicta2Prompts class with the provided model, tokenizer, 
        and a dictionary (feature_dict) to store features.

        Args:
            model: A language model (e.g., from the transformers library) used for generating answers.
            tokenizer: A tokenizer paired with the model to encode/decode text.
            feature_dict (dict): A dictionary where extracted features may be stored or referenced.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cuda'        # Assuming a GPU is available
        # self.feature_dict = feature_dict

    def dicta_response_parser(self, answer, message):
        """
        Parses the model's response when using Dicta-based models.

        Args:
            answer (str): The raw text output from the model.
            message (list[dict]): The prompt message that was passed to the model.

        Returns:
            str: A cleaned answer string without prompt text or extra tokens.
        """
        return (answer
                .replace(message[0]['content'], '')
                .split('\n')[-1]
                .replace("</s>", '')
                .replace(".", ""))

    def c4ai_response_parser(self, answer, message):
        """
        Parses the model's response when using a c4ai-based model. 
        The parser removes model-inserted tokens.

        Args:
            answer (str): The raw text output from the model.
            message (list[dict]): The original prompt message.

        Returns:
            str: A cleaned answer string without special tokens.
        """
        return (answer
                .split('|CHATBOT_TOKEN|')[0]
                .replace('<|END_OF_TURN_TOKEN|>', '')
                .replace('<', '')
                .replace('>', ''))

    def generate_answer(self, message):
        """
        Encodes the prompt message with the tokenizer, generates a response using the model,
        and decodes the response. Then, calls the appropriate parser method based on the 
        model's name_or_path.

        Args:
            message (list[dict]): A list of dictionaries with role/content keys (chat-like format).

        Returns:
            str: A parsed answer from the model.
        """
        # Encode the message using the tokenizer's special chat template method
        encoded = self.tokenizer.apply_chat_template(message, return_tensors="pt").to(self.device)
        # Generate response IDs from the model
        generated_ids = self.model.generate(encoded, max_new_tokens=50, pad_token_id=50256)
        # Decode the token IDs back to text
        decoded = self.tokenizer.batch_decode(generated_ids)

        # Determine which parser to use based on the model's name
        if 'c4ai' in self.model.name_or_path:
            return self.c4ai_response_parser(decoded[0], message)
        else:
            return self.dicta_response_parser(decoded[0], message)

    def ask2extract_CONFESSION(self, text):
        """
        Constructs a prompt that asks if the defendant confessed. 
        Expects an answer of "כן" (yes) or "לא" (no).

        Args:
            text (str): The text from which we want to extract a confession status.

        Returns:
            str: Either "כן" or "לא" as determined by the model.
        """
        message = [
            {
                "role": "user", 
                "content": f"""
                    "{text}"
                    האם הנאשם הודה באשמה? ענה רק [כן, לא] בלבד
                """
            }
        ]
        answer = self.generate_answer(message)
        return answer

    def ask2extract_CIR_TYPE_WEP(self, text):
        """
        Asks what type of weapon is described, restricted to one of a predefined list. 
        Expects an answer such as "אקדח", "תת מקלע", etc.

        Args:
            text (str): The text from which the weapon type is to be extracted.

        Returns:
            list: A list with a single element containing the extracted weapon type.
        """
        message = [
            {
                "role": "user",
                "content": f"""
                    "{text}"
                    מהו סוג הנשק? ענה מהתשובות הבאות בלי מילה נוספת 
                    [אקדח, תת מקלע, תת מקלע מאולתר, בקבוק תבערה, מטען חבלה,
                     רימון רסס, רובה סער, רימון הלם/גז, טיל לאו, טיל מטאדור,
                     רובה צייד, רובה צלפים, מטען חבלה מאולתר, רובה סער מאולתר], !
                """
            }
        ]
        answer = self.generate_answer(message)
        return [answer]

    def ask2extract_CIR_OBTAIN_WAY_WEP(self, text):
        """
        Checks a list of possible ways the defendant might have obtained the weapon: 
        'קיבל', 'מצא', 'גנב', 'ייצר'. For each option, it asks the model if the defendant 
        obtained the weapon in that way, returning a list of all that apply.

        Args:
            text (str): The text to be analyzed.

        Returns:
            list: A list of strings indicating how the weapon was obtained.
        """
        options = ['קיבל', 'מצא', 'גנב', 'ייצר']
        answers = []
        for option in options:
            message = [
                {
                    "role": "user", 
                    "content": f"""
                        אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
                        "{text}"
                        האם הנאשם {option} את כלי הנשק? ענה רק [כן, לא]
                    """
                }
            ]
            answer = self.generate_answer(message)
            if answer == 'כן':
                if option == 'קיבל':
                    # Normalize the response 'קיבל' to 'מאחר'
                    option = 'מאחר'
                answers.append(option)
        return answers

    def ask2extract_CIR_MONEY_PAID(self, text):
        """
        Asks how much money was paid (if any). The model is expected to 
        provide a numeric or textual answer describing the sum.

        Args:
            text (str): The text to be analyzed.

        Returns:
            str: The sum of money paid, as a string.
        """
        message = [
            {
                "role": "user", 
                "content": f"""
                    "{text}"
                    מהו סכום הכסף ששולם?
                """
            }
        ]
        answer = self.generate_answer(message)
        return answer

    def ask2extract_CIR_STATUS_WEP(self, text):
        """
        Determines the status of the weapon from a list of possible statuses.
        If the model responds 'כן', we add that status to the answers list.

        Args:
            text (str): The text being analyzed for weapon status.

        Returns:
            list: A list of statuses found in the text (e.g., "נשק מפורק").
        """
        options = ['מפורק', 'מופרד מתחמושת', 'עם מחסנית בהכנס',  'עם כדור בקנה', 'תקול']
        answers = []
        for option in options:
            message = [
                {
                    "role": "user", 
                    "content": f'''
                    1 דוגמה:
                    אני הולך לשאול אותך שאלה על פי הטקסט הבא:
                    מהאישום השלישי עולה כי לערך יום לאחר ביצוע הירי לעבר המתלונן, 
                    בעת שהנאשם עבד כשומר באתר בניה, הוא החזיק נשק נוסף, מחסנית ותחמושת 
                    בזחל של אחד מכלי העבודה באתר, וכן כ- 17 כדורים מתחת לתקרה אקוסטית במכולה שעמדה במקום.
                    האם הנשק מופקד מתחמושת? ענה [כן,לא] בלבד

                    תשובה:
                    כן
                    2 דוגמה:
                    אני הולך לשאול אותך שאלה על פי הטקסט הבא:
                    על-פי האמור בכתב האישום, ביום 14.10.07 נגנב אקדח מסוג "יריחו" מבעליו, 
                    וביום 18.11.16 החזיק הנאשם באקדח הגנוב, בכיס מעילו, כשהוא דרוך, טעון ומוכן לירי.
                    האם הנשק עם כדור בקנה ? ענה [כן,לא] בלבד
                     תשובה:
                     כן 
                     
                     אני הולך לשאול אותך שאלה על פי הטקסט הבא:
                     !הבהרה - אם הנשק טעון הכוונה טעון במחסנית בהכנס, ואם הנשק דרוך אז הכוונה שהוא עם כדור בקנה,
                     "{text}"
                    האם הנשק {option}? ענה [כן,לא] בלבד
                    '''
                }
            ]
            answer = self.generate_answer(message)
            if answer == 'כן':
                # Normalize the status text
                if option != 'תקול':
                    answers.append("נשק " + option)
                else:
                    answers.append(option)
        return answers

    def ask2extract_CIR_HELD_WAY_WEP(self, text):
        """
        Checks a list of possible locations/ways the weapon might have been held:
        'בבית', 'ברכב', 'על גופו(נושא אותו)', 'מוסלק', 'סמוך לבית'.
        For each location, if the model says 'כן', it is added to the results list.

        Args:
            text (str): The text being analyzed.

        Returns:
            list: Locations where the weapon was held.
        """
        options = ['בבית', 'ברכב', 'על גופו(נושא אותו)', 'מוסלק', 'סמוך לבית']
        answers = []
        for option in options:
            message = [
                {
                    "role": "user", 
                    "content": f"""
                        אני הולך לשאול אותך שאלה על פי הטקסט הבא:
                        "{text}"
                        האם הנשק נמצא ב{option}? ענה [כן,לא]
                    """
                }
            ]
            answer = self.generate_answer(message)
            if answer == 'כן':
                # Normalize 'על גופו(נושא אותו)' to 'על גופו'
                if option == 'מוסלק':
                    option += ' - מוסתר'
                if option == 'על גופו(נושא אותו)':
                    option = option.replace('(נושא אותו)', '')
                answers.append(option)
        return answers

    def ask2extract_CIR_PURPOSE(self, text):
        """
        Checks if the text explicitly mentions certain purposes for using a weapon:
        'תדמית', 'חתונה', 'סכסוך', 'הגנה עצמית', 'בצע כסף'.

        Args:
            text (str): The text being analyzed.

        Returns:
            list: A list of recognized purposes (if any).
        """
        options = ['תדמית', 'חתונה', 'סכסוך', 'הגנה עצמית', 'בצע כסף']
        answers = []
        for option in options:
            message = [
                {
                    "role": "user",
                    "content": f"""
                        "{text}"
                        האם מטרת השימוש בנשק היא {option}? ענה ב [כן, לא] בלבד
                    """
                }
            ]
            answer = self.generate_answer(message)
            if answer == 'כן':
                answers.append(option)
        return answers

    def ask2extract_CIR_USE(self, text):
        """
        Asks if the defendant performed specific actions with the weapon: 
        'ירי', 'ניסיון לירי', 'זריקת רימון', 'הפעלת מטען'.

        Args:
            text (str): The text to be analyzed.

        Returns:
            list: A list of actions the defendant performed (e.g., ['ירי']).
        """
        answers = []
        options = ['ירי', 'ניסיון לירי', 'זריקת רימון', 'הפעלת מטען']
        for option in options:
            message = [
                {
                    "role": "user", 
                    "content": f"""
                        אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
                        "{text}"
                        האם הנאשם ביצע {option}? ענה ב [כן, לא] בלבד
                    """
                }
            ]
            answer = self.generate_answer(message)
            # Handle multi-line responses
            if len(answer.split(' ')) > 1:
                answer = answer.split('\n')[-1]
            if answer == 'כן':
                answers.append(option)
        return answers

    def generate_message(self, text, question):
        """
        Creates a standardized chat-like message from the given text and question.

        Args:
            text (str): The relevant text snippet.
            question (str): The question to be posed about that text.

        Returns:
            list: A list containing one dictionary that acts as the user's message.
        """
        message = [
            {
                "role": "user",
                "content": f"""
                    אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
                    הטקסט:
                    "{text}"

                    השאלה:
                        {question}
                """
            }
        ]
        return message
    
    def ask2extract_PUNISHMENT_ACTUAL(self, text):
        """
        Checks whether the defendant received a non-suspended (actual) prison sentence.
        If yes, a follow-up question asks for the duration.

        Args:
            text (str): The text describing the sentencing details.

        Returns:
            str: 'לא' if no actual prison time, or a brief answer specifying duration.
        """
        # Prompt asking if there was an actual prison sentence
        message = [
            {
                "role": "user",
                "content": f"""
                    אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
                    הטקסט לבדיקה: "{text}"

                    השאלה:
                    האם הנאשם קיבל מאסר שהוא לא על תנאי? . ענה ב [כן, לא] בלבד.
                """
            }
        ]
        answer = self.generate_answer(message)

        if answer == 'לא' or answer.startswith('לא'):
            return None
        else:
            # If yes, follow-up to get the duration of the non-suspended sentence
            message = self.generate_message(
                text,
                'מהו המאסר שהוא לא על תנאי שקיבל הנאשם? ענה בקצרה רק את תקופת המאסר בלי הסברים.'
            )
            answer = self.generate_answer(message)
            return answer
        
    def ask2extract_PUNISHMENT_SUSPENDED(self, text):
        """
        Checks if the defendant received a suspended prison sentence. 
        If yes, asks for the duration of the suspension.

        Args:
            text (str): The text describing the sentencing details.

        Returns:
            str: 'לא' if no suspended sentence, or a brief duration if there is one.
        """
        # Prompt asking if there was a suspended sentence
        message = [
            {
                "role": "user",
                "content": f"""
                    אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
                    הטקסט לבדיקה: "{text}"

                    השאלה:
                    האם הנאשם קיבל מאסר על תנאי? . ענה ב [כן, לא] בלבד.
                """
            }
        ]
        answer = self.generate_answer(message)

        if answer == 'לא' or answer.startswith('לא'):
            return None
        else:
            # If yes, follow-up to get the suspended sentence duration
            message = self.generate_message(
                text,
                'מהו מספר חודשי המאסר על תנאי שקיבל הנאשם? ענה בקצרה רק את תקופת המאסר בלי הסברים.'
            )
            answer = self.generate_answer(message)
            return answer

    def ask2extract_PUNISHMENT_FINE(self, text):
        """
        Checks whether the text explicitly mentions a monetary fine. 
        If yes, asks for the amount of the fine.

        Args:
            text (str): The text describing the sentencing details.

        Returns:
            str: 'לא' if no fine is mentioned, or the amount of the fine if it is.
        """
        # Prompt asking if a fine was mentioned
        message = [
            {
                "role": "user",
                "content": f"""
                    אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
                    הטקסט לבדיקה: "{text}"

                    השאלה:
                    בדוק האם בטקסט הבא רשום במפורש את גודל הקנס שהשופט גזר לנאשם. 
                    ענה ב [כן, לא] בלבד.
                """
            }
        ]
        answer = self.generate_answer(message)

        if answer == 'לא' or answer.startswith('לא'):
            return None
        else:
            # If yes, follow-up to get the fine amount
            message = self.generate_message(
                text,
                'מהו גודל הקנס שקיבל הנאשם? ענה בקצרה רק את גודל בלי הסברים.'
            )
            answer = self.generate_answer(message)
            return answer

    def ask2extract_GENERAL_CIRCUM(self, text):
        """
        Asks a series of yes/no questions about general mitigating or circumstantial factors.
        Each question is stored in 'messeges', and for each one, the model's answer is appended 
        to the result list.

        Args:
            text (str): The full text describing the defendant's context or circumstances.

        Returns:
            list: Each element corresponds to a question's index and the model's yes/no response.
        """
        answers = []
        messeges = [
            "האם הטקסט מתייחס להשפעת העונש על הנאשם, במיוחד בהקשר של גילו.",
            "האם הטקסט מתייחס לפגיעה האפשרית של העונש על בני משפחתו של הנאשם.",
            "האם הטקסט מתייחס לנזקים שנגרמו לנאשם מביצוע העבירה או מהרשעתו.",
            "האם הטקסט מתייחס לכך שהנאשם לקח אחריות על מעשיו, חזר למוטב, או עשה מאמצים לחזור למוטב.",
            "האם הטקסט מתייחס למאמצים של הנאשם לתקן את תוצאות העבירה או לפצות על הנזק שנגרם.",
            "האם הטקסט מתייחס לשיתוף פעולה של הנאשם עם רשויות אכיפת החוק, או לכך שהנאשם כפר באשמה וניהל משפט.",
            "האם הטקסט מתייחס להתנהגות חיובית של הנאשם או לתרומתו לחברה.",
            "האם הטקסט מתייחס לנסיבות חיים קשות של הנאשם שהשפיעו על ביצוע העבירה.",
            "האם הטקסט מתייחס להתנהגות רשויות אכיפת החוק בהקשר של הנאשם או המקרה.",
            "האם הטקסט מתייחס לחלוף הזמן מאז ביצוע העבירה.",
            "האם הטקסט מתייחס לעבר הפלילי של הנאשם או להיעדרו."
        ]
        yes_no = " ענה ב [כן, לא] בלבד."
        i = 1  # Used for labeling question answers

        # For each question, create a prompt and record the response
        for question in messeges:
            question_ = question + yes_no
            message = self.generate_message(text, question_)
            answer = self.generate_answer(message)
            answer = str(i) + " : " + answer  # Attach question index to the answer
            answers.append(answer)
            i += 1
        return answers
