class Dicta2Prompts():
    """
    A class for extracting specific pieces of information from legal/case text 
    regarding drug-related details, punishments, and general circumstances using 
    a given model and tokenizer. Methods are structured to ask particular questions 
    about the text and parse the model's answers accordingly.
    """

    def __init__(self, model, tokenizer) -> None:
        """
        Initializes the Dicta2Prompts class with a model and tokenizer.

        Args:
            model: A language model (e.g., from the transformers library) used for generating answers.
            tokenizer: A tokenizer paired with the model to encode and decode text.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cuda'  # Assuming GPU usage for speed

        # self.feature_dict = feature_dict
        # (feature_dict is commented out here but can be used to store extracted features.)

    def dicta_response_parser(self, answer, message):
        """
        Parses the model's raw text output for a Dicta-based model.

        Args:
            answer (str): The raw text output returned by the model.
            message (list[dict]): The prompt message used to generate the answer.

        Returns:
            str: A cleaned string answer without prompt text or extra tokens.
        """
        return (
            answer
            .replace(message[0]['content'], '')
            .split('\n')[-1]
            .replace("</s>", '')
            .rstrip('.')
        )
    
    def c4ai_response_parser(self, answer, message):
        """
        Parses the model's raw text output for a c4ai-based model, removing 
        specific tokens often added by the model.

        Args:
            answer (str): The raw text output returned by the model.
            message (list[dict]): The prompt message used to generate the answer.

        Returns:
            str: A cleaned string answer.
        """
        return (
            answer
            .split('|CHATBOT_TOKEN|')[0]
            .replace('<|END_OF_TURN_TOKEN|>', '')
            .replace('<', '')
            .replace('>', '')
        )
    
    def generate_answer(self, message):
        """
        Tokenizes the prompt, generates a response using the model, and decodes 
        the resulting IDs back into text. Depending on the model's name, it calls 
        the appropriate parser function.

        Args:
            message (list[dict]): A list with a single dictionary representing 
                                  the user message, typically containing a 'role' 
                                  and 'content' key.

        Returns:
            str: The final, cleaned answer from the model.
        """
        # Encode the prompt message using the tokenizer's special chat template
        encoded = self.tokenizer.apply_chat_template(message, return_tensors="pt").to(self.device)
        # Generate model output with a controlled maximum new tokens
        generated_ids = self.model.generate(encoded, max_new_tokens=80, pad_token_id=50256)
        # Decode the output token IDs into text
        decoded = self.tokenizer.batch_decode(generated_ids)
        
        # Choose a parser based on the model's name/path
        if 'c4ai' in self.model.name_or_path:
            return self.c4ai_response_parser(decoded[0], message)
        else:
            return self.dicta_response_parser(decoded[0], message)
    
    def ask2extract_CIR_TYPE(self, text):
        """
        Asks what type of drug is described in the text.

        Args:
            text (str): The text under analysis.

        Returns:
            str: The extracted drug type as a single word.
        """
        message = [
            {
                "role": "user",
                "content": f"""
                    אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
                    הטקסט לבדיקה: "{text}"

                    השאלה:
                    מהו סוג הסמים ? ענה רק את הסוג.
                """
            }
        ]
        answer = self.generate_answer(message)
        return answer
    
    def ask2extract_CIR_AMOUNT(self, text):
        """
        Asks for the amount of drugs mentioned in the text, expecting a numeric answer.

        Args:
            text (str): The text under analysis.

        Returns:
            str: The drug amount, typically a numeric value.
        """
        message = [
            {
                "role": "user",
                "content": f"""
                    אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
                    הטקסט לבדיקה: "{text}"

                    השאלה:
                    מהי הכמות של הסמים ?
                    ענה בפורמט הבא - [משקל(במספר) - יחידות(כמו גרם, טבליות, כדורים, ק"ג וכו')] .
                """
            }
        ]
        answer = self.generate_answer(message)
        return answer
    
    def ask2extract_CIR_ROLE(self, text):
        """
        Checks whether the defendant had ownership or involvement with the drugs, 
        providing limited answer options.

        Args:
            text (str): The text describing the defendant's role.

        Returns:
            list: A list with a single extracted role (e.g., "בעל המעבדה" or "לא בעל הסמים").
        """
        message = [
            {
                "role": "user",
                "content": f"""
                    "{text}"
                    האם הנאשם היה בעל הסמים? האם הנאשם השתמש במעבדה? 
                    ענה מהתשובות הבאות בלי מילה נוספת 
                    [בעל המעבדה, לא בעל המעבדה, בעל הסמים, לא בעל הסמים], !
                """
            }
        ]
        answer = self.generate_answer(message) 
        return [answer]
        
    def ask2extract_CIR_EQ(self, text):
        """
        Asks if the defendant used a drug lab, expecting a yes/no answer.

        Args:
            text (str): The text under analysis.

        Returns:
            str: "כן" if the defendant used the lab, "לא" otherwise.
        """
        message = [
            {
                "role": "user",
                "content": f"""
                    אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
                    "{text}"    
                    האם הנאשם השתמש במעבדה לסמים ? ענה רק [כן, לא] בלבד.            
                """
            }
        ]
        answer = self.generate_answer(message)
        return answer
    
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
