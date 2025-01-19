import re
import anthropic


class ClaudeSonnet():
    def __init__(self, api_key, feature_dict=None, temperature= 0.5) -> None:
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.feature_dict = feature_dict
        self.temperature = temperature
    
    def generate_answer(self, message):
                
        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            top_k=3,
            top_p=0.9,
            temperature=self.temperature,
            messages=[{"role": "user", "content": [
                            {
                                'type': 'text',
                                'text': f"{message}"
                            }
                        ]
                        }
                    ]
            )
        generated_text = message.content[0].text
        return generated_text

    def ask2extract_CONFESSION(self, text):
        message = [
        {"role": "user", 
         "content": f"""
        "{text}"
        האם הנאשם הודה באשמה? ענה רק [כן, לא] בלבד
        """}] 
        answer = self.generate_answer(message) 
        return answer
    
    def ask2extract_CIR_TYPE_WEP(self, text):
        message = [
        {"role": "user",
        "content": f"""
        "{text}"
        מהו סוג הנשק? ענה מהתשובות הבאות בלי מילה נוספת [אקדח, תת מקלע, תת מקלע מאולתר, בקבוק תבערה, מטען חבלה, רימון רסס, רובה סער, רימון הלם/גז, טיל לאו, טיל מטאדור, רובה צייד, רובה צלפים, מטען חבלה מאולתר, רובה סער מאולתר], !"""}]
        answer = self.generate_answer(message) 
        return [answer]
        
    def ask2extract_CIR_OBTAIN_WAY_WEP(self, text):
        options = ['קיבל', 'מצא', 'גנב', 'ייצר']
        answers = []
        for option in options:
            message = [
            {"role": "user", 
            "content": f"""
            אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
            "{text}"    
            האם הנאשם {option} את כלי הנשק? ענה רק [כן, לא]
            
            """}] 
            answer = self.generate_answer(message)
            if answer == 'כן':
                if option == 'קיבל':
                    option = 'מאחר'
                answers.append(option) 
        return answers
    
    def ask2extract_CIR_MONEY_PAID(self, text):
        message = [
        {"role": "user", 
         "content": f"""
        "{text}"
        מהו סכום הכסף ששולם?
        """}] 
        answer = self.generate_answer(message) 
        return answer

    def ask2extract_CIR_STATUS_WEP(self, text):
        options = ['מפורק', 'מופרד מתחמושת', 'עם מחסנית בהכנס',  'עם כדור בקנה', 'תקול']
        answers = []
        for option in options:
            message = [
            {"role": "user", 
            "content": f"""
            1 דוגמה:
            אני הולך לשאול אותך שאלה על פי הטקסט הבא:
            מהאישום השלישי עולה כי לערך יום לאחר ביצוע הירי לעבר המתלונן, בעת שהנאשם עבד כשומר באתר בניה, הוא החזיק נשק נוסף, מחסנית ותחמושת בזחל של אחד מכלי העבודה באתר, וכן כ- 17 כדורים מתחת לתקרה אקוסטית במכולה שעמדה במקום.
            האם הנשק מופקד מתחמושת? ענה [כן,לא] בלבד

            תשובה:
            כן
            2 דוגמה:
            אני הולך לשאול אותך שאלה על פי הטקסט הבא:
            על-פי האמור בכתב האישום, ביום 14.10.07 נגנב אקדח מסוג "יריחו" מבעליו, וביום 18.11.16 החזיק הנאשם באקדח הגנוב, בכיס מעילו, כשהוא דרוך, טעון ומוכן לירי.
            האם הנשק עם כדור בקנה ? ענה [כן,לא] בלבד
             תשובה:
             כן 
             
             אני הולך לשאול אותך שאלה על פי הטקסט הבא:
             !הבהרה - אם הנשק טעון הכוונה טעון במחסנית בהכנס, ואם הנשק דרוך אז הכוונה שהוא עם כדור בקנה,
             "{text}"
            האם הנשק {option}? ענה [כן,לא] בלבד
            """}] 
            answer = self.generate_answer(message) 
            if answer == 'כן':
                if options != 'תקול':
                    answers.append("נשק "  + option)
                else:
                    answers.append(option)
        return answers

    def ask2extract_CIR_HELD_WAY_WEP(self, text):
        options = ['בבית', 'ברכב', 'על גופו(נושא אותו)', 'מוסלק', 'סמוך לבית']
        answers = []
        for option in options:
            message = [
            {"role": "user", 
            "content": f"""
            אני הולך לשאול אותך שאלה על פי הטקסט הבא:
            "{text}"
            האם הנשק נמצא ב{option}? ענה [כן,לא]
            """}] 
            answer = self.generate_answer(message) 
            if answer == 'כן':
                if option =='מוסלק':
                    option = option + ' - מוסתר'
                if option == 'על גופו(נושא אותו)':
                    option = option.replace('(נושא אותו)', '')
                answers.append(option)
        return answers
    
    def ask2extract_CIR_PURPOSE(self, text):
        options = ['תדמית', 'חתונה', 'סכסוך', 'הגנה עצמית', 'בצע כסף']
        answers = []
        for option in options:
            message = [
            {"role": "user", 
            "content": f"""
            "{text}"
            האם מטרת השימוש בנשק היא {option}? ענה ב [כן, לא] בלבד
            """}] 
            answer = self.generate_answer(message) 
            if answer == 'כן':
                answers.append(option)
        return answers

    def ask2extract_CIR_USE(self, text):
        
        answers = []
        options = ['אזהרה', 'ירי', 'ניסיון לירי', 'זריקת רימון', 'הפעלת מטען']
        for option in options:
            message = [
            {"role": "user", 
            "content": f"""
            אני הולך לשאול אותך שאלה על פי הטקסט הבא, ואז אתה תענה.
            "{text}"
            האם הנאשם ביצע {option}? ענה ב [כן, לא] בלבד
            """}] 
            
            answer = self.generate_answer(message) 
            if answer == 'כן':
                answers.append(option)
        return answers
    