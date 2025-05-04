

class DicatePrompts():
    def __init__(self, model, feature_dict) -> None:
        self.model = model
        self.feature_dict = feature_dict
        
    def ask2extract_CIR_AMMU_AMOUNT_WEP(self, text):
        context = text
        question = "כמה תחמושת?"
        prediction = self.model(question=question, context=context)
        self.feature_dict["AMMO_AMOUNT"].append(prediction['answer'])
        if prediction['score'] > 0.5:
            return prediction['answer']

    def ask2extract_CIR_HELD_WAY_WEP(self, text):
        context = text
        question = "איפה הנשק?"
        prediction = self.model(question=question, context=context)
        self.feature_dict["HELD_WAY"].append(prediction['answer'])
        if prediction['score'] > 0.5:
            return prediction['answer']

    def ask2extract_CIR_OBTAIN_WAY_WEP(self, text):
        context = text
        question = "איך השיג את הנשק?"
        prediction = self.model(question=question, context=context)
        self.feature_dict["OBTAIN_WAY"].append(prediction['answer'])
        if prediction['score'] > 0.5:
            return prediction['answer']

    def ask2extract_CIR_STATUS_WEP(self, text):
        context = text
        question = "האם הנשק טעון או דרוך או מפורק או עם כדור בקנה?"
        prediction = self.model(question=question, context=context)
        self.feature_dict["STATUS_WEP"].append(prediction['answer'])
        if prediction['score'] > 0.5:
            return prediction['answer']

    def ask2extract_CIR_TYPE_WEP(self, text):
        context = text
        question = "איזה נשק?"
        prediction = self.model(question=question, context=context)
        self.feature_dict["TYPE_WEP"].append(prediction['answer'])
        if prediction['score'] > 0.5:
            return prediction['answer']

    def ask2extract_CIR_USE(self, text):
        context = text
        question = "איזה שימוש בנשק?"
        prediction = self.model(question=question, context=context)
        self.feature_dict["USE"].append(prediction['answer'])
        if prediction['score'] > 0.5:
            return prediction['answer']
    
    def ask2extract_CONFESSION(self, text):
        context = text
        question = "הנאשם הודה?"
        prediction = self.model(question=question, context=context)
        self.feature_dict["CONFESSION"] = prediction['answer']
        if prediction['score'] > 0.5:
            return prediction['answer']
        