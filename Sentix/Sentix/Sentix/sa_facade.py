from svm_ml import SVMModel
from BuilderProcessor import TextProcessorBuilder


class SentimentAnalysisFacade:
    def __init__(self):
        self.model = SVMModel()

        self.text_processor = TextProcessorBuilder().set_stem(False).set_remove_stopwords(False).build()

    def analyze_sentiment(self, text: str):
        processed_text = self.text_processor.process(text)
        sentiment = self.model.predict(processed_text)
        return sentiment

    def correct_sentiment(self, text: str, correction: str):
        return self.model.correct_and_retrain(text, correction)
