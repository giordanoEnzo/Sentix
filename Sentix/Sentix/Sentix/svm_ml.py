import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pickle
from BuilderProcessor import TextProcessorBuilder


class SVMModel:
    _instance = None

    def __new__(cls, model_path="models/svm_model.pkl"):
        if cls._instance is None:
            cls._instance = super(SVMModel, cls).__new__(cls)
            with open(model_path, 'rb') as file:
                cls._instance.model, cls._instance.vectorizer = pickle.load(file)
        return cls._instance

    def predict(self, text: str):
        bow_representation = self.vectorizer.transform([text]).toarray()
        prediction = self.model.predict(bow_representation)

        if prediction[0] == 0:
            return "Negativo"
        elif prediction[0] == 1:
            return "Positivo"
        else:
            return "Impossível analisar o resultado."

    def correct_and_retrain(self, text: str, correct_label: str):
        df = pd.read_excel('models/mt_sentimento.xlsx')

        new_row = {'Caracteristicas': text, 'Target': correct_label}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_excel('models/mt_sentimento.xlsx', index=False)

        self.retrain_model(df)

        return "Muito obrigado! Aprendi."

    def retrain_model(self, df):
        text_processor = TextProcessorBuilder().set_stem(False).build()

        X = df.Caracteristicas.apply(text_processor.process)
        y = df.Target

        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(X)
        X_final = bow.toarray()

        y_indexed = [1 if i.lower() == "positivo" else 0 for i in y]

        unique_classes = set(y_indexed)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Não é possível treinar novamente o modelo: apenas uma classe está presente no conjunto de dados. Classe atual: {unique_classes}")

        clf = svm.SVC()
        clf.fit(X_final, y_indexed)

        with open('models/svm_model.pkl', 'wb') as file:
            pickle.dump((clf, vectorizer), file)

        self._instance.model = clf
        self._instance.vectorizer = vectorizer

