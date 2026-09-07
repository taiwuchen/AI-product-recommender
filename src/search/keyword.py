import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordSearch:
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, k=5):
        if k < 1:
            raise ValueError("Result count must be positive.")
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")
        return [int(i) for i in order if scores[i] > 0][:k]
