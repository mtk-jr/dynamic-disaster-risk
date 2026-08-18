from sklearn.feature_extraction.text import TfidfVectorizer

class SocialTextEncoder:
    def __init__(self, max_features=8):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english"
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()
