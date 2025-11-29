from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, word_max_features=5000, char_max_features=5000):
        self.word_tfidf = TfidfVectorizer(
            max_features=word_max_features, ngram_range=(1,2), sublinear_tf=True
        )
        self.char_tfidf = TfidfVectorizer(
            analyzer='char', max_features=char_max_features, ngram_range=(3,5), sublinear_tf=True
        )
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit(self, X_raw, X_clean, y=None):
        self.word_tfidf.fit(X_clean)
        self.char_tfidf.fit(X_clean)
        return self

    def transform(self, X_raw, X_clean):
        X_word = self.word_tfidf.transform(X_clean)
        X_char = self.char_tfidf.transform(X_clean)
        X_tfidf = hstack([X_word, X_char])

        X_embed = self.sentence_model.encode(X_raw, show_progress_bar=False)
        X_embed_sparse = csr_matrix(X_embed)

        return hstack([X_tfidf, X_embed_sparse])
