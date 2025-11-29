from src.utils.utils import load_data, save_objects
from src.scripts.preprocessor import TextPreprocessor
from nltk.corpus import stopwords
import re
import numpy as np
import string
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

    
if __name__ == "__main__":

    df = load_data('data/datasets/final_data/data.csv')
    print("Data Loaded")

    df['clean_text'] = df['statement'].apply(clean_text)

    X_train_raw, X_test_raw, X_train_clean, X_test_clean, y_train, y_test = train_test_split(
    df['statement'],
    df['clean_text'],
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)
    
    preprocessor = TextPreprocessor()
    preprocessor.fit(X_clean=X_train_clean, X_raw=X_train_raw)
    save_objects(preprocessor, "preprocessor.joblib")

    X_train_final = preprocessor.transform(X_train_raw, X_train_clean)


    estimators = [
        ('lr_model', LogisticRegression(max_iter=1000, random_state=42)),
        ('linearSVC', CalibratedClassifierCV(LinearSVC(), cv=3)),
        ('xgb', XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))

    ]
    classifier =  StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3,
        stack_method='predict_proba',
        passthrough=False
    )
    classifier.fit(X_train_final, y_train)
    save_objects(classifier, 'classifier.joblib')