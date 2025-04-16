import argparse
import pandas as pd
import numpy as np
import spacy
import joblib
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

from authorship_features import extract_features_single

nlp = spacy.load("ru_core_news_sm")

class StylisticFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.punct_set = set('.,!?;:')
        self.pronouns = {"я", "ты", "он", "она", "мы", "вы", "они", "меня", "тебя"}
        self.modal_words = {"может", "должен", "надо", "следует", "возможно", "необходимо"}
        self.intro_words = {"итак", "однако", "впрочем", "например", "наконец", "во-первых"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in tqdm(X, desc="Извлечение признаков"):
            doc = nlp(text)
            sentences = list(doc.sents)
            tokens = [token for token in doc if token.is_alpha]
            words = [token.text for token in tokens]
            pos_tags = [token.pos_ for token in tokens]

            if not sentences or not words:
                features.append([0]*12)
                continue

            avg_sent_len = np.mean([len(sent) for sent in sentences])
            avg_word_len = np.mean([len(token) for token in words])
            punct_freq = sum(text.count(p) for p in self.punct_set) / len(text)
            upper_words = sum(1 for w in words if w.isupper())
            word_counts = Counter(words)
            hapax_legomena = sum(1 for count in word_counts.values() if count == 1)
            pron_freq = sum(1 for w in words if w.lower() in self.pronouns)
            modal_freq = sum(1 for w in words if w.lower() in self.modal_words)
            intro_freq = sum(1 for w in words if w.lower() in self.intro_words)
            pos_counts = Counter(pos_tags)
            num_verbs = pos_counts.get("VERB", 0)
            num_adjs = pos_counts.get("ADJ", 0)
            num_advs = pos_counts.get("ADV", 0)
            depths = [self._token_depth(token) for token in doc]
            avg_tree_depth = np.mean(depths) if depths else 0

            features.append([
                avg_sent_len, avg_word_len, punct_freq, upper_words,
                hapax_legomena, pron_freq, modal_freq, intro_freq,
                num_verbs, num_adjs, num_advs, avg_tree_depth
            ])
        return np.array(features)

    def _token_depth(self, token):
        depth = 0
        while token.head != token:
            depth += 1
            token = token.head
        return depth

class PosExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [' '.join([token.pos_ for token in nlp(text)]) for text in tqdm(X, desc="POS теги")]

def train():
    print("Обучение модели...")
    df = pd.read_csv("your_dataset.csv")
    top_authors = df['author'].value_counts().head(10).index
    df = df[df['author'].isin(top_authors)]

    X_text = df['text']
    y = df['author']

    X_train_text, _, y_train, _ = train_test_split(X_text, y, test_size=0.2, random_state=42)

    tfidf_word = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
    tfidf_pos = TfidfVectorizer(ngram_range=(1, 3), max_features=300)

    combined_features = FeatureUnion([
        ("style", StylisticFeatures()),
        ("tfidf_word", tfidf_word),
        ("tfidf_pos", Pipeline([
            ('pos', PosExtractor()),
            ('vec', tfidf_pos)
        ]))
    ])

    X_train = combined_features.fit_transform(X_train_text, y_train)
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    joblib.dump(clf, "model.pkl")
    joblib.dump(tfidf_word, "tfidf_word.pkl")
    joblib.dump(tfidf_pos, "tfidf_pos.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Модель обучена и сохранена.")

def predict(file_path):
    print(f"Предсказание для файла: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        unknown_text = f.read()

    clf = joblib.load("model.pkl")
    tfidf_word = joblib.load("tfidf_word.pkl")
    tfidf_pos = joblib.load("tfidf_pos.pkl")
    scaler = joblib.load("scaler.pkl")

    features = extract_features_single(unknown_text, nlp, tfidf_word, tfidf_pos)
    features = scaler.transform([features])

    pred = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]

    print(f"\n📌 Предсказанный автор: {pred}")
    print("\n🔢 Вероятности:")
    for author, prob in zip(clf.classes_, proba):
        print(f"{author}: {prob:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Authorship classifier")
    parser.add_argument("--train", action="store_true", help="Обучить модель")
    parser.add_argument("--predict", type=str, help="Путь к файлу для предсказания")

    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        predict(args.predict)
    else:
        print("❗ Укажи флаг --train или --predict path/to/file.txt")
