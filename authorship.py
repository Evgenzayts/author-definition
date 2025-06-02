import argparse
import os
import numpy as np
import spacy
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
from tqdm import tqdm
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from spacy.lang.ru import Russian

# Константы
data_dir = 'data'
min_texts = 5
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Загрузка spaCy и стоп-слов
nlp = spacy.load('ru_core_news_sm')
try:
    russian_stop_words = list(Russian.Defaults.stop_words)
except AttributeError:
    russian_stop_words = []

class StylometricFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        feats = []
        for text in tqdm(X, desc='Stylometric Features'):
            doc = nlp(text)
            sents = list(doc.sents)
            toks = [t for t in doc if t.is_alpha]
            words = [t.text for t in toks]
            if not sents or not words:
                feats.append([0]*14)
                continue
            avg_s = np.mean([len(s) for s in sents])
            avg_w = np.mean([len(w) for w in words])
            L = max(len(text),1)
            punct = sum(ch in set('.,!?;:') for ch in text)/L
            upper = sum(w.isupper() for w in words)
            wc = Counter(words)
            hapax = sum(c==1 for c in wc.values())
            pron = sum(w.lower() in {"я","ты","он","она","мы","вы","они","меня","тебя"} for w in words)
            modal = sum(w.lower() in {"может","должен","надо","следует","возможно","необходимо"} for w in words)
            intro = sum(w.lower() in {"итак","однако","впрочем","например","наконец","во-первых"} for w in words)
            posc = Counter(t.pos_ for t in toks)
            depths = [len(list(tok.ancestors)) for tok in doc]
            tree = np.mean(depths) if depths else 0
            # poetry features
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            last = [l.split()[-1] for l in lines if l.split()][:4]
            rhyme = sum(last[i][-3:]==last[i+1][-3:] for i in range(len(last)-1))
            metaph = sum(any(tok.text.lower() in {"как","словно","подобно","будто"} for tok in sent) for sent in sents)
            feats.append([avg_s,avg_w,punct,upper,hapax,pron,modal,intro,
                          posc.get('VERB',0),posc.get('ADJ',0),posc.get('ADV',0),tree,
                          rhyme,metaph])
        return np.array(feats)

class CharNGram(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(3,5), max_features=1000):
        self.vec = TfidfVectorizer(analyzer='char', ngram_range=ngram_range, max_features=max_features)
    def fit(self, X, y=None): self.vec.fit(X); return self
    def transform(self, X): return self.vec.transform(X)

class SpacyEmbeddings(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        emb = []
        for text in tqdm(X, desc='SpaCy Embeds'):
            doc = nlp(text)
            emb.append(doc.vector)
        return np.vstack(emb)

def visualize_style_features(X, authors, le):
    """Визуализация стилометрических признаков: гистограмма частей речи"""
    pos_counts = {author: Counter() for author in le.classes_}
    for text, author in zip(X, le.inverse_transform(authors)):
        doc = nlp(text)
        pos = [token.pos_ for token in doc]
        pos_counts[author].update(pos)
    
    plt.figure(figsize=(12, 6))
    for i, author in enumerate(le.classes_):
        counts = pos_counts[author]
        total = sum(counts.values())
        freqs = {k: v/total for k, v in counts.items()}
        plt.subplot(2, 2, i+1)
        plt.bar(freqs.keys(), freqs.values())
        plt.title(f'{author} - Распределение частей речи')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pos_distribution.png'))
    plt.close()

def visualize_tfidf(feature_union, texts, authors, le):
    """Визуализация TF-IDF для ключевых слов"""
    tfidf = feature_union.named_transformers['word']
    words = ['любовь', 'смерть', 'природа', 'время', 'ночь']
    X_tfidf = tfidf.transform(texts)
    
    word_indices = [tfidf.vocabulary_.get(word) for word in words]
    valid_words = [words[i] for i, idx in enumerate(word_indices) if idx is not None]
    valid_indices = [idx for idx in word_indices if idx is not None]
    
    if not valid_indices:
        return
    
    tfidf_scores = X_tfidf[:, valid_indices].toarray()
    
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame({
        'author': np.repeat(le.inverse_transform(authors), len(valid_words)),
        'word': np.tile(valid_words, len(texts)),
        'score': tfidf_scores.flatten()
    })
    sns.boxplot(x='author', y='score', hue='word', data=df)
    plt.title('Распределение TF-IDF для ключевых слов')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'tfidf_distribution.png'))
    plt.close()

def visualize_embeddings(embeddings, authors, le):
    """Визуализация эмбеддингов с использованием t-SNE"""
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], 
                    hue=le.inverse_transform(authors), palette='viridis')
    plt.title('t-SNE проекция эмбеддингов текстов')
    plt.savefig(os.path.join(output_dir, 'tsne_embeddings.png'))
    plt.close()

def visualize_stacking():
    """Схема архитектуры стекинга (требует graphviz)"""
    from graphviz import Digraph
    dot = Digraph(comment='Stacking Architecture')
    dot.node('A', 'Исходные данные')
    dot.node('B', 'Логистическая регрессия')
    dot.node('C', 'XGBoost')
    dot.node('D', 'Объединенные признаки')
    dot.node('E', 'Мета-модель (Логистическая регрессия)')
    
    # Исправление: передача кортежей (tail, head)
    edges = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'D'),
        ('C', 'D'),
        ('D', 'E')
    ]
    dot.edges(edges)
    
    dot.render(os.path.join(output_dir, 'stacking_arch'), format='png', cleanup=True)

# Load texts
def load_data():
    texts, authors = [], []
    for auth in os.listdir(data_dir):
        path = os.path.join(data_dir, auth)
        if not os.path.isdir(path): continue
        cnt = 0
        for fn in os.listdir(path):
            fp = os.path.join(path, fn)
            text = open(fp, 'r', encoding='utf-8').read().strip()
            if text:
                texts.append(text)
                authors.append(auth)
                cnt += 1
        if cnt < min_texts:
            raise ValueError(f"Author {auth} has only {cnt} texts")
    return texts, authors

# Main
def train():
    texts, authors = load_data()
    le = LabelEncoder()
    y = le.fit_transform(authors)
    joblib.dump(le, 'le.pkl')
    
    # Разделение данных на train/test
    idx = np.arange(len(texts))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Определение X_train_texts и X_test_texts
    X_train_texts = [texts[i] for i in train_idx]
    X_test_texts = [texts[i] for i in test_idx]
    
    # Визуализация до обучения
    visualize_style_features(texts, y, le)
    
    # Кэшируем эмбеддинги
    embeddings = SpacyEmbeddings().transform(texts)
    visualize_embeddings(embeddings, y, le)
    
    feature_union = FeatureUnion([
        ('stylo', StylometricFeatures()),
        ('word', TfidfVectorizer(ngram_range=(1,2), max_features=2000, stop_words=russian_stop_words or None)),
        ('char', CharNGram()),
        ('embed', SpacyEmbeddings())
    ])
    X_feat = feature_union.fit_transform(X_train_texts)
    scaler = StandardScaler(with_mean=False)
    X_feat = scaler.fit_transform(X_feat)
    joblib.dump(feature_union, 'fu.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Визуализация после извлечения признаков
    visualize_tfidf(feature_union, texts, y, le)
    visualize_embeddings(SpacyEmbeddings().transform(texts), y, le)
    visualize_stacking()

    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1, use_label_encoder=False))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf.fit(X_feat, y_train)
    joblib.dump(clf, 'clf.pkl')

    X_test_feat = scaler.transform(feature_union.transform(X_test_texts))
    y_pred = clf.predict(X_test_feat)
    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred), zero_division=0))


def predict(path):
    le = joblib.load('le.pkl')
    fu = joblib.load('fu.pkl')
    scaler = joblib.load('scaler.pkl')
    clf = joblib.load('clf.pkl')
    text = open(path, 'r', encoding='utf-8').read().strip()
    x = scaler.transform(fu.transform([text]))
    pred = clf.predict(x)[0]
    proba = clf.predict_proba(x)[0]
    print('Predicted:', le.inverse_transform([pred])[0])
    for a,p in zip(le.classes_, proba): print(a, f"{p:.3f}")

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--train', action='store_true')
    p.add_argument('--predict')
    args=p.parse_args()
    if args.train: train()
    elif args.predict: predict(args.predict)
    else: print('Use --train or --predict <file>')

