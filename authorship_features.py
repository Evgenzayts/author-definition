import numpy as np
from collections import Counter

punct_set = set('.,!?;:')
pronouns = {"я", "ты", "он", "она", "мы", "вы", "они", "меня", "тебя"}
modal_words = {"может", "должен", "надо", "следует", "возможно", "необходимо"}
intro_words = {"итак", "однако", "впрочем", "например", "наконец", "во-первых"}

def token_depth(token):
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
    return depth

def extract_features_single(text, nlp_model, tfidf_word, tfidf_pos):
    doc = nlp_model(text)
    sentences = list(doc.sents)
    tokens = [token for token in doc if token.is_alpha]
    words = [token.text for token in tokens]
    pos_tags = [token.pos_ for token in tokens]

    if not sentences or not words:
        base_features = [0]*12
    else:
        avg_sent_len = np.mean([len(sent) for sent in sentences])
        avg_word_len = np.mean([len(token) for token in words])
        punct_freq = sum(text.count(p) for p in punct_set) / len(text)
        upper_words = sum(1 for w in words if w.isupper())
        word_counts = Counter(words)
        hapax_legomena = sum(1 for count in word_counts.values() if count == 1)
        pron_freq = sum(1 for w in words if w.lower() in pronouns)
        modal_freq = sum(1 for w in words if w.lower() in modal_words)
        intro_freq = sum(1 for w in words if w.lower() in intro_words)
        pos_counts = Counter(pos_tags)
        num_verbs = pos_counts.get("VERB", 0)
        num_adjs = pos_counts.get("ADJ", 0)
        num_advs = pos_counts.get("ADV", 0)
        depths = [token_depth(token) for token in doc]
        avg_tree_depth = np.mean(depths) if depths else 0

        base_features = [
            avg_sent_len, avg_word_len, punct_freq, upper_words,
            hapax_legomena, pron_freq, modal_freq, intro_freq,
            num_verbs, num_adjs, num_advs, avg_tree_depth
        ]

    word_tfidf = tfidf_word.transform([text]).toarray()[0]
    pos_str = ' '.join(pos_tags)
    pos_tfidf = tfidf_pos.transform([pos_str]).toarray()[0]

    return np.concatenate([base_features, word_tfidf, pos_tfidf])
