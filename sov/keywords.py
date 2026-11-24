"""c-TF-IDF keywords per cluster, used as hints for the naming prompt.

Only the keyword counting ignores links and @handles; the posts themselves are
never changed.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

SPANISH_STOP_WORDS = set("""
a al algo algunas algunos ante antes como con contra cual cuando de del desde donde
durante e el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso
esos esta estaba estado estais estamos estan estar estas este esto estos estoy fue
fueron fui ha habia han has hasta hay la las le les lo los mas me mi mis mucho muy
nada ni no nos nosotros o os otra otro para pero poco por porque que quien se sea
ser si sin sobre son su sus tambien te tengo tiene tienen todo todos tu tus un una
uno unos y ya yo él más qué también está están así sí cómo ser hacer puede pueden
esta este ese esa eso vez bien sólo solo cada hoy día años año gracias
""".split())

# Tokens that show up in nearly every post here and say nothing about the sub-theme.
DOMAIN_STOP_WORDS = {"rt", "amp", "https", "http", "co", "www", "com"}

STOP_WORDS = list(ENGLISH_STOP_WORDS | SPANISH_STOP_WORDS | DOMAIN_STOP_WORDS)


def _for_counting(text):
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"@\w+", " ", text)


def ctfidf_keywords(texts, labels, top_n=12):
    """Return {label: [keywords]} for every non-noise label."""
    clusters = sorted(l for l in set(labels) if l >= 0)
    if not clusters:
        return {}
    docs = [" ".join(_for_counting(t) for t, l in zip(texts, labels) if l == c) for c in clusters]
    vec = CountVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=1,
                          token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    counts = vec.fit_transform(docs).toarray().astype(float)
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    avg_words = counts.sum() / len(clusters)
    idf = np.log(1 + avg_words / np.maximum(counts.sum(axis=0), 1))
    scores = tf * idf
    vocab = np.array(vec.get_feature_names_out())
    return {c: vocab[np.argsort(-scores[i])[:top_n]].tolist() for i, c in enumerate(clusters)}
