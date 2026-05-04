import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def get_text_stats(text: str) -> dict:
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    return {
        "length": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "top_words": dict(Counter(words).most_common(15)) if words else {}
    }

def get_text_top_words_fig(top_words: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#f8fafc")
    if top_words:
        sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), palette="viridis", ax=ax)
    ax.set_title("Top 15 Most Common Words")
    plt.tight_layout()
    return fig

def get_text_wordcloud_fig(text: str) -> plt.Figure:
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#f8fafc")
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.set_title("Word Cloud")
        plt.tight_layout()
        return fig
    except ImportError:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "pip install wordcloud requis pour le nuage de mots.", ha="center")
        return fig

def get_text_ngrams_fig(text: str, n: int = 2) -> plt.Figure:
    words = re.findall(r"\b\w+\b", text.lower())
    ngrams = zip(*[words[i:] for i in range(n)])
    ngram_counts = Counter([" ".join(ngram) for ngram in ngrams])
    top_ngrams = dict(ngram_counts.most_common(15))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#f8fafc")
    if top_ngrams:
        sns.barplot(x=list(top_ngrams.values()), y=list(top_ngrams.keys()), palette="magma", ax=ax)
    ax.set_title(f"Top 15 {n}-Grams")
    plt.tight_layout()
    return fig

def get_text_sentiment(text: str) -> dict:
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
    except ImportError:
        # Fallback to simple lexicon approach
        pos_words = set(['good', 'great', 'awesome', 'excellent', 'happy', 'love', 'best', 'positive'])
        neg_words = set(['bad', 'terrible', 'awful', 'sad', 'hate', 'worst', 'negative'])
        words = re.findall(r"\b\w+\b", text.lower())
        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)
        total = max(1, pos_count + neg_count)
        return {
            "polarity": (pos_count - neg_count) / total,
            "subjectivity": "Install TextBlob for better sentiment and subjectivity analysis"
        }

def get_text_complexity_fig(text: str) -> plt.Figure:
    words = re.findall(r"\b\w+\b", text)
    word_lengths = [len(w) for w in words]
    sentences = re.split(r'[.!?]+', text)
    sent_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences if s.strip()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8fafc")
    
    if word_lengths:
        sns.histplot(word_lengths, bins=20, ax=axes[0], color='teal')
        axes[0].set_title(f"Word Lengths (Avg: {sum(word_lengths)/len(word_lengths):.2f})")
    
    if sent_lengths:
        sns.histplot(sent_lengths, bins=20, ax=axes[1], color='coral')
        axes[1].set_title(f"Sentence Lengths (Avg: {sum(sent_lengths)/len(sent_lengths):.2f})")
        
    plt.tight_layout()
    return fig
