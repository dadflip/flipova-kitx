import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import re

def _check_bs4():
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup
    except ImportError:
        return None

def get_web_meta(html_content):
    BeautifulSoup = _check_bs4()
    if not BeautifulSoup: return {"Error": "beautifulsoup4 required"}
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string if soup.title else "No Title"
        meta_tags = soup.find_all('meta')
        meta_info = {}
        for tag in meta_tags:
            if tag.get('name'):
                meta_info[tag.get('name')] = tag.get('content')
            elif tag.get('property'):
                meta_info[tag.get('property')] = tag.get('content')
        return {"Title": title, "Meta Tags": meta_info}
    except Exception as e:
        return {"Error": str(e)}

def get_web_links_fig(html_content):
    BeautifulSoup = _check_bs4()
    if not BeautifulSoup: return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
        
        internal = [l for l in links if l.startswith('/') or l.startswith('#')]
        external = [l for l in links if l.startswith('http')]
        other = len(links) - len(internal) - len(external)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#f8fafc")
        ax.pie([len(internal), len(external), other], labels=["Internal", "External", "Other"], autopct="%1.1f%%")
        ax.set_title(f"Links Distribution (Total: {len(links)})")
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def get_web_dom_distribution_fig(html_content):
    BeautifulSoup = _check_bs4()
    if not BeautifulSoup: return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        tags = [tag.name for tag in soup.find_all(True)]
        counts = Counter(tags)
        top_tags = dict(counts.most_common(15))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#f8fafc")
        sns.barplot(x=list(top_tags.values()), y=list(top_tags.keys()), palette="crest", ax=ax)
        ax.set_title("Top 15 DOM Tags")
        ax.set_xlabel("Count")
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def get_web_wordcloud_fig(html_content):
    BeautifulSoup = _check_bs4()
    if not BeautifulSoup: return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        from wordcloud import WordCloud
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#f8fafc")
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Content Word Cloud")
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def get_web_text_stats(html_content):
    BeautifulSoup = _check_bs4()
    if not BeautifulSoup: return {"Error": "beautifulsoup4 required"}
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        words = re.findall(r"\b\w+\b", text.lower())
        char_count = len(text)
        word_count = len(words)
        
        h1_count = len(soup.find_all('h1'))
        h2_count = len(soup.find_all('h2'))
        h3_count = len(soup.find_all('h3'))
        
        return {
            "Total Characters": char_count,
            "Total Words": word_count,
            "H1 Tags": h1_count,
            "H2 Tags": h2_count,
            "H3 Tags": h3_count
        }
    except Exception as e:
        return {"Error": str(e)}
