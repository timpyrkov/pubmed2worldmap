#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import pylab as plt
import itertools

from wordfreq import word_frequency
import wordcloud
import difflib
import nltk
import re

"""
Utility functions and classes to extract keywords from texts based on
frequency (p) and baseline frequency (q) of the most common form or stem of a word.

Keywords are selected as the words with highest D_KL = p * log(p / q)
"""



def find_keywords(s, text, keywords):
    """
    Find and markup keywords in text
    
    Parameters
    ----------
    text : str
        Text string
    keywords : list
        List of keywords

    Returns
    -------
    str
        Text with markeup keywords
    list
        List of keywords found in text

    """
    regex = re.compile('[^a-zA-Z]')
    words = text.split()
    text = []
    found_keys = []
    for word in words:
        w = regex.sub("", word.lower())
        lem = s.word_forms[w] if w in s.word_forms else ""
        if lem in keywords:
            text.append(f"<b>{word}</b>")
            found_keys.append(lem)
        else:
            text.append(word)
    return " ".join(text), found_keys


def keywords_wordcloud(keywords, bg="#FFFFFF", cmap="cividis"):
    """
    Show a wordcloud of keywords
    
    Parameters
    ----------
    keywords : list
        List of keywords
    bg : str, default "#FFFFFF"
        Background color
    cmap : str
        Matplotlib cmap name, default "cividis"

    """
    cloud = wordcloud.WordCloud(background_color=bg, colormap=cmap,
                                width=1200, height=400)
    cloud.generate(" ".join(keywords))
    #cloud.to_file('wordcloud.png')
    plt.figure(figsize=(12,4), facecolor=bg)
    plt.imshow(cloud)
    plt.axis("off")
    plt.show()
    return




class KeyWordParser():
    """
    Class to parse keywords from text samples

    Parameters
    ----------
    texts : list or None, default None
        List of all texts to calculate baseline word frequencies
        Otherwise use word frequency in english language as the baseline
    extended : bool, default False
        If False - match words to stems; If True - find close matches using "difflib"
        (works slowly, speed-up by providing pre-calculated dictionary as "word_forms")
    word_forms : dict or None, default None
        Use pre-calculated dictionary of close matches to common word forms

    Attributes
    ----------
    lemmatizer : class object
        WordLemmatizer() initialized with parameters "word_forms" and "extended"
    stoplist : list
        List of words to exclude from keywords
    baseline_freq : dict or None
        Dictionary of word baseline frequencies

    """
    def __init__(self, texts=None, extended=False, word_forms=None):
        self.lemmatizer = WordLemmatizer(extended=extended, word_forms=word_forms)
        self.stoplist = ["university", "institute", "center", "department", 
                         "laboratory", "faculty", "research", "national", "unit"]
        if hasattr(self.lemmatizer, "stoplist"):
            self.stoplist = self.stoplist + self.lemmatizer.stoplist
        self.baseline_freq = None
        if texts is not None:
            words = self.texts_to_lemmas(texts)
            self.baseline_freq = self.word_freq(words)


    def texts_to_lemmas(self, texts):
        """
        Convert text to list of common word forms
        """
        words = self.texts_to_words(texts)
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return words


    @staticmethod
    def texts_to_words(texts):
        """
        Convert text to list of words
        """
        regex = re.compile('[^a-zA-Z ]')
        is_list = isinstance(texts, (list, np.ndarray))
        if isinstance(texts, str):
            text = texts
        elif is_list and isinstance(texts[0], str):
            text = " ".join(texts)
        elif is_list and isinstance(texts[0], list):
            text = list(itertools.chain(*texts))
            text = " ".join(text)
        else:
            msg = f"Failed to extract words from {texts}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None
        text = text.lower()
        words = regex.sub(" ", text).split()
        words = np.array(words)
        return words

    
    @staticmethod
    def word_freq(words, return_counts=False):
        """
        Calculate word counts or frequencies
        """
        freq = dict(Counter(words).most_common())
        if not return_counts:
            total = sum(list(freq.values()))
            total = max(1, total)
            freq = {w: freq[w] / total for w in freq}
        return freq


    def get_baseline_freq(self, words, epsilon=1e-10):
        """
        Get baseline frequency of words
        """
        if self.baseline_freq is None:
            freq = {w: max(epsilon, word_frequency(w, "en")) for w in words}
        else:
            freq = {w: self.baseline_freq[w] if w in self.baseline_freq else epsilon for w in words}
        return freq


    def keyword_stats_df(self, texts):
        """
        Calculate words frequency and D_KL = p * log (p / q) in a sample of text,
        where p - frequency in the given text sample, q - baseline frquency of a word
        """
        words = self.texts_to_lemmas(texts)
        bfreq = self.get_baseline_freq(words)
        freq = self.word_freq(words)
        # Calc lemma stats
        logratio = {}
        dkl = {}
        for k, v in freq.items():
            logratio[k] = np.log10(freq[k] / bfreq[k])
            dkl[k] = freq[k] * np.log(freq[k] / bfreq[k])
        d = {"freq": freq, "bfreq": bfreq, "dkl": dkl}
        d = pd.DataFrame.from_dict(d)
        return d


    def keywords(self, texts, nmax=5):
        """
        Select keywords as the words with abnormally high frequency in a sample of text
        """
        d = self.keyword_stats_df(texts)
        d.sort_values(by="dkl", inplace=True, ascending=False)
        n = int(np.round(d.shape[0] / 5))
        n = max(1, min(nmax, n))
        d = d[:n]
        keywords = [w for w in d.index.values if len(w) >= 3 and w not in self.stoplist]
        return keywords










class WordLemmatizer():
    """
    Class to convert words to stems or the most common form (if extended == True)

    Parameters
    ----------
    texts : list or None, default None
        List of all texts to calculate baseline word frequencies
        Otherwise use word frequency in english language as the baseline
    extended : bool, default False
        If False - match words to stems; If True - find close matches using "difflib"
        (works slowly, speed-up by providing pre-calculated dictionary as "word_forms")
    word_forms : dict or None, default None
        Use pre-calculated dictionary of close matches to common word forms

    Attributes
    ----------
    stemmer : class object
        "nltk" stemmer
    stoplist : list
        List of words to exclude from keywords
    whitelist : list
        List of words to skip stemmer

    """
    def __init__(self, texts=None, extended=False, word_forms=None):
        self.extended = extended
        # https://dictionary.cambridge.org/grammar/british-grammar/word-formation/prefixes
        self.prefixes = {
            "anti": "",    # e.g. antibacterial
            "auto": "",    # e.g. autobiography, automobile
            "co": "",      # e.g. co-existance, coeducation
            "chrono": "",  # e.g. chronotype
            # "de": "",      # e.g. de-classify, decontaminate, demotivate
            "dis": "",     # e.g. disagree, displeasure, disqualify
            "down": "",    # e.g. downgrade, downhearted
            "epi": "",     # e.g. epicenter, epigraph
            "extra": "",   # e.g. extraordinary, extraterrestrial
            # "hyper": "",   # e.g. hyperactive, hypertension
            # "il": "",     # e.g. illegal
            # "im": "",     # e.g. impossible
            # "in": "",     # e.g. insecure
            # "ir": "",     # e.g. irregular
            "inter": "",  # e.g. interactive, international
            "mega": "",   # e.g. megabyte, mega-deal, megaton
            "mid": "",    # e.g. midday, midnight, mid-October
            "mis": "",    # e.g. misaligned, mislead, misspelt
            "non": "",    # e.g. non-payment, non-smoking
            "over": "",  # e.g. overcook, overcharge, overrate
            "out": "",    # e.g. outdo, out-perform, outrun
            "post": "",   # e.g. post-election, post-warn
            "pre": "",    # e.g. prehistoric, pre-war
            "pro": "",    # e.g. pro-communist, pro-democracy
            # "re": "",     # e.g. reconsider, redo, rewrite
            "semi": "",   # e.g. semicircle, semi-retired
            "sub": "",    # e.g. submarine, sub-Saharan
            "super": "",   # e.g. super-hero, supermodel
            "tele": "",    # e.g. television, telephathic
            "trans": "",   # e.g. transatlantic, transfer
            "ultra": "",   # e.g. ultra-compact, ultrasound
            # "un": "",      # e.g. under-cook, underestimate
            # "up": "",      # e.g. upgrade, uphill
        }
        try:
            #self.lemmatizer = nltk.wordnet.WordNetLemmatizer()
            # self.stemmer = nltk.stem.SnowballStemmer("english")
            # self.stemmer = nltk.stem.LancasterStemmer(strip_prefix_flag=True)
            self.stemmer = nltk.stem.porter.PorterStemmer()
            self.stoplist = nltk.corpus.stopwords.words("english")
            self.whitelist = list(nltk.corpus.wordnet.words()) + nltk.corpus.words.words()
        except:
            nltk.download('words')
            nltk.download("wordnet")
            nltk.download("omw-1.4")
            nltk.download("stopwords")
            self.stemmer = nltk.stem.porter.PorterStemmer()
            self.stoplist = nltk.corpus.stopwords.words("english")
            self.whitelist = list(nltk.corpus.wordnet.words()) + nltk.corpus.words.words()
        self.word_forms = {}
        if isinstance(word_forms, dict):
            self.word_forms = word_forms
        if word_forms is None or isinstance(word_forms, str):
            if os.path.exists(str(word_forms)):
                df = pd.read_csv(word_forms, delimiter=";", index_col=0, keep_default_na=False)
                self.word_forms = df[df.columns[0]].to_dict()
            elif texts is not None:
                words = KeyWordParser.texts_to_words(texts)
                word_freq = KeyWordParser.word_freq(words)
                words = np.array(list(word_freq.keys()))
                freqs = np.array(list(word_freq.values()))
                mask = [w not in self.stoplist for w in words]
                words = words[mask]
                freqs = freqs[mask]
                n = len(words)
                roots = np.array([self.full_stemmer(w) for w in words])
                root_idx = dict(zip(roots[::-1], np.arange(n)[::-1]))
                # Extended flag grops words by close match of roots
                # Otherwise words are grouped only by exact match of roots
                root_forms = self.find_root_forms(root_idx)
                parent = np.zeros((n)).astype(int) - 1
                for i, root in enumerate(roots):
                    idx = np.array([root_idx[r] for r in root_forms[root]])
                    if parent[i] < 0:
                        parents = parent[idx]
                        parents = parents[parents >= 0]
                        parent[i] = parents.min() if len(parents) > 0 else i
                    idx = idx[idx > i]
                    for j in idx:
                        parent[j] = parent[i]
                self.word_forms = {}
                for p in np.unique(parent):
                    mask = parent == p
                    word = list(dict.fromkeys(words[mask]))
                    for w in word:
                        self.word_forms[w] = word[0]
                if isinstance(word_forms, str):
                    df = pd.DataFrame.from_dict(self.word_forms, orient="index")
                    df.columns = ["WORD"]
                    df.index.name = "FORM"
                    df.to_csv(word_forms, sep=";")
            else:
                msg = f"Failed to initialize word forms dictionary: texts={texts}, word_forms={word_forms}"
                warnings.warn(msg, UserWarning, stacklevel=2)
        words = list(dict.fromkeys(list(self.word_forms.values())))
        self.word_roots = {self.full_stemmer(w): w for w in words}


    def full_stemmer(self, word):
        """
        Stem words with removed suffixes and prefixes
        """
        root = self.remove_prefix_suffix(word.lower(), prefixes=self.prefixes)
        root = self.stemmer.stem(root)
        return root
        

    @staticmethod
    def remove_prefix_suffix(word, prefixes={}):
        """
        Removed suffixes and prefixes from words
        """
        w = word
        # remove suffix
        for suffix in ["log", "path", "ther"]:
            if w.find(suffix) > 0:
                w = w.split(suffix)[0]
        # remove prefix
        # https://stackoverflow.com/questions/52140526/python-nltk-stemmers-never-remove-prefixes
        for prefix in sorted(prefixes, key=len, reverse=True):
            w_, nsub = re.subn("{}[\-]?".format(prefix), "", w)
            if nsub > 0 and len(w_) > 5:
                return w_
        return w

 
    def find_root_forms(self, root_idx):
        """
        Find close matches of word roots (stems / most common forms) if extended == True
        """
        keys = sorted(root_idx, key=root_idx.get)
        root_idx = {k: root_idx[k] for k in keys}
        roots = list(root_idx.keys())
        if not self.extended:
            root_forms = {root: [root] for root in roots}
        else:
            root_forms = {}
            for i, root in enumerate(tqdm(roots, desc="Extended root forms")):
                cutoff = self.cutoff(root)
                ancestors = difflib.get_close_matches(root, roots[:i+1], n=50, cutoff=cutoff)
                descendants = difflib.get_close_matches(root, roots[i+1:], n=50, cutoff=cutoff)
                forms = [r for r in ancestors if r[0] == root[0]]
                for r in descendants:
                    if r[0] == root[0]:
                        forms.append(r)
                    elif len(root) > 5 and len(r) > 5:
                        # First letter lost
                        if r[:2] == root[1:3]:
                            forms.append(r)
                        # First letter added
                        elif r[1:3] == root[:2] and r[0] in list("abcde"):
                            forms.append(r)
                root_forms[root] = forms
        return root_forms


    @staticmethod
    def cutoff(word):
        """
        Set "difflib" close match score cutoff based on word length
        """
        cutoff = np.clip(0.75 + 0.05 * (10 - len(word)), 0.80, 0.90)
        return cutoff


    def lemmatize(self, word):
        """
        Get the most common form of a word
        """
        w = word.lower()
        if w not in self.word_forms:
            root = self.full_stemmer(w)
            if self.extended:
                cutoff = self.cutoff(root)
                roots = difflib.get_close_matches(root, list(self.word_roots.keys()), n=50, cutoff=cutoff)
                if len(roots) > 0:
                    root = roots[0]
            if root not in self.word_roots:
                self.word_roots[root] = w
            self.word_forms[w] = self.word_roots[root]
        return self.word_forms[w]







