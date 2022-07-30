#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import string
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import scipy.cluster.hierarchy as sch

import pylab as plt
import itertools

import geoutils
from wordutils import *
from scraperutils import update_dict


"""
Utility functions and classes to parse geographic names as well as 
year, title, author names, affiliations from PubMed format data files.
"""



def parse_geo_data(s, path=None):
    """
    Parse geographic names (city, country/US state) from author affiliations

    Parameters
    ----------
    s : class object
        PubMedScraper() instance object
    path : str or None, default None
        Folder to store "geopandas" maps and 
        "geonamescahce" data

    Returns
    -------
    dict
        Number of publications per country/US state
        (can be used to show world map by "geopandas")

    """
    # Make affs dictionary and parse geo data
    affs = list(s.author_affs.values())
    affs = list(itertools.chain(*affs))
    affs = np.unique(np.array(affs))
    aff_country = {}
    aff_city = {}
    g = geoutils.GeoTextParser(path)
    for aff in tqdm(affs, desc="Parse affiliations geo data"):
        codes, cities = g.find_counries_cities(aff)
        aff_country[aff] = codes
        aff_city[aff] = cities
    # Retrieve auther geo data (country/state code, city)
    # from affs dictionary
    s.author_country = {}
    s.author_cities = {}
    for a in s.author_affs:
        affs = s.author_affs[a]
        country = [aff_country[aff] for aff in affs]
        city = [aff_city[aff] for aff in affs]
        country = list(itertools.chain(*country))
        city = list(itertools.chain(*city))
        loc = np.array([f"{country[i]}___{city[i]}" for i in range(len(country))])
        if len(loc) > 0:
            loc = np.array(Counter(loc).most_common()).T[0]
            country, city = loc[0].split("___")
            s.author_country[a] = country
            s.author_cities[a] = []
            for loc_ in loc:
                country, city = loc_.split("___")
                if country == s.author_country[a] and city not in s.author_cities[a]:
                    s.author_cities[a].append(city)
        else:
            s.author_country[a] = ""
            s.author_cities[a] = []
    # Retrieve paper (pmid) geo data (country/state code) from affs dictionary
    s.pmid_country = {}
    for p in s.pmid_affs:
        affs = s.pmid_affs[p]
        country = [aff_country[aff] for aff in affs]
        country = list(itertools.chain(*country))
        country = list(np.unique(np.array(country)))
        s.pmid_country[p] = country
    return npmid_by_country(s)


def npmid_by_country(s, start=None, end=None):
    """
    Number of publications per country/US state

    Parameters
    ----------
    start : int or None, default None
       Min year cutoff

    end : int or None, default None
        Max year cutoff

    Returns
    -------
    dict
        Number of publications per country/US state
        (can be used to show world map by "geopandas")

    """
    def valid_year(y, start, end):
        return y >= start and y <= end
    if start is None:
        start = min(s.pmid_year.values())
    if end is None:
        end = max(s.pmid_year.values())
    pmids = [p for p in s.pmid_year if valid_year(s.pmid_year[p], start, end)]
    codes = [s.pmid_country[p] for p in pmids if p in s.pmid_country]
    codes = list(itertools.chain(*codes))
    codes, counts = np.unique(codes, return_counts=True)
    dct = dict(zip(codes, counts))
    return dct


def sort_dict(dct):
    """
    Utility function to sort dictionary descending by values
    """
    x = np.array(list(dct.keys()))
    y = [len(dct[x_]) for x_ in x] if isinstance(dct[x[0]], list) else list(dct.values())
    y = np.array(y)
    idx = np.argsort(y)[::-1]
    x = x[idx]
    y = y[idx]
    sorted_dct = {x_: dct[x_] for x_ in x}
    return sorted_dct


def item_id_dict(samples, features, nmax=None):
    """
    Utility function to match unique items sorted descending to id
    """
    item_id = [features[s_] for s_ in samples]
    item_id = list(itertools.chain(*item_id))
    item_id = np.array(Counter(item_id).most_common()).T[0]
    item_id = item_id if nmax is None else item_id[:nmax]
    item_id = dict(zip(item_id, np.arange(len(item_id))))
    return item_id


def connectivity_mask(func):
    """
    Decorator to mask connectivity matrix by another connectivity matrix
    """
    def wrapper(*args, **kwargs):
        c = func(*args, **kwargs)
        if "mask" in kwargs:
            for feat in kwargs["mask"]:
                cmask = connectivity_matrix(args[0], feat)
                cmask = (cmask + np.eye(len(cmask)) > 0).astype(float)
                c = c * cmask
        return c
    return wrapper


@connectivity_mask
def connectivity_matrix(samples, features, mask=[], nmax=None):
    """
    Utility function to calculate connectivity matrix
    """
    x = data_matrix(samples, features, nmax=nmax)
    c = np.dot(x, x.T)
    return c


def combined_samples(func):
    """
    Decorator to calculate data matrix for combined samples
    """
    def wrapper(*args, **kwargs):
        if isinstance(args[0][0], list):
            samples = list(itertools.chain(*args[0]))
            features = args[1]
            x = data_matrix(samples, features, **kwargs)
            idx = [0] + [len(s_) for s_ in args[0]]
            idx = np.cumsum(np.array(idx))
            x = np.stack([np.max(x[idx[i]:idx[i+1]], 0) for i in range(len(idx)-1)])
        else:
            x = func(*args, **kwargs)
        return x
    return wrapper


@combined_samples
def data_matrix(samples, features, nmax=None, item_id=None):
    """
    Utility function to calculate data matrix
    """
    if item_id is None:
        item_id = item_id_dict(samples, features, nmax=nmax)
    x = np.zeros((len(samples), len(item_id)))
    for i, s_ in enumerate(samples):
        for item in features[s_]:
            if item in item_id:
                j = item_id[item]
                x[i,j] += 1
    return x


def normalize_symmetric_matrix(x):
    """
    Utility function to normalize connectivity matrix
    """
    norm = np.diag(x)[..., None]
    norm = np.tile(norm, x.shape[0])
    norm = np.stack([norm, norm.T])
    norm = np.min(norm, axis=0)
    norm = np.clip(norm, 1, None)
    x = x / norm
    return x


def connectivity_to_distance(c, full_matrix=False):
    """
    Utility function to calculate distances from connectivity matrix
    """
    d = 1.0 - normalize_symmetric_matrix(c)
    if not full_matrix:
        d = [d[i][i+1:] for i in range(len(c)-1)]
        d = np.concatenate(d)
    return d


def covariance_matrix(x, normalize=False):
    """
    Utility function to calculate data covariance matrix
    """
    x_ = x - np.nanmean(x, axis=0)
    if normalize:
        x_ = x_ / np.nanstd(x_, axis=0)
    cov = np.dot(x_.T, x_) / float(x_.shape[1] - 1)
    return cov

def pca(x, normalize=False):
    """
    Utility function to calculate PCA
    """
    c = covariance_matrix(x, normalize=normalize)
    _, w, v = np.linalg.svd(c, full_matrices=False)
    return w, v


def hierarchical_cluster_linkage(d, cutoff=0.9, mode="distance"):
    """
    Utility function to apply "scipy" hierarchical clustering
    """
    linkage = sch.linkage(d, method='complete')
    if mode == "knee_point":
        d_cutoff = knee_point(linkage)
        idx = sch.fcluster(linkage, d_cutoff, "distance")
    else:
        idx = sch.fcluster(linkage, cutoff, mode)
    return idx, linkage


def plot_dendrogram(linkage):
    """
    Utility function to show "scipy" hierarchical clustering dendrogram
    """
    tree = sch.dendrogram(linkage, orientation='left')
    plt.yticks([])
    ax = plt.gca()
    for e in ['top', 'bottom', 'right', 'left']:
        ax.spines[e].set_color("#00000000")
    return tree


def renumber_clusters(idx):
    """
    Utility function to renumber clusters ascending by number of members
    (Clusters with less members have higher number)
    """
    d = dict(Counter(idx).most_common())
    for v, k in enumerate(d):
        d[k] = v
    idx = np.vectorize(d.get)(idx)
    return idx


def hierarchical_clustering(samples, features, mask=[], plot=False, cutoff=0.9, mode="distance"):
    """
    Utility function to cluster teams/publications by keywords

    Example
    -------
    >>> idx, _ = hierarchical_clustering(keywords, keyword_teams, cutoff=0.99)

    """
    c = connectivity_matrix(samples, features, mask=mask)
    d = connectivity_to_distance(c)
    idx, linkage = hierarchical_cluster_linkage(d, cutoff=cutoff, mode=mode)
    idx = renumber_clusters(idx)
    if plot:
        plt.figure(figsize=(18,9), facecolor="w")
        plt.subplot2grid((1,3), (0,0))
        tree = plot_dendrogram(linkage)
        i = tree['leaves']
        plt.subplot2grid((1,3), (0,1), colspan=2)
        plt.yticks(np.arange(len(samples))+0.5, samples)
        plt.pcolor(c[i,:][:,i])
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    return idx, linkage


def density_clustering(samples, features, mask=[], density=None):
    """
    Utility function to cluster authors by publications/affiliations

    Example
    -------
    >>> idx = density_clustering(authors, s.author_pmids, 
    >>>                          mask=[s.author_cities, s.author_affs],
    >>>                          density=[s.author_score[a] for a in authors])

    """
    c = connectivity_matrix(samples, features, mask=mask)
    if density is None:
        density = np.sum(c, axis=0)
    idx = np.argsort(density)
    idx0 = np.argsort(idx)
    samples = np.array(samples)[idx]
    density = np.array(density)[idx]
    c = c[idx,:][:,idx]
    
    # Assign centroids id
    n = c.shape[0]
    super_id = np.arange(n)
    for i in range(1,n):
        mask = c[i] > 0
        if np.sum(mask[:i]) > 0:
            super_id[i] = np.arange(n)[mask][0]
    # Assign members id
    idx = np.zeros((n)).astype(int)
    for i in range(n):
        if super_id[i] == i:
            idx[i] = idx.max() + 1
    for i in range(n):
        if idx[i] == 0:
            j = super_id[i]
            idx[i] = idx[j]
    idx = idx[idx0]
    idx = renumber_clusters(idx)
    return idx



def cluster_sanity_check(idx, samples, features, mask=[]):
    """
    Utility function to show sorted dendrogram and connectivity matrix

    Example
    -------
    >>> cluster_sanity_check(idx, authors, s.author_pmids, 
    >>>                      mask=[s.author_cities, s.author_affs])

    """
    print(len(idx), len(np.unique(idx)))
    idx1 = []
    for i in np.unique(idx):
        for j in range(len(idx)):
            if idx[j] == i:
                idx1.append(j)
    samples_ = np.array(samples)[idx1]
    idx_ = idx[idx1]
    c = connectivity_matrix(samples_, features, mask=mask)
    plt.figure(figsize=(10,8), facecolor="w")
    plt.pcolor(c)
    plt.yticks(np.arange(len(idx_))+0.5, idx_)
    plt.colorbar()
    plt.show()
    return


def smooth(x, y, window=0.01):
    """
    Utility function to smooth a knee point curve
    """
    y_ = np.zeros_like(y)
    for i, x_ in enumerate(x):
        mask = (x >= x_ - window) & (x <= x_ + window)
        y_[i] = np.mean(y[mask])
    return y_


def knee_point(linkage, plot=False):
    """
    Utility function to find cuoff knee point for hierarchical clustering
    """
    x = linkage[:,2]
    y = np.arange(len(x))[::-1]
    y0 = y.max()
    y = smooth(x, y) - y0 * (1 - x)
    i = np.argmax(y)
    if plot:
        plt.figure(figsize=(8,4), facecolor="w")
        plt.title(f"{i} d = {x[i]}")
        plt.plot(x, y)
        plt.show()
    return x[i]


def update_pmid_keywords(s, keywords, abstract=False):
    """
    Utility function to assign/update keywords for PubMed ids
    """
    for pmid in s.pmids:
        text = s.pmid_title[pmid]
        if abstract:
            text = text + " " + s.pmid_abstract[pmid]
        _, keys = find_keywords(s, text, keywords)
        pmid_keys = s.pmid_keywords[pmid]
        for k in keys:
            if k not in pmid_keys:
                pmid_keys.append(k)
        s.pmid_keywords[pmid] = pmid_keys
    s.keyword_pmids = keywords_to_items(s.pmid_keywords)
    return


def keywords_to_items(item_keywords, keywords=None, items=None):
    """
    Utility function to convert item -> keywords dictionary
    to keyword -> items dictionary
    """
    if keywords is None:
        keywords = list(item_keywords.values())
        keywords = list(itertools.chain(*keywords))
        keywords = list(np.unique(keywords))
    if items is None:
        items = list(item_keywords.keys())
    keyword_items = {}
    for key in keywords:
        key_items = []
        for t in items:
            if key in item_keywords[t]:
                key_items.append(t)
        keyword_items[key] = key_items
    return keyword_items


def keyword_clustering(s, nmaxclust=100):
    """
    Utility function to cluster teams/publications by keywords
    """
    """ Teams with at least 2 publications """
    team_list = np.array(list(s.team_pmids.keys()))
    team_list = np.array([t for t in s.team_pmids if len(s.team_pmids[t]) >= 2])
    """ Select up to 1000 top keywords """
    keywords = [s.team_keywords[t] for t in team_list]
    keywords = list(itertools.chain(*keywords))
    keywords, counts = np.array(Counter(keywords).most_common()).T
    s.ranked_keywords = list(keywords)
    counts = counts.astype(int)
    cutoff = 0.01 + np.round(counts / counts.max(), 2)[:1000][-1]
    keywords = keywords[counts > cutoff * counts.max()]
    s.top_keywords = list(keywords)
    s.topic_keywords = {}
    """ Assign pmid keywords """
    update_pmid_keywords(s, s.top_keywords)
    """ Dictionary keyword -> teams """
    keyword_teams = keywords_to_items(s.team_keywords, keywords, team_list)
    idx, linkage = hierarchical_clustering(keywords, keyword_teams, cutoff=0.99)
    idxs = np.unique(idx)
    if len(idxs) < nmaxclust:
        for i in idxs:
            words = keywords[idx == i]
            s.topic_keywords[i] = [k for k in keywords if k in words]
    else:
        keywords_ = [list(keywords[idx == i]) for i in idxs]
        idx, linkage = hierarchical_clustering(keywords_, keyword_teams, mode="knee_point")
        idxs = np.unique(idx)
        if len(idxs) > nmaxclust:
            idx, linkage = hierarchical_clustering(keywords_, keyword_teams, 
                                                   cutoff=nmaxclust, mode="maxclust")
            idxs = np.unique(idx)
        for i in idxs.astype(int):
            words = [keywords_[j] for j in range(len(idx)) if idx[j] == i]
            words = list(itertools.chain(*words))
            s.topic_keywords[i] = [k for k in keywords if k in words]
    print("TOPICS:", len(idxs))
    return


def get_contry_authors(s, country=None):
    """
    Utility function to get author list by country code/US state
    """
    country_auth = {}
    for a in s.author_country:
        update_dict(country_auth, s.author_country[a], a)
    country_list = geoutils.iso_codes()
    country_auth = {c: country_auth[c] for c in country_list if c in country_auth}
    if country is not None:
        country_auth = country_auth[country]
    return country_auth


def sort_pmids_by_year(s, pmids=None):
    """
    Utility function to sort PubMed ids by year of publication
    """
    if pmids is None:
        pmids = s.pmids
    years = np.array([s.pmid_year[p] for p in pmids])
    idx = np.argsort(years)[::-1]
    return list(np.array(pmids)[idx])


def cluster_topics(s, extended=False, nmaxclust=100):
    """
    Cluster PubMed ids to topics by keywords

    Parameters
    ----------
    s : class object
        PubMedScraper() instance object
    extended : bool, default False
        If False - match words to stems; If True - use extended search of close matches 
        based on "difflib", works slowly, can be initialized by pre-calculated dictionary
    maxclust : int, default 100
        "maxclust" parameter for "scipy" hierarchical clustering
        (if "scipy" cannot satisfy "maxclsut", it will assign all samples to a single cluster)

    """
    s.team_authors = {}
    s.team_pmids = {}
    s.team_affs = {}
    s.team_city = {}
    s.team_country = {}
    s.city_teams = {}
    country_auths = get_contry_authors(s)
    for country in tqdm(country_auths, desc="Cluster authors"):
        author_score = {a: s.author_score[a] for a in country_auths[country]} 
        author_score = sort_dict(author_score)
        auths = np.array(list(author_score.keys()))
        scores = np.array(list(author_score.values()))
        mask = scores > 0
        auths = auths[mask]
        scores = scores[mask]
        if len(auths) == 0:
            continue
        city_list = list(item_id_dict(auths, s.author_cities).keys())
        idx = density_clustering(auths, s.author_pmids, density=scores,
                                 mask=[s.author_cities, s.author_affs])
        team_city = {}
        team_authors = {}
        for i in range(1,idx.max()+1):
            mask = idx == i
            team_authors[i] = list(auths[mask])
            team_city[i] = list(item_id_dict(auths[mask], s.author_cities).keys())[0]
        # Assign team data
        team_id = max(list(s.team_authors.keys())) if len(s.team_authors) > 0 else 0
        for city in city_list:
            city_teams = []
            for i, auths in team_authors.items():
                if team_city[i] == city:
                    team_id += 1
                    city_teams.append(team_id)
                    s.team_city[team_id] = city
                    s.team_country[team_id] = country
                    s.team_authors[team_id] = auths
                    s.team_affs[team_id] = list(item_id_dict(auths, s.author_affs).keys())
                    pmids = list(item_id_dict(auths, s.author_pmids).keys())
                    s.team_pmids[team_id] = sort_pmids_by_year(s, pmids)
            s.city_teams[(country, city)] = city_teams
    """ Collect word forms """
    texts = list(s.pmid_affs.values())
    texts = list(itertools.chain(*texts))
    texts = [" ".join(t.split()[:5]) for t in texts]
    texts = texts + list(s.pmid_title.values())
    texts = texts + list(s.pmid_abstract.values())
    word_forms = "wordform.csv" if extended else None
    lemmatizer = WordLemmatizer(texts, word_forms=word_forms, extended=extended)
    s.word_forms = lemmatizer.word_forms
    """ Cluster keywords """
    s.team_keywords = {t: [] for t in s.team_authors}
    """ Keywords based on author affiliations """
    texts = list(s.pmid_affs.values())
    texts = list(itertools.chain(*texts))
    texts = [" ".join(t.split()[:5]) for t in texts]
    kwp = KeyWordParser(texts, word_forms=s.word_forms)
    for i, affs in tqdm(s.team_affs.items(), desc="Keywords from affiliations"):
        texts = [" ".join(aff.split()[:5]) for aff in affs]
        s.team_keywords[i] = s.team_keywords[i] + kwp.keywords(texts)
    """ Keywords based on paper titles """
    texts = list(s.pmid_title.values())
    kwp = KeyWordParser(texts, word_forms=s.word_forms)
    for i, pmids in tqdm(s.team_pmids.items(), desc="Keywords from titles"):
        texts = [s.pmid_title[p] for p in pmids]
        s.team_keywords[i] = s.team_keywords[i] + kwp.keywords(texts, nmax=10)
    """ Keywords based on paper titles """
    texts = list(s.pmid_abstract.values())
    kwp = KeyWordParser(texts, word_forms=s.word_forms)
    for i, pmids in tqdm(s.team_pmids.items(), desc="Keywords from abstracts"):
        texts = [s.pmid_abstract[p] for p in pmids]
        s.team_keywords[i] = s.team_keywords[i] + kwp.keywords(texts, nmax=10)
        s.team_keywords[i]  =list(np.unique(s.team_keywords[i]))
    """ Cluster topics by keywords """
    s.pmid_keywords = {p: [] for p in s.pmids}
    keyword_clustering(s, nmaxclust=nmaxclust)
    return


def team_summary(s, t, pmids=None, keywords=None, min_year=0,
                 max_affs=3, max_authors=5, max_pmids=50, show_emails=True):
    """"
    Utility function to make a team summary in html format
    """
    if pmids is None:
        pmids = s.team_pmids[t]
    pmids = np.array(pmids)
    years = np.array([s.pmid_year[p] for p in pmids])
    mask = years >= min_year
    pmids = pmids[mask]
    summary = ""
    if len(pmids) == 0:
        return summary
    loc = geoutils.country_code_to_name(s.team_country[t])
    if loc is None:
        loc = geoutils.state_code_to_name(s.team_country[t])
    loc = f"[{s.team_city[t]}, {loc}]"
    for i, aff in enumerate(s.team_affs[t][:max_affs]):
        aff_, _ = find_keywords(s, aff, s.top_keywords)
        if i == 0:
            summary = summary + f"<i>{aff_} {loc}</i><br>\n"
        else:
            summary = summary + f"<i>{aff_}</i><br>\n"            
    for i, pmid in enumerate(pmids):
        year = s.pmid_year[pmid]
        if i < max_pmids and year >= min_year:
            title, _ = find_keywords(s, s.pmid_title[pmid], s.top_keywords)
            #title = s.pmid_title[pmid]
            link = f"<a href='https://pubmed.ncbi.nlm.nih.gov/{pmid}/'" \
            f"target='_blank' rel='noopener noreferrer'>{pmid}</a>"
            info = f"{s.pmid_year[pmid]} [PMID {link}] {title}"
            summary = summary + f"{info}<br>\n"
    count_author = 0
    for a in s.team_authors[t]:
        email = s.author_email[a]
        if s.author_score[a] > 0 and len(email) > 0 and count_author < max_authors:
            if show_emails:
                summary = summary + f"{s.author_name[a]}   [{email[0]}]<br>\n"
            else:
                summary = summary + f"{s.author_name[a]}<br>\n"
            count_author += 1
    for a in s.team_authors[t]:
        email = s.author_email[a]
        if s.author_score[a] > 0 and len(email) == 0 and count_author < max_authors:
            summary = summary + f"{s.author_name[a]}<br>\n"
            count_author += 1
    summary = summary + "<br>\n<br>\n"
    return summary





def topic_summary(s, topic, min_year=0):
    """
    Utility function to calculate topic summary
    """
    assert isinstance(topic, (list, int, np.int64))
    topic_id = 0
    keywords = topic
    if isinstance(topic, (int, np.int64)):
        if topic in s.topic_keywords:
            topic_id = topic
            keywords = s.topic_keywords[topic]
        else:
            msg = f"Wrong topic id {topic}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return
    """ Update PMID keywords if needed """
    new_keywords = [k for k in keywords if k not in s.top_keywords]
    if len(new_keywords) > 0:
        update_pmid_keywords(s, new_keywords)
    """ Assign topic pmids """
    topic_pmids = [s.keyword_pmids[k] for k in keywords if k in s.keyword_pmids]
    topic_pmids = list(itertools.chain(*topic_pmids))
    topic_pmids = sort_pmids_by_year(s, topic_pmids)
    """ Assign topic teams """
    topic_teams = {}
    for t, pmids in s.team_pmids.items():
        topic_teams[t] = [p for p in pmids if p in topic_pmids and s.pmid_year[p] >= min_year]
    topic_teams = sort_dict(topic_teams)
    duplicates = []
    for t, pmids in topic_teams.items():
        pmids_ = [p for p in pmids if p not in duplicates]
        duplicates = duplicates + pmids_
        topic_teams[t] = pmids_
    topic_teams = sort_dict(topic_teams)
    topic_keywords = []
    for t in topic_teams:
        for key in s.team_keywords[t]:
            if key in keywords:
                topic_keywords.append(key)
    return topic_teams, topic_pmids, topic_keywords


def topic_html(s, topic, folder="pubmed_summary", min_year=0):
    """"
    Save html topic summary

    Parameters
    ----------
    s : class object
        PubMedScraper() instance object
    topic : int
        Topic id (starts from zero)
    folder : str, default "pubmed_summary"
        Output folder
    min_year : int, default 0
        Min year cutoff

    """
    topic_teams, topic_pmids, topic_keywords = topic_summary(s, topic, min_year)
    if not os.path.exists(folder):
        os.makedirs(folder)
    fname = "_".join(topic_keywords[:3])
    fname = f"{folder}/{topic:02d}_{fname}.html"
    f = open(fname, "w+")
    for t, pmids in topic_teams.items():
        pmids_ = [p for p in pmids if p in topic_pmids]
        summary = team_summary(s, t, pmids=pmids_, keywords=topic_keywords, 
            min_year=min_year, max_affs=1, max_authors=2, show_emails=False)
        f.write(summary)
    f.close()
    return


def topic_wordcloud(s, topic, min_year=0):
    """"
    Show wordcloud of topic keywords

    Parameters
    ----------
    s : class object
        PubMedScraper() instance object
    topic : int
        Topic id (starts from zero)
    min_year : int, default 0
        Min year cutoff

    """
    _, _, topic_keywords = topic_summary(s, topic, min_year)
    print(f"Topic #{topic} keywords cloud")
    keywords_wordcloud(topic_keywords)
    return








