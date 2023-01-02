#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import string
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from collections import Counter
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import requests
import asyncio
import aiohttp
import unidecode
import re

"""
PubMedScraper() implements methods:

- to asynchronously scrape search resuklts pages:
    - scrape_page() - scrape one page (200 records)
    - scrape_pmids() - scrape all results, automatically switch to 
    year-by-year search if PubMed limit of 10_000 records exceeded

- to read and parse year, title, author names, affiliations:
    - species_filter() to skip results for plants or animals
    - parse_pmid_data()
"""




class PubMedScraper():
    """
    Class to asynchronously scrape PubMed search results web-pages

    Parameters
    ----------
    search_terms : str
        Search terms, e.g. "hydrothermal vents"
    semaphore : int, default 10
        Limit for number of requests per time: Higher is faster, but if too high, 
        PubMed will return error code (PubMedScraper() will print warning message)
    animal : bool, default True
        When all data downloaded - flag to parse or not records containing animal keywords
    plant : bool, default True
        When all data downloaded - flag to parse or not records containing plant keywords

    Attributes
    ----------
    pmids : ndarray
        List of PubMed id
    data : DataFrame
        DataFrame with year, pmid, and title of all search results
    pmid_affs : dict
        Dictionary of affiliations for PubMed id
    pmid_year : dict
        Dictionary of year for PubMed id
    pmid_title : dict
        Dictionary of title for PubMed id
    pmid_abstract : dict
        Dictionary of abstract for PubMed id
    pmid_authors : dict
        Dictionary of author names for PubMed id
    author_name : dict
        Dictionary of full author name for author name id
    author_affs : dict
        Dictionary of affiliations for author name id
    author_pmids : dict
        Dictionary of PubMed ids for author name id
    author_email : dict
        Dictionary of email for author name id
    author_score : dict
        Dictionary of author score for author name id
        Score = Num first author + Num last author + 1 if has emails

    Example
    -------
    >>> s = PubMedScraper("hydrothermal vents")
    >>> s.data.head()

    """
    def __init__(self, search_terms, semaphore=10, animal=True, plant=True):
        self.folder = "_".join(search_terms.lower().split()) + "/"
        self.download = self.folder + "/download/"
        self.search_terms = search_terms
        self.semaphore = semaphore
        self.animal = animal
        self.plant = plant
        # Look up for folder "_".join(search_terms.lower()split()) + "/download/"
        # If folder does not exist - scrape PubMed website
        if not os.path.exists(self.download):
            print("SCRAPING WEBSITE")
            if not os.path.exists(self.download):
                os.makedirs(self.download)
            asyncio.run(self.scrape_pmids())
            self.data = pd.DataFrame(index=pd.Index(self.pmids, name='pmid'))
        # If folder exists (created right now, or before) - parse pmids .txt files
        if os.path.exists(self.download):
            print("READING FOLDER")
            self.pmids = np.array([f.split(".")[0] for f in os.listdir(self.download)])
            self.data = self.parse_pmid_data()


    async def scrape_pmids(self):
        """
        Scrape all results, automatically switch to 
        year-by-year search if PubMed limit of 10_000 records exceeded
        """
        def extract_timeline(soup):
            t = soup.find("table", {"id": "timeline-table"})
            t = pd.read_html(str(t), index_col="Year")[0]
            n = t["Number of Results"].values
            n = n // 200 + (n % 200 > 0)
            t = dict(zip(t.index.values, n))
            return t

        def extract_num_pages(soup):
            e = soup.find("span", {"class": "total-pages"})
            npage = int(e.get_text().replace(",","")) if e is not None else 0
            return npage
            
        # Generate primary search url
        url = f"https://pubmed.ncbi.nlm.nih.gov/?sort=date&size=200"
        url = f"{url}&term={'+'.join(self.search_terms.split())}"
        # Parse number of pages
        soup = BeautifulSoup(requests.get(url).content, features="html.parser")
        npage = extract_num_pages(soup)
        # Generate all output urls to parse search results page-by-page
        if npage <= 50:
            urls = [f"{url}&format=pubmed&page={i+1}" for i in range(min(npage,50))]
        else:
            # If "50 pages" limit exceeded - try to scrape timeline year by year
            timeline = extract_timeline(soup)
            urls = []
            for y, n in timeline.items():
                for i in range(n):
                    urls.append(f"{url}&filter=years.{y}-{y}&format=pubmed&page={i+1}")
        # Asynchronously fetch pmids data from list of urls
        semaphore = asyncio.BoundedSemaphore(self.semaphore)
        tasks = []
        for url in urls:
            tasks.append(self.scrape_page(url, semaphore))
        pmids = await asyncio.gather(*tasks)
        pmids = list(itertools.chain(*pmids))
        self.pmids = np.unique(np.array(pmids))[::-1]
        print("DEBUG", len(self.pmids))
        return


    async def scrape_page(self, url, semaphore):
        """
        Scrape one page (200 records)
        """
        def split_files(text):
            pmids = []
            for i, line in enumerate(text):
                if line.find("PMID- ") == 0:
                    pmid = line[6:]
                    pmids.append(pmid)
                    if i > 0:
                        fout.close()
                    fout = open(f"{self.download}/{pmid}.txt", "w+")
                fout.write(f"{line}\n")
            fout.close()
            return pmids
            
        header = {"User-Agent": str(UserAgent().random)}
        async with aiohttp.ClientSession(headers=header) as session:
            async with semaphore, session.get(url) as response:
                data = await response.text()
                # HTTP status code should be 200 (OK)
                # If not - it's likely there were too many requests
                # try to set lower semaphore value
                if response.status != 200:
                    msg = f"Try to set lower semaphore: too many requests," + \
                    f"status {response.status} for {url}"
                    warnings.warn(msg, UserWarning, stacklevel=2)
                # Parse pmids and descriptions
                soup = BeautifulSoup(data, features="html.parser")
                e = soup.find("pre", {"class": "search-results-chunk"})
                pmids = e.get_text().split("\r\n") if e is not None else []
                if len(pmids) > 0:
                    pmids = split_files(pmids)
        return pmids


    def parse_pmid_data(self):
        """
        Read and parse year, title, author names, affiliations
        """
        self.pmid_affs = {}
        self.pmid_year = {}
        self.pmid_title = {}
        self.pmid_review = {}
        self.pmid_authors = {}
        self.pmid_abstract = {}
        self.author_name = {}
        self.author_affs = {}
        self.author_pmids = {}
        self.author_email = {}
        self.author_score = {}
        for pmid in tqdm(self.pmids, desc="Parse pmid data"):
            full_author = ""
            author = ""
            review = 0
            year = 0
            title = ""
            abstract = ""
            authors = []
            mesh = ""
            # Read and parse pmids .txt one by one
            desc = read_pmid_txt(f"{self.download}/{pmid}.txt")
            if not self.species_filter(desc):
                continue
            for line in desc:
                w = line.split()
                # Info line should have >= 3 words: tag, dash and info
                if len(w) < 3:
                    continue
                # Date of publication
                if w[0] == "DP":
                    year = w[2][:4]
                    if len(year) == 4 and year.isdigit():
                        year = int(year)
                # Title of publication
                if w[0] == "TI":
                    title = " ".join(w[2:])
                # Abstarct of publication
                if w[0] == "AB":
                    abstract = " ".join(w[2:])
                # MESH terms
                if w[0] == "MH":
                    mesh = mesh + ", " + " ".join(w[2:])
                # Publication type
                if w[0] == "PT" and line.lower().find("review") > 0:
                    review = 1
                # Full author name
                if w[0] == "FAU":
                    full_author = " ".join(w[2:])
                    full_author = " ".join(full_author.split(", ")[::-1])
                # Standard author name (use this as author id)
                if w[0] == "AU":
                    author = " ".join(" ".join(w[2:]).split("-"))
                    author = unidecode.unidecode(author)
                # Author description: affiliations, emails
                if w[0] == "AD" and author != "":
                    aff = " ".join(w[2:])
                    affs, email = parse_aff(aff)
                    update_dict(self.author_affs, author, affs)
                    update_dict(self.author_name, author, full_author)
                    update_dict(self.author_pmids, author, pmid)
                    update_dict(self.author_email, author, email)
                    authors.append(author)
                    full_author = ""
                    author = ""
            # Drop publications without year or title
            if year > 0 and len(title) > 0:
                affs = [self.author_affs[a] for a in authors]
                affs = list(np.unique(np.array(list(itertools.chain(*affs)))))
                self.pmid_affs[pmid] = affs
                self.pmid_year[pmid] = year
                self.pmid_title[pmid] = title
                self.pmid_review[pmid] = review
                self.pmid_authors[pmid] = authors
                self.pmid_abstract[pmid] = abstract
                # Store first and last author for each pmid
                if len(authors) > 0:
                    update_dict(self.author_score, authors[0], 1)
                    update_dict(self.author_score, authors[-1], 1)
        self.pmids = list(self.pmid_year.keys())        
        # Calculate author score + N first author + N last author + has email
        # Also select the most frequest email and full name with many non-unicode chars
        for a in self.author_name:
            if len(self.author_email[a]) > 0:
                self.author_email[a] = [Counter(self.author_email[a]).most_common(1)[0][0]]
            self.author_name[a] = select_author_name(self.author_name[a])
            score = int(len(self.author_pmids[a])) + int(len(self.author_email[a]) > 0)
            if a in self.author_score:
                score += sum(self.author_score[a])
            self.author_score[a] = score
        # Build and output pmid data DataFrame
        d = {"year": self.pmid_year, "title": self.pmid_title,
             "authors": {p: ",".join(self.pmid_authors[p]) for p in self.pmid_authors}}
        d = pd.DataFrame.from_dict(d)
        d.index.name = "pmid"
        d.sort_values(by="year", inplace=True, ascending=False)
        self.pmids = d.index.values
        return d


    def species_filter(self, desc):
        """
        Filter to skip results for plants or animals
        """
        if self.animal and self.plant:
            return True
        text = []
        for line in desc:
            w = line.split()
            if len(w) >= 3 and w[0] in ["TI", "AB", "AD", "MH"]:
                text.append(" ".join(w[2:]))
        text = " ".join(text).lower()
        if not self.animal:
            for w in ["animal", "bacteria", "rabbit", "mice", "mouse", "murine",
                      " rat ", " rat,", " rat.", " rats ", " rats,", " rats."]:
                if text.find(w) >= 0:
                    return False
        if not self.plant and text.find("plant") >= 0:
            return False
        return True


def read_pmid_txt(fname):
    """
    Utility function to read PubMed format file 
    and return lines with one line per tag
    """
    desc = ""
    try:
        f = open(fname, "r")
        for line in f:
            if len(desc) == 0:
                desc = line[:-1]
            elif len(line) < 6 or line[:6] != "      ":
                desc = desc + "\n" + line[:-1]
            else:
                desc = desc + line[6:-1]
        f.close()
        desc = desc.split("\n")
    except:
        pass
    return desc


def update_dict(dct, key, val):
    """
    Utility function to update dictionary
    """
    known_vals = []
    if key in dct:
        known_vals = dct[key]
    if isinstance(val, list):
        for v in val:
            known_vals.append(v)
    else:
        known_vals.append(val)
    dct[key] = known_vals
    return 


def count_nonlatin(text):
    """
    Utility function to count non-latin symbols
    """
    alphabet = string.ascii_lowercase + string.ascii_uppercase + " "
    count = 0
    for ch in text:
        if ch not in alphabet:
            count += 1
    return count


def select_author_name(names):
    """
    Utility function to select the most complete full author name
    """
    if len(names) == 1:
        return names[0]
    x = np.array(names)
    n = np.array([len(x_) for x_ in x])
    mask = n == n.max()
    x = x[mask]
    if len(x) > 1:
        n = np.array([count_nonlatin(x_) for x_ in x])
        mask = n == n.max()
        x = x[mask]
    return x[0]


def parse_text_punctuation(text):
    """
    Utility function to parse punctuation in text
    """
    # Change affs delimiter "[*]" to ";"
    t = re.sub(r"\[\d+\]", ". ;", text)
    # Remove technical terms
    ex = [r"1\]", r"\[corrected\]", r"\[", r"\]", "<", ">", ":", 
          "email", "Email", r"e\-mail", r"E\-mail", r"e\-Mail", r"E\-Mail",
          "emails", "Emails", r"e\-mails", r"E\-mails", r"e\-Mails", r"E\-Mails",
          "Electronic address", "electronic address"]
    for e in ex:
        t = re.sub(e, "", t)
    # Remove initials
    if t.find("(") >=0 and t.find(")") >= 0:
        t = re.sub(r"(\()?([A-Z]\.){2,}(,)?(\))?", "", t)
    # Remove spaces before punctuations
    t = re.sub(r'\s+([?.,;:!"])', r'\1', t)
    return t


def parse_text_emails(text):
    """
    Utility function to parse emails from text
    """
    email = text
    if email[-1] in [".", ",", ";", ")", "-"]:
        email = email[:-1]
    if email[0] == "(":
        email = email[1:]
    return email


def parse_text_beginning(words):
    """
    Utility function to find beginning word in text (exclude numbering)
    """
    ex = ["and", "from", "iii", "ii.", "iv.", "vi.", "vii"]
    words = np.array(words)
    length = np.array([len(w) for w in words])
    mask = length > 2
    for i in range(len(words)):
        if length[i] in [3,4]:
            mask[i] = not any([words[i].lower().find(e) == 0 for e in ex])
    if np.sum(mask) > 0:
        i = np.arange(len(words))[mask][0]
        words = words[i:]
    else:
        words = []
    return words    


def parse_aff(aff):
    """
    Utility function to parse ";"-separated author affiliations from text
    """
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    text = parse_text_punctuation(aff)
    emails = []
    affs = []
    for t in text.split(";"):
        if len(t) < 5:
            continue
        words = t.split()
        nword = len(words)
        for i, w in enumerate(words):
            if w.find("ISNI") == 0 or w.find("GRID") == 0 or w.find("ORCID") == 0 :
                nword = min(i, nword)
            if w.find("@") > 0:
                nword = min(i, nword)
                emails.append(parse_text_emails(w))
        words = parse_text_beginning(words[:nword])
        if len(words) > 0:
            aff = " ".join(words[:nword])
            aff = first_letter(aff)
            if len(aff) > 0:
                affs.append(aff)
    return affs, emails


def first_letter(text):
    """
    Utility function to find beginning alphabetical symbol (exclude numbering)
    """
    try:
        i = re.search(r"[a-zA-Z]", text).start()
        if i is not None:
            return text[i:]
    except:
        return ""





