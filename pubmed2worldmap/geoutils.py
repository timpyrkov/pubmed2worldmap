#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import unidecode
import difflib
import json
import re
import pylab as plt
import warnings

import geopandas

import geonamescache
gc = geonamescache.GeonamesCache()

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="get_state_for_city")


"""
Features:

- Get "geonamescache" city, country, and US state names and codes.

- Sort country list GDP per capita (GDP from "geopandas")

- Use "geopy" to assign US state to city based on (latitude, longitude)

- Convert all codes to match "geopandas" maps:
    - 3-symbol iso3 code for country
    - 2-symbol code for US state
"""




def plot_worldmap(data, cmap="cividis_r", caption="", us_shapefile=None, vmin=None, vmax=None):
    """
    Plot data by country (and US state) on world map

    Parameters
    ----------
    data : dict
        Dictionary of country/US state code to values
    cmap : str, default "cividis_r"
        Matplotlib colormap name
    caption : str or None, default None
        Caption to show below world map
    us_shapefile : str or None, default None
        path to downloaded US states ".shp" shape file
    vmin : float or None, default None
        Min value, needed to sync colormap for world and US maps
    vmax : float or None, default None
        Max value, needed to sync colormap for world and US maps

    """
    world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    world = world[world.name!="Antarctica"]
    world["data"] = world["iso_a3"].map(data).fillna(0)
    if us_shapefile is not None:
        try:
            states = geopandas.read_file(us_shapefile)
            states["data"] = states["State_Code"].map(data).fillna(0)
            # If data for US states was privided - set USA to NaN in
            # the low resolution world map to show up the Greate lakes
            if states["data"].sum() > 0:
                x = world["data"].values
                x[world["iso_a3"] == "USA"] = np.nan
                world["data"] = x
        except:
            data_ = data
            data_usa = sum([data[d] for d in data if len(d) == 2])
            data_["USA"] = data["USA"] + data_usa if "USA" in data else data_usa
            world["data"] = world["iso_a3"].map(data_).fillna(0)
            msg = f"Failed to load US states map from {us_shapefile}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            us_shapefile = None
    cmap = sqrtcmap(cmap, reverse=False)
    plt.figure(figsize=(18,6), facecolor='w')
    ax = plt.gca()
    x = world["data"].values
    vmin = vmin if vmin is not None else np.nanmin(x)
    vmax = vmax if vmax is not None else np.quantile(x[np.isfinite(x)], 0.99)
    world.plot(column="data", ax=ax, legend=True, cmap=cmap, vmin=vmin, vmax=vmax,
        legend_kwds={"label": caption, "pad": 0.05, "orientation": "horizontal", "shrink": 0.3})
    if us_shapefile is not None:
        states.plot(column="data", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, figsize=None)
    ax = plt.gca()
    for e in ["top", "bottom", "right", "left"]:
        ax.spines[e].set_color("#00000000")
    ax.set_aspect(1.2)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(-60,100)
    fig = plt.gcf()
    fig.set_size_inches((32,8))
    plt.tight_layout()
    plt.show()
    return


class GeoTextParser():
    """
    Class to find city/state/country names in text

    Notes
    -----
    If geonamescache fails to download city data, run:
    jupyter notebook --NotebookApp.iopub_data_rate_limit=1e10

    Parameters
    ----------
    path : str
        Path to store city data for fast reload

    Attributes
    ----------
    country_code : dict
        Dictionary of country name to iso3 3-symbol code
    extended_code : dict
        Dictionary of alternative country name to iso3 3-symbol code
    state_code : dict
        Dictionary of US state name to 2-symbol code
    city_data : DataFrame
        City data
    city_idx : ndarray
        City index
    city_name : ndarray
        City name
    city_country : ndarray
        Country iso3 code for each city
    city_state : ndarray
        US state 2-letter code or empty (nan or null)
    city_names : nested list
        Lists of alternative city names 

    Example
    -------
    >>> g = geoutils.GeoTextParser()
    >>> countries, cities = g.find_codes(text)

    """
    def __init__(self, path=None):
        # Country data
        self.country_code = country_name_to_code()
        # Extended data
        self.extended_code = {"United States": "USA", "United states": "USA", 
                              "united states": "USA", "USA": "USA", "U.S.": "USA", 
                              "U.S.A.": "USA", "UK": "GBR", "U.K.": "GBR"}
        # State data
        self.state_code = state_name_to_code()
        # City data
        self.city_data = get_cities(path)
        self.city_idx = np.arange(self.city_data.shape[0])
        self.city_name = self.city_data["name"].values.astype(str)
        self.city_country = self.city_data["countrycode"].values.astype(str)
        self.city_state = self.city_data["statecode"].values.astype(str)
        self.city_names = self.city_data["alternatenames"].values


    @staticmethod
    def find_full_name(text, name):
        """
        For country/state codes - get city indices in array of cities
        """
        j0 = text.title().find(name.title())
        if j0 < 0:
            return False
        j1 = j0 + len(name)
        return j1 >= len(text) or not text[j1:][0].isalpha()


    def city_idx_slice_by_code(self, code):
        """
        Get all city indices in array of cities for a country/US state code
        """
        mask = self.city_country == code
        if len(code) == 2:
            mask = self.city_state == code
        return self.city_idx[mask]

    
    def find_city_idx(self, text, code=None, len_min=4):
        """
        Find city name in text and output city index if found else None
        If codes given - search cities for country/states else for the whole world
        """
        i0 = None
        idx = self.city_idx
        if code is not None:
            len_min -= 1
            idx = self.city_idx_slice_by_code(code)
        t = unidecode.unidecode(text).title()
        # 1. Find exact match to the default city name
        for i in idx:
            name = self.city_name[i]
            if len(name) >= len_min and self.find_full_name(t, name):
                return i
        # 2. Find exact match to an alternative city name
        for i in idx:
            names = self.city_names[i]
            for name in names:
                if len(name) >= len_min and self.find_full_name(t, name):
                    return i
        # 3. Find close match to the default city name
        words = (re.sub(r"[,.]", " ", t))
        words = (re.sub(r'"', " ", words)).split()
        for i in idx:
            name = self.city_name[i]
            if len(name) >= len_min and len(difflib.get_close_matches(name, words)) > 0:
                    return i
        return i0

    
    def confirm_codes(self, text, codes):
        """
        Output list of confirmed country/state code by finding a city name in text
        """
        confirmed_codes = []
        confirmed_cities = []
        for code in codes:
            i = self.find_city_idx(text, code)
            if i is not None:
                confirmed_codes.append(code)
                confirmed_cities.append(self.city_name[i])
        return confirmed_codes, confirmed_cities

    
    def find_counries_cities(self, text):
        """
        Find city  and country (US state) names in text
        """
        # First look for countries/states
        iso_codes = []
        iso3_codes = []
        for k, v in self.state_code.items():
            words = (re.sub(r"[,.]", " ", text))
            words = (re.sub(r'"', " ", words)).split()
            if self.find_full_name(text, k) or v in words:
                iso_codes.append(v)
        for k, v in self.extended_code.items():
            if self.find_full_name(text, k) and v not in iso3_codes:
                iso3_codes.append(v)
        for k, v in self.country_code.items():
            if self.find_full_name(text, k) and v not in iso3_codes:
                iso3_codes.append(v)
        if "USA" in iso3_codes and len(iso_codes) == 0:
            iso_codes = list(self.state_code.values())
        codes = iso_codes + [c for c in iso3_codes if c != "USA"]
        # For found coutries/states - try to confirm by finding a city
        codes, cities = self.confirm_codes(text, codes)
        # If no country/state was found or confirmed - try to find a city
        if len(codes) == 0:
            i = self.find_city_idx(text)
            if i is not None:
                cities = [self.city_name[i]]
                codes = [self.city_state[i]]
                if len(codes[0]) != 2:
                    codes = [self.city_country[i]]
        return codes, cities












def store_locally(func):
    """
    Decorator for get_countries() and get_cities()
    to save/load geonamescache data locally if path provided
    
    Parameters
    ----------
    func : function obj
        get_countries() or get_cities()

    Returns
    -------
    decorator : function object
        Function to save or load pre-saved dataframe

    Example
    -------
    >>> from pyutils.animation import save_frame, make_gif
    >>> @store_locally
    >>> def city_dataframe():
    >>>     data = gc.get_cities()
    >>>     df = pd.DataFrame.from_dict(data, orient="index")
    >>>     return df
    >>>
    >>> df = city_dataframe("geodata/city.csv")

    """
    def wrapper(*args, **kwargs):
        path = None
        if  "path" in kwargs:
             path = kwargs["path"]
        elif len(args) > 0 and isinstance(args[0], str):
            path = args[0]
        if os.path.exists(str(path)):
            df = pd.read_csv(path, delimiter=";", index_col=0)
            if "alternatenames" in df.columns:
                alt_names = [json.loads(s) for s in df["alternatenames"].values]
                df["alternatenames"] = np.array(alt_names, dtype=object)
        else:
            df = func(*args, **kwargs)
            if path is not None:
                folder = os.path.expanduser(os.path.dirname(path))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                if "alternatenames" in df.columns:
                    df["alternatenames"] = [json.dumps(s) for s in df["alternatenames"].values]

                df.to_csv(path, sep=";")
        return df
    return wrapper


@store_locally
def get_countries(*args, sort_by_gdp=True):
    """
    Cet DataFrame of geonamescache country data
    """
    data = gc.get_countries()
    df = pd.DataFrame.from_dict(data, orient="index")
    if sort_by_gdp:
        dct = country_gdp_per_capita()
        df["gdp"] = np.vectorize(dct.get)(df["iso3"].values, np.nan)
        mask = np.isfinite(df["gdp"])
        df = pd.concat([
            df[mask].sort_values(by="gdp", ascending=False),
            df[~mask].sort_values(by="population", ascending=False),
        ])
    return df


@store_locally
def get_cities(*args, sort_by_gdp=True):
    """
    Cet DataFrame of geonamescache city data
    """
    data = gc.get_cities()
    df = pd.DataFrame.from_dict(data, orient="index")
    # Change country 2-letter codes to 3-letter iso3 codes
    # for compatibility with geopandas
    dct = country_code_to_iso3()
    df["countrycode"] = np.vectorize(dct.get)(df["countrycode"])
    # Add US state 2-letter codes
    dct = hardcoded_us_state_for_cities()
    df["statecode"] = np.vectorize(dct.get)(df["geonameid"].values, "")
    # Fix default city names
    dct = {"New York City": "New York", "University": "University CDP"}
    df["name"] = [dct[x] if x in dct else x.title() for x in df["name"].values]
    # Deunicode alternatenames and exclude too short names
    df = ununicode_alt_city_names(df)
    # Sort cities by population
    df = df.sort_values(by="population", ascending=False)
    # Optionally sort by country GDP per capita
    if sort_by_gdp:
        codes = iso_codes()
        df = [df[df["countrycode"] == c] for c in codes]
        df = pd.concat(df)
    return df


def ununicode_alt_city_names(df):
    """
    Exclude alternative city names not containing latin symbols, then un-unicode
    """
    def is_latin(x):
        return len(re.sub("[^a-zA-Z]", "", x)) > 0
    
    names = df["name"].values
    dct = dict(zip(names, df["alternatenames"].values))
    exclude_names = ["University", "Hospital", "Clinic", "College"]
    for i, name in enumerate(names):
        alt_names = []
        for x in dct[name]:
            if is_latin(x):
                x_ = unidecode.unidecode(x).title()
                if x_ != name and len(x_) >= len(name) and x_ not in exclude_names:
                    alt_names.append(x_)
        dct[name] = [name.title()] + list(set(alt_names))
    df["alternatenames"] = np.array([dct[n_] for n_ in names], dtype=object)
    return df


def iso_codes():
    """
    Get list of all US states (2-letter) and countries (3-letter) codes
    """
    codes = list(gc.get_us_states().keys())
    codes = codes + list(country_gdp_per_capita().keys())
    codes = codes + list(set(country_population().keys()) - set(codes))
    return codes


def geonamescache_to_dictionary(func):
    """
    Decorator to get value by key if key provided,
    otherwise return dictionary
    """
    def wrapper(*args, **kwargs):
        dct = func()
        if len(args) != 0:
            if isinstance(args[0], (list, np.ndarray)):
                x = np.asarray(args[0])
                x = list(np.vectorize(dct.get)(x))
            else:
                x = dct[args[0]] if args[0] in dct else None
            return x
        return dct
    return wrapper


@geonamescache_to_dictionary
def country_gdp_per_capita():
    """
    Get dictionary of GDP per capita for country codes (based on "geopandas")
    """
    world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    world = world[world.name!="Antarctica"]
    world["gdp_per_capita"] = 1e+6 * world["gdp_md_est"] / world["pop_est"]
    dct = world.set_index("iso_a3")["gdp_per_capita"].to_dict()
    dct = dict(sorted(dct.items(), key = lambda item: item[1], reverse=True))
    return dct


@geonamescache_to_dictionary
def country_population():
    """
    Get dictionary of population for country codes (based on "geonamescache")
    """
    data = gc.get_countries()
    dct = {data[d]["iso3"]: data[d]["population"] for d in data}
    dct = dict(sorted(dct.items(), key = lambda item: item[1], reverse=True))
    return dct


@geonamescache_to_dictionary
def country_code_to_iso3():
    """
    Get dictionary of 3-symbol iso3 codes for 2-symbol country codes (based on "geonamescache")
    """
    data = gc.get_countries()
    dct = {data[d]["iso"]: data[d]["iso3"] for d in data}
    return dct


@geonamescache_to_dictionary
def country_code_to_name():
    """
    Get dictionary of names for country codes (based on "geonamescache")
    """
    data = gc.get_countries()
    dct = {data[d]["iso3"]: data[d]["name"] for d in data}
    return dct


@geonamescache_to_dictionary
def country_name_to_code():
    """
    Get dictionary of codes for country names (based on "geonamescache")
    """
    data = gc.get_countries()
    dct = {data[d]["name"]: data[d]["iso3"] for d in data}
    return dct


@geonamescache_to_dictionary
def state_code_to_name():
    """
    Get dictionary of names for US state codes (based on "geonamescache")
    """
    data = gc.get_us_states()
    dct = {data[d]["code"]: data[d]["name"] for d in data}
    return dct


@geonamescache_to_dictionary
def state_name_to_code():
    """
    Get dictionary of codes for US state names (based on "geonamescache")
    """
    data = gc.get_us_states()
    dct = {data[d]["name"]: data[d]["code"] for d in data}
    return dct









def sqrtcmap(name, reverse=False):
    """
    Make nice pastel-toned colors from matplotlib cmap
    """
    sqrtscale = np.power(np.linspace(0,1,11)[2:-2], 0.5)
    colors = plt.cm.get_cmap(name)(sqrtscale)
    if reverse:
        colors = colors[::-1]
    cmap = LinearSegmentedColormap.from_list('sqrtscale', colors)
    return cmap




def fetch_us_state(latitude, longitude):
    """
    Fetch US state name based on city (latitude, longitude) and "geopy"
    """
    try:
        location = geolocator.reverse(f"{latitude}, {longitude}")
        state = location.raw["address"]["state"]
    except:
        state = ""
    return state


def fetch_us_state_for_cities():
    """
    Fetch US state code based on cities (latitude, longitude) and "geopy"
    Works slowly, use hardcoded_us_state_for_cities() instead
    """
    city_data = gc.get_cities()
    df = pd.DataFrame.from_dict(city_data, orient="index")
    mask = df["countrycode"] == "US"
    dct = df[["latitude", "longitude"]][mask].T.to_dict("list")
    for d in tqdm(dct, desc="Get geopy US state for city"):
        state = fetch_us_state(*dct[d])
        dct[d] = geoutils.state_name_to_code(state)
    return dct


def hardcoded_us_state_for_cities():
    """
    Get US state code based on cities (latitude, longitude) and "geopy"
    """
    dct = {
        4046704: "VA", 4048023: "AL", 4048662: "KY", 4049979: "AL",
        4054378: "AL", 4057835: "AL", 4058219: "AL", 4058553: "AL",
        4059102: "AL", 4059870: "AL", 4060791: "AL", 4061234: "AL",
        4062577: "AL", 4062644: "AL", 4063619: "AL", 4066811: "AL",
        4067927: "AL", 4067994: "AL", 4068446: "AL", 4068590: "AL",
        4074267: "AL", 4076239: "AL", 4076598: "AL", 4076784: "AL",
        4078646: "AL", 4080555: "AL", 4081644: "AL", 4081914: "AL",
        4082569: "AL", 4082866: "AL", 4084796: "AL", 4084888: "AL",
        4089114: "AL", 4092788: "AL", 4093753: "AL", 4094163: "AL",
        4094212: "AL", 4094455: "AL", 4095415: "AL", 4101114: "AR",
        4101241: "AR", 4101260: "AR", 4103448: "AR", 4103957: "AR",
        4106458: "AR", 4109785: "AR", 4110486: "AR", 4111410: "AR",
        4115412: "AR", 4116315: "AR", 4116834: "AR", 4119403: "AR",
        4120792: "AR", 4124112: "AR", 4125388: "AR", 4126226: "AR",
        4128894: "AR", 4129397: "AR", 4130430: "AR", 4131116: "AR",
        4132093: "AR", 4133367: "AR", 4134716: "AR", 4135865: "AR",
        4137457: "DC", 4140463: "DC", 4140963: "DC", 4141363: "DE",
        4142290: "DE", 4143637: "DE", 4143861: "DE", 4145381: "DE",
        4145805: "FL", 4145941: "FL", 4146166: "FL", 4146389: "FL",
        4146429: "FL", 4146723: "FL", 4146901: "FL", 4146934: "FL",
        4147241: "FL", 4147290: "FL", 4148207: "FL", 4148399: "FL",
        4148411: "FL", 4148533: "FL", 4148677: "FL", 4148708: "FL",
        4148757: "FL", 4148803: "FL", 4149077: "FL", 4149269: "FL",
        4149956: "FL", 4149962: "FL", 4150066: "FL", 4150115: "FL",
        4150118: "FL", 4150190: "FL", 4151157: "FL", 4151316: "FL",
        4151352: "FL", 4151440: "FL", 4151455: "FL", 4151460: "FL",
        4151824: "FL", 4151871: "FL", 4151909: "FL", 4151921: "FL",
        4152064: "FL", 4152093: "FL", 4152311: "FL", 4152564: "FL",
        4152574: "FL", 4152772: "FL", 4152820: "FL", 4152872: "FL",
        4152890: "FL", 4152926: "FL", 4153071: "FL", 4153132: "FL",
        4153146: "FL", 4153471: "FL", 4153759: "FL", 4154008: "FL",
        4154031: "FL", 4154047: "FL", 4154205: "FL", 4154280: "FL",
        4154489: "FL", 4154568: "FL", 4154606: "FL", 4155017: "FL",
        4155529: "FL", 4155594: "FL", 4155726: "FL", 4155966: "FL",
        4155995: "FL", 4156018: "FL", 4156042: "FL", 4156091: "FL",
        4156331: "FL", 4156404: "FL", 4156857: "FL", 4156920: "FL",
        4156931: "FL", 4157467: "FL", 4157827: "FL", 4157898: "FL",
        4158476: "FL", 4158482: "FL", 4158865: "FL", 4158928: "FL",
        4159050: "FL", 4159553: "FL", 4159805: "FL", 4159896: "FL",
        4160021: "FL", 4160023: "FL", 4160100: "FL", 4160610: "FL",
        4160705: "FL", 4160711: "FL", 4160812: "FL", 4160822: "FL",
        4160983: "FL", 4161178: "FL", 4161313: "FL", 4161373: "FL",
        4161400: "FL", 4161422: "FL", 4161424: "FL", 4161438: "FL",
        4161461: "FL", 4161534: "FL", 4161580: "FL", 4161616: "FL",
        4161625: "FL", 4161705: "FL", 4161771: "FL", 4161785: "FL",
        4161797: "FL", 4163033: "FL", 4163049: "FL", 4163220: "FL",
        4163388: "FL", 4163407: "FL", 4163918: "FL", 4163971: "FL",
        4164092: "FL", 4164138: "FL", 4164143: "FL", 4164167: "FL",
        4164186: "FL", 4164601: "FL", 4165519: "FL", 4165565: "FL",
        4165637: "FL", 4165869: "FL", 4165913: "FL", 4166066: "FL",
        4166195: "FL", 4166222: "FL", 4166232: "FL", 4166233: "FL",
        4166274: "FL", 4166583: "FL", 4166638: "FL", 4166673: "FL",
        4166776: "FL", 4166805: "FL", 4167003: "FL", 4167147: "FL",
        4167178: "FL", 4167348: "FL", 4167424: "FL", 4167499: "FL",
        4167519: "FL", 4167536: "FL", 4167538: "FL", 4167545: "FL",
        4167583: "FL", 4167601: "FL", 4167634: "FL", 4167694: "FL",
        4167829: "FL", 4168139: "FL", 4168228: "FL", 4168459: "FL",
        4168590: "FL", 4168630: "FL", 4168659: "FL", 4168773: "FL",
        4168782: "FL", 4168930: "FL", 4169014: "FL", 4169060: "FL",
        4169130: "FL", 4169156: "FL", 4169171: "FL", 4169345: "FL",
        4169452: "FL", 4169455: "FL", 4170005: "FL", 4170156: "FL",
        4170174: "FL", 4170358: "FL", 4170617: "FL", 4170688: "FL",
        4170797: "FL", 4170965: "FL", 4171563: "FL", 4171782: "FL",
        4172086: "FL", 4172131: "FL", 4172372: "FL", 4172434: "FL",
        4173392: "FL", 4173497: "FL", 4173600: "FL", 4173838: "FL",
        4174201: "FL", 4174317: "FL", 4174383: "FL", 4174402: "FL",
        4174425: "FL", 4174600: "FL", 4174715: "FL", 4174738: "FL",
        4174744: "FL", 4174757: "FL", 4174855: "FL", 4174861: "FL",
        4174969: "FL", 4175091: "FL", 4175117: "FL", 4175179: "FL",
        4175296: "FL", 4175437: "FL", 4175538: "FL", 4176217: "FL",
        4176318: "FL", 4176380: "FL", 4176409: "FL", 4177671: "FL",
        4177703: "FL", 4177727: "FL", 4177779: "FL", 4177834: "FL",
        4177865: "FL", 4177872: "FL", 4177887: "FL", 4177897: "FL",
        4177908: "FL", 4177960: "FL", 4177965: "FL", 4178003: "FL",
        4178550: "FL", 4178552: "FL", 4178560: "FL", 4178573: "FL",
        4178775: "FL", 4179074: "GA", 4179320: "GA", 4179574: "GA",
        4179667: "GA", 4180386: "GA", 4180439: "GA", 4180531: "GA",
        4181936: "GA", 4184530: "GA", 4184845: "GA", 4185657: "GA",
        4186213: "GA", 4186416: "GA", 4186531: "GA", 4187204: "GA",
        4188985: "GA", 4189213: "GA", 4190581: "GA", 4191124: "GA",
        4191955: "GA", 4192205: "GA", 4192289: "GA", 4192375: "GA",
        4192674: "GA", 4193699: "GA", 4194474: "GA", 4195701: "GA",
        4196586: "GA", 4198322: "GA", 4200671: "GA", 4203696: "GA",
        4204007: "GA", 4204230: "GA", 4205196: "GA", 4205885: "GA",
        4207226: "GA", 4207400: "GA", 4207783: "GA", 4207981: "GA",
        4208442: "GA", 4209448: "GA", 4212684: "GA", 4212888: "GA",
        4212937: "GA", 4212992: "GA", 4212995: "GA", 4215110: "GA",
        4215114: "GA", 4215391: "GA", 4216948: "GA", 4218165: "GA",
        4219001: "GA", 4219762: "GA", 4219934: "GA", 4220629: "GA",
        4221333: "GA", 4221552: "GA", 4223379: "GA", 4223413: "GA",
        4224413: "GA", 4224681: "GA", 4225039: "GA", 4225309: "GA",
        4226348: "GA", 4226552: "GA", 4227213: "GA", 4227777: "GA",
        4228147: "GA", 4229476: "GA", 4231354: "GA", 4231523: "GA",
        4231874: "GA", 4232679: "IL", 4233813: "IL", 4235193: "IL",
        4235668: "IL", 4236191: "IL", 4236895: "IL", 4237579: "IL",
        4237717: "IL", 4238132: "IL", 4239509: "IL", 4239714: "IL",
        4241704: "IL", 4243899: "IL", 4244099: "IL", 4245152: "IL",
        4245926: "IL", 4247703: "IL", 4250542: "IL", 4251841: "IL",
        4254021: "IN", 4254679: "IN", 4254957: "IN", 4255056: "IN",
        4255466: "IN", 4255836: "IN", 4256038: "IN", 4257227: "IN",
        4257494: "IN", 4258285: "IN", 4258313: "IN", 4259418: "IN",
        4259640: "IN", 4259671: "IN", 4260210: "IN", 4262045: "IN",
        4262072: "IN", 4263108: "IN", 4263681: "IN", 4264617: "IN",
        4264688: "IN", 4265737: "IN", 4266307: "IN", 4267336: "IN",
        4270356: "KS", 4271086: "KS", 4271935: "KS", 4272340: "KS",
        4272782: "KS", 4273299: "KS", 4273680: "KS", 4273837: "KS",
        4274277: "KS", 4274305: "KS", 4274317: "KS", 4274356: "KS",
        4274994: "KS", 4276248: "KS", 4276614: "KS", 4276873: "KS",
        4277241: "KS", 4277718: "KS", 4278890: "KS", 4279247: "KS",
        4280539: "KS", 4281730: "KS", 4282757: "KY", 4285268: "KY",
        4286281: "KY", 4288809: "KY", 4289445: "KY", 4290988: "KY",
        4291255: "KY", 4291620: "KY", 4291945: "KY", 4292071: "KY",
        4292188: "KY", 4292686: "KY", 4294494: "KY", 4294799: "KY",
        4295251: "KY", 4295776: "KY", 4295940: "KY", 4296218: "KY",
        4297983: "KY", 4297999: "KY", 4299276: "KY", 4299670: "KY",
        4300488: "KY", 4302035: "KY", 4302504: "KY", 4302529: "KY",
        4302561: "KY", 4303012: "KY", 4303436: "KY", 4304713: "KY",
        4305504: "KY", 4305974: "KY", 4307238: "KY", 4308122: "KY",
        4308225: "KY", 4311963: "KY", 4313697: "KY", 4314550: "LA",
        4315588: "LA", 4315768: "LA", 4317639: "LA", 4319435: "LA",
        4319518: "LA", 4323873: "LA", 4326575: "LA", 4326868: "LA",
        4327047: "LA", 4328010: "LA", 4329753: "LA", 4330145: "LA",
        4330236: "LA", 4330525: "LA", 4332628: "LA", 4333177: "LA",
        4333190: "LA", 4333669: "LA", 4334720: "LA", 4334971: "LA",
        4335045: "LA", 4336153: "LA", 4338012: "LA", 4339348: "LA",
        4341378: "LA", 4341513: "LA", 4341727: "LA", 4342816: "LA",
        4343327: "LA", 4346788: "LA", 4346913: "MD", 4346991: "MD",
        4347242: "MD", 4347371: "MD", 4347426: "MD", 4347553: "MD",
        4347778: "MD", 4347839: "MD", 4348353: "MD", 4348599: "MD",
        4349159: "MD", 4350160: "MD", 4350288: "MD", 4350413: "MD",
        4350635: "MD", 4351383: "MD", 4351851: "MD", 4351871: "MD",
        4351887: "MD", 4351977: "MD", 4352053: "MD", 4352539: "MD",
        4352681: "MD", 4352728: "MD", 4353765: "MD", 4353925: "MD",
        4353962: "MD", 4354087: "MD", 4354163: "MD", 4354234: "MD",
        4354256: "MD", 4354265: "MD", 4354428: "MD", 4354573: "MD",
        4354821: "MD", 4355355: "MD", 4355585: "MD", 4355843: "MD",
        4356050: "MD", 4356165: "MD", 4356188: "MD", 4356783: "MD",
        4356847: "MD", 4357116: "MD", 4357141: "MD", 4357340: "MD",
        4358082: "MD", 4358701: "MD", 4358821: "MD", 4358864: "MD",
        4360201: "MD", 4360287: "MD", 4360314: "MD", 4360369: "MD",
        4360954: "MD", 4361831: "MD", 4362344: "MD", 4362438: "MD",
        4362743: "MD", 4363836: "MD", 4363843: "MD", 4363990: "MD",
        4364362: "MD", 4364537: "MD", 4364727: "MD", 4364759: "MD",
        4364946: "MD", 4364964: "MD", 4364990: "MD", 4365227: "MD",
        4365387: "MD", 4366001: "MD", 4366476: "MD", 4366593: "MD",
        4366647: "MD", 4367175: "MD", 4367372: "MD", 4367419: "MD",
        4367734: "MD", 4368711: "MD", 4368918: "MD", 4369076: "MD",
        4369190: "MD", 4369224: "MD", 4369596: "MD", 4369928: "MD",
        4369964: "MD", 4369978: "MD", 4370616: "MD", 4370890: "MD",
        4371582: "MD", 4372599: "MD", 4373104: "MD", 4373238: "MD",
        4373349: "MD", 4373449: "MD", 4374054: "MD", 4374513: "MO",
        4375087: "MO", 4375663: "MO", 4376482: "MO", 4377664: "MO",
        4379966: "MO", 4381072: "MO", 4381478: "MO", 4381982: "MO",
        4382072: "MO", 4382837: "MO", 4385018: "MO", 4386289: "MO",
        4386387: "MO", 4386802: "MO", 4387990: "MO", 4388460: "MO",
        4389418: "MO", 4389967: "MO", 4391812: "MO", 4392388: "MO",
        4392768: "MO", 4393217: "MO", 4393739: "MO", 4394870: "MO",
        4394905: "MO", 4395052: "MO", 4396915: "MO", 4397340: "MO",
        4397962: "MO", 4400648: "MO", 4401242: "MO", 4401618: "MO",
        4402178: "MO", 4402245: "MO", 4404233: "MO", 4405180: "MO",
        4405188: "MO", 4405434: "MO", 4406282: "MO", 4406831: "MO",
        4407010: "MO", 4407066: "MO", 4407237: "MO", 4408000: "MO",
        4408672: "MO", 4409591: "MO", 4409896: "MO", 4412697: "MO",
        4413595: "MO", 4413872: "MO", 4414001: "MO", 4414749: "MO",
        4418478: "MS", 4419290: "MS", 4421935: "MS", 4422133: "MS",
        4422442: "MS", 4427569: "MS", 4428475: "MS", 4428495: "MS",
        4428667: "MS", 4429295: "MS", 4429589: "MS", 4430400: "MS",
        4431410: "MS", 4433039: "MS", 4434069: "MS", 4434663: "MS",
        4435764: "MS", 4437982: "MS", 4439506: "MS", 4439869: "MS",
        4440076: "MS", 4440397: "MS", 4440559: "MS", 4443296: "MS",
        4446675: "MS", 4447161: "MS", 4448903: "MS", 4449620: "MS",
        4450687: "MS", 4452303: "NC", 4452808: "NC", 4453035: "NC",
        4453066: "NC", 4456703: "NC", 4458228: "NC", 4459343: "NC",
        4459467: "NC", 4460162: "NC", 4460243: "NC", 4460943: "NC",
        4461015: "NC", 4461574: "NC", 4461941: "NC", 4464368: "NC",
        4464873: "NC", 4465088: "NC", 4466033: "NC", 4467485: "NC",
        4467657: "NC", 4467732: "NC", 4468261: "NC", 4469146: "NC",
        4469160: "NC", 4470244: "NC", 4470566: "NC", 4470778: "NC",
        4471025: "NC", 4471641: "NC", 4471851: "NC", 4472370: "NC",
        4472687: "NC", 4473083: "NC", 4474040: "NC", 4474221: "NC",
        4474436: "NC", 4475347: "NC", 4475622: "NC", 4475640: "NC",
        4475773: "NC", 4477525: "NC", 4478334: "NC", 4479759: "NC",
        4479946: "NC", 4480125: "NC", 4480219: "NC", 4480285: "NC",
        4481682: "NC", 4485272: "NC", 4487042: "NC", 4488232: "NC",
        4488762: "NC", 4489985: "NC", 4490329: "NC", 4491180: "NC",
        4493186: "NC", 4493316: "NC", 4494942: "NC", 4497290: "NC",
        4498303: "NC", 4499379: "NC", 4499389: "NC", 4499612: "NC",
        4500546: "NJ", 4500688: "NJ", 4500942: "NJ", 4501018: "NJ",
        4501198: "NJ", 4501931: "NJ", 4502434: "NJ", 4502687: "NJ",
        4502901: "NJ", 4503078: "NJ", 4503136: "NJ", 4503346: "NJ",
        4503548: "NJ", 4503646: "NJ", 4504048: "NJ", 4504118: "NJ",
        4504225: "NJ", 4504476: "NJ", 4504618: "NJ", 4504621: "NJ",
        4504871: "NJ", 4505542: "OH", 4506008: "OH", 4508204: "OH",
        4508722: "OH", 4509177: "OH", 4509884: "OH", 4511263: "OH",
        4511283: "OH", 4512060: "OH", 4513409: "OH", 4513575: "OH",
        4514746: "OH", 4515843: "OH", 4516233: "OH", 4516412: "OH",
        4517698: "OH", 4518188: "OH", 4518264: "OH", 4519995: "OH",
        4520760: "OH", 4520905: "OH", 4521209: "OH", 4521816: "OH",
        4522411: "OH", 4522586: "OH", 4525304: "OH", 4525353: "OH",
        4526576: "OH", 4526993: "OH", 4527124: "OH", 4528259: "OH",
        4528291: "OH", 4528810: "OH", 4528923: "OH", 4529096: "OK",
        4529292: "OK", 4529469: "OK", 4529987: "OK", 4530372: "OK",
        4530801: "OK", 4531405: "OK", 4533029: "OK", 4533580: "OK",
        4534934: "OK", 4535389: "OK", 4535414: "OK", 4535740: "OK",
        4535783: "OK", 4535961: "OK", 4539615: "OK", 4540737: "OK",
        4542367: "OK", 4542765: "OK", 4542975: "OK", 4543338: "OK",
        4543352: "OK", 4543762: "OK", 4544349: "OK", 4547407: "OK",
        4548267: "OK", 4550659: "OK", 4550881: "OK", 4551177: "OK",
        4552215: "OK", 4552707: "OK", 4553433: "OK", 4556165: "OK",
        4557109: "PA", 4557137: "PA", 4557606: "PA", 4558475: "PA",
        4560303: "PA", 4560349: "PA", 4561407: "PA", 4562144: "PA",
        4562193: "PA", 4562237: "PA", 4562407: "PA", 4569067: "SC",
        4569298: "SC", 4571722: "SC", 4574324: "SC", 4574989: "SC",
        4575352: "SC", 4575461: "SC", 4577263: "SC", 4578737: "SC",
        4580098: "SC", 4580543: "SC", 4580569: "SC", 4580599: "SC",
        4581023: "SC", 4581832: "SC", 4581833: "SC", 4585000: "SC",
        4586523: "SC", 4588165: "SC", 4588718: "SC", 4589368: "SC",
        4589387: "SC", 4589446: "SC", 4593142: "SC", 4593724: "SC",
        4595374: "SC", 4595864: "SC", 4596208: "SC", 4597200: "SC",
        4597919: "SC", 4597948: "SC", 4598353: "SC", 4599937: "SC",
        4600541: "SC", 4604183: "TN", 4608408: "TN", 4608418: "TN",
        4608657: "TN", 4612862: "TN", 4613868: "TN", 4614088: "TN",
        4614748: "TN", 4614867: "TN", 4615145: "TN", 4618057: "TN",
        4619800: "TN", 4619943: "TN", 4619947: "TN", 4620131: "TN",
        4621846: "TN", 4623560: "TN", 4624180: "TN", 4624601: "TN",
        4625282: "TN", 4626334: "TN", 4628735: "TN", 4632595: "TN",
        4633419: "TN", 4634662: "TN", 4634946: "TN", 4635031: "TN",
        4636045: "TN", 4639848: "TN", 4641239: "TN", 4642938: "TN",
        4643336: "TN", 4644312: "TN", 4644585: "TN", 4645421: "TN",
        4646571: "TN", 4656585: "TN", 4657077: "TN", 4658590: "TN",
        4659446: "TN", 4659557: "TN", 4663494: "TN", 4669635: "TX",
        4669828: "TX", 4670074: "TX", 4670146: "TX", 4670234: "TX",
        4670249: "TX", 4670300: "TX", 4670555: "TX", 4670592: "TX",
        4670866: "TX", 4671240: "TX", 4671524: "TX", 4671654: "TX",
        4672059: "TX", 4672731: "TX", 4672989: "TX", 4673094: "TX",
        4673353: "TX", 4673425: "TX", 4673482: "TX", 4676206: "TX",
        4676740: "TX", 4676798: "TX", 4676920: "TX", 4677008: "TX",
        4677551: "TX", 4678901: "TX", 4679195: "TX", 4679803: "TX",
        4679867: "TX", 4680388: "TX", 4681462: "TX", 4681485: "TX",
        4681976: "TX", 4682127: "TX", 4682464: "TX", 4682478: "TX",
        4682991: "TX", 4683021: "TX", 4683217: "TX", 4683244: "TX",
        4683317: "TX", 4683416: "TX", 4683462: "TX", 4684724: "TX",
        4684888: "TX", 4685524: "TX", 4685737: "TX", 4685892: "TX",
        4685907: "TX", 4686163: "TX", 4686593: "TX", 4687331: "TX",
        4688275: "TX", 4689311: "TX", 4689550: "TX", 4690198: "TX",
        4691585: "TX", 4691833: "TX", 4691930: "TX", 4692400: "TX",
        4692521: "TX", 4692559: "TX", 4692746: "TX", 4692883: "TX",
        4693003: "TX", 4693150: "TX", 4693342: "TX", 4694482: "TX",
        4694568: "TX", 4695066: "TX", 4695317: "TX", 4695912: "TX",
        4696202: "TX", 4696233: "TX", 4697645: "TX", 4699066: "TX",
        4699442: "TX", 4699540: "TX", 4699575: "TX", 4699626: "TX",
        4700168: "TX", 4701458: "TX", 4702732: "TX", 4702828: "TX",
        4703078: "TX", 4703223: "TX", 4703384: "TX", 4703811: "TX",
        4704027: "TX", 4704108: "TX", 4704628: "TX", 4705191: "TX",
        4705349: "TX", 4705692: "TX", 4705708: "TX", 4706057: "TX",
        4706736: "TX", 4707055: "TX", 4707814: "TX", 4708308: "TX",
        4709013: "TX", 4709272: "TX", 4709796: "TX", 4710178: "TX",
        4710749: "TX", 4710826: "TX", 4711156: "TX", 4711725: "TX",
        4711729: "TX", 4711801: "TX", 4712933: "TX", 4713507: "TX",
        4713735: "TX", 4713932: "TX", 4714131: "TX", 4714582: "TX",
        4715292: "TX", 4716805: "TX", 4717232: "TX", 4717560: "TX",
        4717782: "TX", 4718097: "TX", 4718209: "TX", 4718711: "TX",
        4718721: "TX", 4719457: "TX", 4720039: "TX", 4720131: "TX",
        4720833: "TX", 4722625: "TX", 4723406: "TX", 4723914: "TX",
        4724129: "TX", 4724194: "TX", 4724564: "TX", 4724642: "TX",
        4726206: "TX", 4726290: "TX", 4726440: "TX", 4726491: "TX",
        4727326: "TX", 4727605: "TX", 4727756: "TX", 4728328: "TX",
        4733042: "TX", 4733313: "TX", 4733624: "TX", 4734005: "TX",
        4734350: "TX", 4734825: "TX", 4734909: "TX", 4735702: "TX",
        4735966: "TX", 4736028: "TX", 4736096: "TX", 4736134: "TX",
        4736388: "TX", 4736476: "TX", 4738214: "TX", 4738574: "TX",
        4738606: "TX", 4738721: "TX", 4739157: "TX", 4739526: "TX",
        4740214: "TX", 4740328: "TX", 4740364: "TX", 4740629: "TX",
        4741100: "TX", 4741616: "TX", 4741752: "TX", 4743275: "TX",
        4744091: "VA", 4744468: "VA", 4744709: "VA", 4744870: "VA",
        4745272: "VA", 4747845: "VA", 4748305: "VA", 4748993: "VA",
        4749627: "VA", 4749950: "VA", 4751421: "VA", 4751839: "VA",
        4751935: "VA", 4752031: "VA", 4752136: "VA", 4752186: "VA",
        4752229: "VA", 4752665: "VA", 4753671: "VA", 4754966: "VA",
        4755158: "VA", 4755280: "VA", 4756955: "VA", 4758023: "VA",
        4759968: "VA", 4760059: "VA", 4760232: "VA", 4761951: "VA",
        4762894: "VA", 4763231: "VA", 4763793: "VA", 4764127: "VA",
        4764826: "VA", 4765520: "VA", 4765553: "VA", 4768351: "VA",
        4768678: "VA", 4769125: "VA", 4769608: "VA", 4769667: "VA",
        4770714: "VA", 4771075: "VA", 4771401: "VA", 4771414: "VA",
        4772354: "VA", 4772566: "VA", 4772735: "VA", 4773677: "VA",
        4776024: "VA", 4776222: "VA", 4776970: "VA", 4778626: "VA",
        4779999: "VA", 4780011: "VA", 4780837: "VA", 4781530: "VA",
        4781708: "VA", 4782167: "VA", 4782864: "VA", 4784112: "VA",
        4785576: "VA", 4786667: "VA", 4786714: "VA", 4787117: "VA",
        4787440: "VA", 4787534: "VA", 4788145: "VA", 4788158: "VA",
        4790207: "VA", 4790534: "VA", 4791160: "VA", 4791259: "VA",
        4792522: "VA", 4792867: "VA", 4792901: "VA", 4793846: "VA",
        4794120: "VA", 4794350: "VA", 4794531: "VA", 4798308: "WV",
        4801859: "WV", 4802316: "WV", 4805404: "WV", 4809537: "WV",
        4813878: "WV", 4815352: "WV", 4817641: "WV", 4828193: "AR",
        4828382: "IN", 4828890: "OH", 4829307: "TX", 4829762: "AL",
        4829791: "AL", 4830198: "AL", 4830668: "AL", 4830796: "AL",
        4832038: "IL", 4832272: "MA", 4832294: "MA", 4832353: "CT",
        4832425: "CT", 4832554: "MI", 4833320: "OH", 4833403: "CT",
        4833425: "CT", 4833505: "CT", 4834040: "CT", 4834157: "CT",
        4834272: "CT", 4835003: "CT", 4835512: "CT", 4835654: "CT",
        4835797: "CT", 4837278: "CT", 4837648: "CT", 4838116: "CT",
        4838174: "CT", 4838204: "CT", 4838524: "CT", 4838633: "CT",
        4838652: "CT", 4838887: "CT", 4839222: "CT", 4839292: "CT",
        4839319: "CT", 4839366: "CT", 4839416: "CT", 4839497: "CT",
        4839704: "CT", 4839745: "CT", 4839822: "CT", 4839843: "CT",
        4840755: "CT", 4840767: "CT", 4842818: "CT", 4842898: "CT",
        4843353: "CT", 4843362: "CT", 4843564: "CT", 4843786: "CT",
        4843811: "CT", 4844309: "CT", 4844459: "CT", 4845056: "CT",
        4845193: "CT", 4845411: "CT", 4845419: "CT", 4845519: "CT",
        4845585: "CT", 4845612: "CT", 4845823: "CT", 4845871: "CT",
        4845898: "CT", 4845920: "CT", 4845984: "CT", 4846757: "IA",
        4846834: "IA", 4846960: "IA", 4848489: "IA", 4849826: "IA",
        4850699: "IA", 4850751: "IA", 4852022: "IA", 4852065: "IA",
        4852640: "IA", 4852832: "IA", 4853423: "IA", 4853828: "IA",
        4854529: "IA", 4857486: "IA", 4861719: "IA", 4862034: "IA",
        4862760: "IA", 4866263: "IA", 4866371: "IA", 4866445: "IA",
        4868404: "IA", 4868907: "IA", 4869195: "IA", 4870380: "IA",
        4876523: "IA", 4879890: "IA", 4880889: "IA", 4880981: "IA",
        4881346: "IA", 4882920: "IL", 4883012: "IL", 4883078: "IL",
        4883207: "IL", 4883555: "IL", 4883679: "IL", 4883817: "IL",
        4883904: "IL", 4884141: "IL", 4884192: "IL", 4884434: "IL",
        4884442: "IL", 4884453: "IL", 4884509: "IL", 4884597: "IL",
        4885156: "IL", 4885164: "IL", 4885186: "IL", 4885265: "IL",
        4885342: "IL", 4885418: "IL", 4885565: "IL", 4885573: "IL",
        4885597: "IL", 4885689: "IL", 4885955: "IL", 4885983: "IL",
        4886255: "IL", 4886662: "IL", 4886676: "IL", 4886738: "IL",
        4887158: "IL", 4887284: "IL", 4887398: "IL", 4887442: "IL",
        4887463: "IL", 4888015: "IL", 4888892: "IL", 4889107: "IL",
        4889229: "IL", 4889426: "IL", 4889447: "IL", 4889553: "IL",
        4889668: "IL", 4889772: "IL", 4889959: "IL", 4890009: "IL",
        4890075: "IL", 4890119: "IL", 4890507: "IL", 4890536: "IL",
        4890549: "IL", 4890701: "IL", 4890864: "IL", 4890925: "IL",
        4891010: "IL", 4891051: "IL", 4891176: "IL", 4891382: "IL",
        4891431: "IL", 4893037: "IL", 4893070: "IL", 4893171: "IL",
        4893365: "IL", 4893392: "IL", 4893591: "IL", 4893811: "IL",
        4893886: "IL", 4894061: "IL", 4894320: "IL", 4894465: "IL",
        4894861: "IL", 4895066: "IL", 4895298: "IL", 4895876: "IL",
        4896012: "IL", 4896075: "IL", 4896336: "IL", 4896353: "IL",
        4896691: "IL", 4896728: "IL", 4897543: "IL", 4898015: "IL",
        4898182: "IL", 4898401: "IL", 4898846: "IL", 4899012: "IL",
        4899170: "IL", 4899184: "IL", 4899340: "IL", 4899581: "IL",
        4899739: "IL", 4899911: "IL", 4899966: "IL", 4900080: "IL",
        4900292: "IL", 4900358: "IL", 4900373: "IL", 4900579: "IL",
        4900611: "IL", 4900801: "IL", 4900817: "IL", 4901445: "IL",
        4901514: "IL", 4901663: "IL", 4901710: "IL", 4901868: "IL",
        4902475: "IL", 4902476: "IL", 4902559: "IL", 4902667: "IL",
        4902754: "IL", 4902763: "IL", 4902900: "IL", 4903024: "IL",
        4903184: "IL", 4903279: "IL", 4903360: "IL", 4903363: "IL",
        4903466: "IL", 4903535: "IL", 4903730: "IL", 4903780: "IL",
        4903818: "IL", 4903858: "IL", 4903862: "IL", 4903940: "IL",
        4903976: "IL", 4904056: "IL", 4904286: "IL", 4904365: "IL",
        4904381: "IL", 4904937: "IL", 4904996: "IL", 4905006: "IL",
        4905211: "IL", 4905263: "IL", 4905337: "IL", 4905367: "IL",
        4905599: "IL", 4905687: "IL", 4906125: "IL", 4906500: "IL",
        4906882: "IL", 4907907: "IL", 4907959: "IL", 4908033: "IL",
        4908052: "IL", 4908068: "IL", 4908173: "IL", 4908236: "IL",
        4908237: "IL", 4908737: "IL", 4910713: "IL", 4911418: "IL",
        4911600: "IL", 4911863: "IL", 4911893: "IL", 4911934: "IL",
        4911951: "IL", 4912013: "IL", 4912499: "IL", 4912691: "IL",
        4913110: "IL", 4913723: "IL", 4914557: "IL", 4914570: "IL",
        4914738: "IL", 4914830: "IL", 4915539: "IL", 4915545: "IL",
        4915734: "IL", 4915963: "IL", 4915987: "IL", 4915989: "IL",
        4916003: "IL", 4916028: "IL", 4916079: "IL", 4916118: "IL",
        4916140: "IL", 4916207: "IL", 4916288: "IL", 4916311: "IL",
        4916732: "IL", 4917067: "IL", 4917089: "IL", 4917123: "IL",
        4917298: "IL", 4917358: "IL", 4917592: "IN", 4919381: "IN",
        4919451: "IN", 4919820: "IN", 4919857: "IN", 4919987: "IN",
        4920423: "IN", 4920473: "IN", 4920607: "IN", 4920808: "IN",
        4920869: "IN", 4920986: "IN", 4921100: "IN", 4921402: "IN",
        4921476: "IN", 4921725: "IN", 4922388: "IN", 4922459: "IN",
        4922462: "IN", 4922673: "IN", 4922968: "IN", 4923210: "IN",
        4923482: "IN", 4923531: "IN", 4923670: "IN", 4924006: "IN",
        4924014: "IN", 4924104: "IN", 4924198: "IN", 4925006: "IN",
        4926170: "IN", 4926563: "IN", 4927537: "IN", 4928096: "IN",
        4928118: "IN", 4928662: "MA", 4928703: "MA", 4928788: "MA",
        4929004: "MA", 4929022: "MA", 4929023: "MA", 4929180: "MA",
        4929283: "MA", 4929399: "MA", 4929417: "MA", 4929771: "MA",
        4930282: "MA", 4930505: "MA", 4930511: "MA", 4930577: "MA",
        4930956: "MA", 4931429: "MA", 4931482: "MA", 4931737: "MA",
        4931972: "MA", 4932214: "MA", 4932869: "MA", 4932879: "MA",
        4933002: "MA", 4933743: "MA", 4934500: "MA", 4934664: "MA",
        4935038: "MA", 4935211: "MA", 4935434: "MA", 4935582: "MA",
        4935623: "MA", 4936008: "MA", 4936087: "MA", 4936159: "MA",
        4936812: "MA", 4937230: "MA", 4937232: "MA", 4937276: "MA",
        4937557: "MA", 4937829: "MA", 4938048: "MA", 4938378: "MA",
        4938836: "MA", 4939085: "MA", 4939647: "MA", 4939783: "MA",
        4940764: "MA", 4941720: "MA", 4941873: "MA", 4941935: "MA",
        4942508: "MA", 4942618: "MA", 4942744: "MA", 4942807: "MA",
        4942939: "MA", 4943021: "MA", 4943097: "MA", 4943170: "MA",
        4943629: "MA", 4943677: "MA", 4943828: "MA", 4943888: "MA",
        4943958: "MA", 4944193: "MA", 4944994: "MA", 4945055: "MA",
        4945121: "MA", 4945256: "MA", 4945283: "MA", 4945588: "MA",
        4945819: "MA", 4945911: "MA", 4945952: "MA", 4946620: "MA",
        4946863: "MA", 4947459: "MA", 4948247: "MA", 4948403: "MA",
        4948462: "MA", 4948924: "MA", 4950065: "MA", 4950267: "MA",
        4950898: "MA", 4951248: "MA", 4951257: "MA", 4951305: "MA",
        4951397: "MA", 4951473: "MA", 4951594: "MA", 4951788: "MA",
        4952121: "MA", 4952206: "MA", 4952320: "MA", 4952487: "MA",
        4952629: "MA", 4952762: "MA", 4954265: "MA", 4954380: "MA",
        4954611: "MA", 4954738: "MA", 4955089: "MA", 4955190: "MA",
        4955219: "MA", 4955336: "MA", 4955840: "MA", 4955884: "MA",
        4955993: "MA", 4956032: "MA", 4956184: "MA", 4956335: "MA",
        4956976: "ME", 4957003: "ME", 4957280: "ME", 4958141: "ME",
        4959473: "ME", 4969398: "ME", 4975802: "ME", 4977222: "ME",
        4977762: "ME", 4979244: "ME", 4979245: "ME", 4982236: "ME",
        4982720: "ME", 4982753: "ME", 4983811: "MI", 4984016: "MI",
        4984029: "MI", 4984247: "MI", 4984565: "MI", 4985153: "MI",
        4985180: "MI", 4985744: "MI", 4986172: "MI", 4987482: "MI",
        4987990: "MI", 4989133: "MI", 4990510: "MI", 4990512: "MI",
        4990729: "MI", 4991640: "MI", 4991735: "MI", 4992523: "MI",
        4992635: "MI", 4992982: "MI", 4993125: "MI", 4993659: "MI",
        4994358: "MI", 4994391: "MI", 4994871: "MI", 4995197: "MI",
        4995514: "MI", 4995664: "MI", 4996248: "MI", 4996306: "MI",
        4997384: "MI", 4997500: "MI", 4997787: "MI", 4998018: "MI",
        4998830: "MI", 4999311: "MI", 4999837: "MI", 5000500: "MI",
        5000947: "MI", 5001929: "MI", 5002344: "MI", 5002656: "MI",
        5002714: "MI", 5003132: "MI", 5004005: "MI", 5004062: "MI",
        5004188: "MI", 5004359: "MI", 5006166: "MI", 5006233: "MI",
        5006250: "MI", 5006917: "MI", 5007402: "MI", 5007531: "MI",
        5007655: "MI", 5007804: "MI", 5007989: "MI", 5009586: "MI",
        5010636: "MI", 5010646: "MI", 5010978: "MI", 5011148: "MI",
        5011908: "MI", 5012495: "MI", 5012521: "MI", 5012639: "MI",
        5013924: "MI", 5014051: "MI", 5014130: "MI", 5014208: "MI",
        5014224: "MI", 5014681: "MI", 5015599: "MI", 5015618: "MI",
        5015688: "MI", 5016024: "MN", 5016374: "MN", 5016450: "MN",
        5016494: "MN", 5016884: "MN", 5018651: "MN", 5018739: "MN",
        5019330: "MN", 5019335: "MN", 5019588: "MN", 5019767: "MN",
        5020859: "MN", 5020881: "MN", 5020938: "MN", 5021828: "MN",
        5022025: "MN", 5022134: "MN", 5023571: "MN", 5024719: "MN",
        5024825: "MN", 5025219: "MN", 5025264: "MN", 5025471: "MN",
        5026291: "MN", 5026321: "MN", 5027117: "MN", 5027482: "MN",
        5028163: "MN", 5029181: "MN", 5029500: "MN", 5030005: "MN",
        5030670: "MN", 5031412: "MN", 5034059: "MN", 5034767: "MN",
        5036420: "MN", 5036493: "MN", 5036588: "MN", 5037649: "MN",
        5037784: "MN", 5037790: "MN", 5038108: "MN", 5039080: "MN",
        5039094: "MN", 5039675: "MN", 5039978: "MN", 5040477: "MN",
        5040647: "MN", 5041926: "MN", 5042373: "MN", 5042561: "MN",
        5042773: "MN", 5043193: "MN", 5043473: "MN", 5043779: "MN",
        5043799: "MN", 5044407: "MN", 5045021: "MN", 5045258: "MN",
        5045360: "MN", 5046001: "MN", 5046063: "MN", 5046997: "MN",
        5047234: "MN", 5048033: "MN", 5048814: "MN", 5052361: "MN",
        5052467: "MN", 5052658: "MN", 5052916: "MN", 5053156: "MN",
        5053358: "MN", 5055787: "MO", 5059163: "ND", 5059429: "ND",
        5059836: "ND", 5062458: "ND", 5063805: "NE", 5066001: "NE",
        5068725: "NE", 5069297: "NE", 5069802: "NE", 5071348: "NE",
        5071665: "NE", 5072006: "NE", 5073965: "NE", 5074472: "NE",
        5074792: "NE", 5083221: "NH", 5084868: "NH", 5085374: "NH",
        5085382: "NH", 5085520: "NH", 5085688: "NH", 5088262: "NH",
        5088438: "NH", 5089178: "NH", 5089478: "NH", 5090046: "NH",
        5091383: "NH", 5091872: "NH", 5092268: "NH", 5095281: "NJ",
        5095325: "NJ", 5095409: "NJ", 5095445: "NJ", 5095549: "NJ",
        5095611: "NJ", 5095779: "NJ", 5096316: "NJ", 5096686: "NJ",
        5096699: "NJ", 5096798: "NJ", 5097017: "NJ", 5097239: "NJ",
        5097315: "NJ", 5097357: "NJ", 5097402: "NJ", 5097441: "NJ",
        5097529: "NJ", 5097598: "NJ", 5097627: "NJ", 5097672: "NJ",
        5097751: "NJ", 5097773: "NJ", 5098109: "NJ", 5098135: "NJ",
        5098343: "NJ", 5098706: "NJ", 5098863: "NJ", 5098909: "NJ",
        5099079: "NJ", 5099093: "NJ", 5099133: "NJ", 5099292: "NJ",
        5099724: "NJ", 5099738: "NJ", 5099836: "NJ", 5099967: "NJ",
        5100280: "NJ", 5100506: "NJ", 5100572: "NJ", 5100604: "NJ",
        5100619: "NJ", 5100706: "NJ", 5100748: "NJ", 5100776: "NJ",
        5100854: "NJ", 5100886: "NJ", 5101170: "NJ", 5101334: "NJ",
        5101427: "NJ", 5101717: "NJ", 5101766: "NJ", 5101798: "NJ",
        5101873: "NJ", 5101879: "NJ", 5101938: "NJ", 5102076: "NJ",
        5102162: "NJ", 5102213: "NJ", 5102369: "NJ", 5102387: "NJ",
        5102427: "NJ", 5102443: "NJ", 5102466: "NJ", 5102578: "NJ",
        5102713: "NJ", 5102720: "NJ", 5102796: "NJ", 5102922: "NJ",
        5103055: "NJ", 5103086: "NJ", 5103269: "NJ", 5103500: "NJ",
        5103580: "NJ", 5104404: "NJ", 5104405: "NJ", 5104473: "NJ",
        5104504: "NJ", 5104755: "NJ", 5104835: "NJ", 5104836: "NJ",
        5104844: "NJ", 5104853: "NJ", 5104882: "NJ", 5105127: "NJ",
        5105262: "NJ", 5105433: "NJ", 5105496: "NJ", 5105608: "NJ",
        5105634: "NJ", 5105860: "NJ", 5106160: "NJ", 5106279: "NJ",
        5106292: "NJ", 5106298: "NJ", 5106331: "NJ", 5106453: "NJ",
        5106529: "NJ", 5106615: "NJ", 5106834: "NY", 5107129: "NY",
        5107152: "NY", 5107464: "NY", 5107505: "NY", 5107760: "NY",
        5108093: "NY", 5108111: "NY", 5108169: "NY", 5108186: "NY",
        5108193: "NY", 5108707: "NY", 5108815: "NY", 5108955: "NY",
        5109177: "NY", 5109790: "NY", 5110077: "NY", 5110159: "NY",
        5110161: "NY", 5110266: "NY", 5110302: "NY", 5110309: "NY",
        5110446: "NY", 5110629: "NY", 5110918: "NY", 5111141: "NY",
        5111412: "NY", 5112035: "NY", 5112078: "NY", 5112375: "NY",
        5112540: "NY", 5112710: "NY", 5112861: "NY", 5112961: "NY",
        5113142: "NY", 5113302: "NY", 5113412: "NY", 5113481: "NY",
        5113683: "NY", 5113694: "NY", 5113779: "NY", 5113790: "NY",
        5114418: "NY", 5114731: "NY", 5114900: "NY", 5115107: "NY",
        5115614: "NY", 5115699: "NY", 5115835: "NY", 5115843: "NY",
        5115960: "NY", 5115962: "NY", 5115985: "NY", 5115989: "NY",
        5116004: "NY", 5116060: "NY", 5116083: "NY", 5116093: "NY",
        5116118: "NY", 5116303: "NY", 5116495: "NY", 5116497: "NY",
        5116508: "NY", 5116570: "NY", 5116917: "NY", 5116937: "NY",
        5117378: "NY", 5117388: "NY", 5117438: "NY", 5117549: "NY",
        5117575: "NY", 5117663: "NY", 5117891: "NY", 5117949: "NY",
        5118005: "NY", 5118226: "NY", 5118626: "NY", 5118670: "NY",
        5118743: "NY", 5119019: "NY", 5119049: "NY", 5119167: "NY",
        5119211: "NY", 5119347: "NY", 5119383: "NY", 5120034: "NY",
        5120095: "NY", 5120228: "NY", 5120442: "NY", 5120478: "NY",
        5120521: "NY", 5120656: "NY", 5120824: "NY", 5120987: "NY",
        5121026: "NY", 5121163: "NY", 5121407: "NY", 5121636: "NY",
        5121650: "NY", 5121666: "NY", 5122331: "NY", 5122413: "NY",
        5122432: "NY", 5122477: "NY", 5122520: "NY", 5122534: "NY",
        5123247: "NY", 5123280: "NY", 5123344: "NY", 5123443: "NY",
        5123456: "NY", 5123477: "NY", 5123533: "NY", 5123718: "NY",
        5123840: "NY", 5124045: "NY", 5124078: "NY", 5124276: "NY",
        5124497: "NY", 5125011: "NY", 5125086: "NY", 5125125: "NY",
        5125523: "NY", 5125738: "NY", 5125771: "NY", 5126013: "NY",
        5126180: "NY", 5126183: "NY", 5126187: "NY", 5126208: "NY",
        5126518: "NY", 5126550: "NY", 5126555: "NY", 5126630: "NY",
        5126827: "NY", 5126842: "NY", 5127134: "NY", 5127315: "NY",
        5127536: "NY", 5127550: "NY", 5127670: "NY", 5127835: "NY",
        5128266: "NY", 5128481: "NY", 5128549: "NY", 5128566: "NY",
        5128581: "NY", 5128654: "NY", 5128723: "NY", 5128884: "NY",
        5128886: "NY", 5128898: "NY", 5128904: "NY", 5129134: "NY",
        5129245: "NY", 5129248: "NY", 5129603: "NY", 5130045: "NY",
        5130081: "NY", 5130334: "NY", 5130561: "NY", 5130572: "NY",
        5130780: "NY", 5130831: "NY", 5131638: "NY", 5131692: "NY",
        5132002: "NY", 5132028: "NY", 5132029: "NY", 5132143: "NY",
        5133271: "NY", 5133273: "NY", 5133279: "NY", 5133640: "NY",
        5133825: "NY", 5133858: "NY", 5134086: "NY", 5134203: "NY",
        5134295: "NY", 5134316: "NY", 5134323: "NY", 5134395: "NY",
        5134449: "NY", 5134453: "NY", 5134693: "NY", 5136334: "NY",
        5136421: "NY", 5136433: "NY", 5136454: "NY", 5137507: "NY",
        5137600: "NY", 5137849: "NY", 5138022: "NY", 5138539: "NY",
        5138950: "NY", 5139287: "NY", 5139301: "NY", 5139568: "NY",
        5140221: "NY", 5140402: "NY", 5140405: "NY", 5140675: "NY",
        5140945: "NY", 5141342: "NY", 5141502: "NY", 5141927: "NY",
        5141930: "NY", 5141963: "NY", 5142056: "NY", 5142109: "NY",
        5142182: "NY", 5142296: "NY", 5143056: "NY", 5143198: "NY",
        5143307: "NY", 5143396: "NY", 5143620: "NY", 5143630: "NY",
        5143832: "NY", 5143866: "NY", 5143992: "NY", 5144040: "NY",
        5144336: "NY", 5144400: "NY", 5144580: "NY", 5144975: "NY",
        5145028: "NY", 5145034: "NY", 5145067: "NY", 5145215: "NY",
        5145476: "OH", 5145607: "OH", 5146055: "OH", 5146089: "OH",
        5146233: "OH", 5146256: "OH", 5146277: "OH", 5146282: "OH",
        5146286: "OH", 5146491: "OH", 5146675: "OH", 5147097: "OH",
        5147784: "OH", 5147968: "OH", 5148273: "OH", 5148326: "OH",
        5148480: "OH", 5149222: "OH", 5150529: "OH", 5150792: "OH",
        5151613: "OH", 5151861: "OH", 5151891: "OH", 5152333: "OH",
        5152599: "OH", 5152833: "OH", 5153207: "OH", 5153420: "OH",
        5153680: "OH", 5153924: "OH", 5155207: "OH", 5155393: "OH",
        5155499: "OH", 5155858: "OH", 5156371: "OH", 5157588: "OH",
        5158067: "OH", 5158164: "OH", 5159537: "OH", 5160315: "OH",
        5160783: "OH", 5161262: "OH", 5161723: "OH", 5161803: "OH",
        5161902: "OH", 5162077: "OH", 5162097: "OH", 5162188: "OH",
        5162512: "OH", 5162645: "OH", 5162851: "OH", 5163799: "OH",
        5164390: "OH", 5164466: "OH", 5164582: "OH", 5164706: "OH",
        5164862: "OH", 5164903: "OH", 5164916: "OH", 5165101: "OH",
        5165734: "OH", 5166009: "OH", 5166177: "OH", 5166184: "OH",
        5166516: "OH", 5166819: "OH", 5168491: "OH", 5170691: "OH",
        5171728: "OH", 5172078: "OH", 5172387: "OH", 5172485: "OH",
        5173048: "OH", 5173171: "OH", 5173210: "OH", 5173237: "OH",
        5173572: "OH", 5173623: "OH", 5173930: "OH", 5174035: "OH",
        5174358: "OH", 5174550: "OH", 5175496: "OH", 5175865: "OH",
        5176472: "OH", 5176517: "OH", 5176937: "OH", 5177358: "OH",
        5177568: "OH", 5177773: "PA", 5178127: "PA", 5178165: "PA",
        5178195: "PA", 5178800: "PA", 5178940: "PA", 5179995: "PA",
        5180199: "PA", 5180225: "PA", 5183234: "PA", 5188140: "PA",
        5188843: "PA", 5192726: "PA", 5193011: "PA", 5193309: "PA",
        5195561: "PA", 5196220: "PA", 5197079: "PA", 5197159: "PA",
        5197517: "PA", 5197796: "PA", 5198034: "PA", 5200499: "PA",
        5201734: "PA", 5202215: "PA", 5202765: "PA", 5203127: "PA",
        5203506: "PA", 5205377: "PA", 5205849: "PA", 5206379: "PA",
        5206606: "PA", 5207069: "PA", 5207490: "PA", 5207728: "PA",
        5211303: "PA", 5213681: "PA", 5216895: "PA", 5218270: "PA",
        5218802: "PA", 5219287: "PA", 5219488: "PA", 5219501: "PA",
        5219585: "PA", 5219619: "PA", 5220798: "RI", 5221077: "RI",
        5221341: "RI", 5221637: "RI", 5221659: "RI", 5221703: "RI",
        5221931: "RI", 5223358: "RI", 5223505: "RI", 5223593: "RI",
        5223672: "RI", 5223681: "RI", 5223869: "RI", 5224082: "RI",
        5224151: "RI", 5224949: "RI", 5225507: "RI", 5225627: "RI",
        5225631: "RI", 5225809: "RI", 5225857: "SD", 5226534: "SD",
        5229794: "SD", 5231851: "SD", 5232741: "SD", 5234372: "VT",
        5235024: "VT", 5240509: "VT", 5241248: "VT", 5244080: "WI",
        5244267: "WI", 5245193: "WI", 5245359: "WI", 5245387: "WI",
        5246835: "WI", 5247415: "WI", 5249871: "WI", 5250201: "WI",
        5251436: "WI", 5253219: "WI", 5253352: "WI", 5253710: "WI",
        5254218: "WI", 5254962: "WI", 5255068: "WI", 5257029: "WI",
        5257754: "WI", 5258296: "WI", 5258393: "WI", 5258957: "WI",
        5261457: "WI", 5261585: "WI", 5261969: "WI", 5262596: "WI",
        5262630: "WI", 5262634: "WI", 5262649: "WI", 5262838: "WI",
        5263045: "WI", 5264049: "WI", 5264223: "WI", 5264381: "WI",
        5264870: "WI", 5265228: "WI", 5265499: "WI", 5265702: "WI",
        5265838: "WI", 5267403: "WI", 5268249: "WI", 5268986: "WI",
        5272893: "WI", 5273812: "WI", 5274644: "WI", 5275020: "WI",
        5275191: "WI", 5278005: "WI", 5278052: "WI", 5278120: "WI",
        5278159: "WI", 5278420: "WI", 5278422: "WI", 5278693: "WI",
        5279436: "WI", 5280814: "WV", 5280822: "WV", 5280854: "WV",
        5281551: "CT", 5282804: "CT", 5282835: "CT", 5283054: "CT",
        5283837: "CT", 5284756: "CA", 5287262: "AZ", 5287565: "AZ",
        5288636: "AZ", 5288661: "AZ", 5288786: "AZ", 5289282: "AZ",
        5292387: "AZ", 5293083: "AZ", 5293183: "AZ", 5293996: "AZ",
        5294167: "AZ", 5294810: "AZ", 5294902: "AZ", 5294937: "AZ",
        5295143: "AZ", 5295177: "AZ", 5295903: "AZ", 5295985: "AZ",
        5296266: "AZ", 5296802: "AZ", 5301067: "AZ", 5301388: "AZ",
        5303705: "AZ", 5303752: "AZ", 5303929: "AZ", 5304391: "AZ",
        5306611: "AZ", 5307540: "AZ", 5308305: "AZ", 5308480: "AZ",
        5308655: "AZ", 5309842: "AZ", 5309858: "AZ", 5310193: "AZ",
        5311433: "AZ", 5312544: "AZ", 5312913: "AZ", 5313457: "AZ",
        5314328: "AZ", 5315062: "AZ", 5316201: "AZ", 5316205: "AZ",
        5316428: "AZ", 5316890: "AZ", 5317058: "AZ", 5317071: "AZ",
        5318313: "AZ", 5322053: "AZ", 5322400: "CA", 5322551: "CA",
        5322553: "CA", 5322737: "CA", 5322850: "CA", 5323060: "CA",
        5323163: "CA", 5323525: "CA", 5323566: "CA", 5323694: "CA",
        5323810: "CA", 5324105: "CA", 5324200: "CA", 5324363: "CA",
        5324477: "CA", 5324802: "CA", 5324862: "CA", 5324903: "CA",
        5325011: "CA", 5325111: "CA", 5325187: "CA", 5325372: "CA",
        5325423: "CA", 5325738: "CA", 5325866: "CA", 5326032: "CA",
        5326297: "CA", 5326305: "CA", 5326561: "CA", 5327098: "CA",
        5327298: "CA", 5327319: "CA", 5327422: "CA", 5327455: "CA",
        5327550: "CA", 5327684: "CA", 5328041: "CA", 5329408: "CA",
        5329649: "CA", 5330167: "CA", 5330413: "CA", 5330567: "CA",
        5330582: "CA", 5330642: "CA", 5331575: "CA", 5331835: "CA",
        5331920: "CA", 5332593: "CA", 5332698: "CA", 5333180: "CA",
        5333282: "CA", 5333689: "CA", 5333913: "CA", 5333944: "CA",
        5334223: "CA", 5334336: "CA", 5334519: "CA", 5334799: "CA",
        5334928: "CA", 5335006: "CA", 5335650: "CA", 5335663: "CA",
        5336054: "CA", 5336269: "CA", 5336477: "CA", 5336537: "CA",
        5336545: "CA", 5336667: "CA", 5336899: "CA", 5337561: "CA",
        5337696: "CA", 5337908: "CA", 5338122: "CA", 5338166: "CA",
        5338196: "CA", 5338783: "CA", 5339066: "CA", 5339111: "CA",
        5339539: "CA", 5339631: "CA", 5339663: "CA", 5339840: "CA",
        5340175: "CA", 5341051: "CA", 5341114: "CA", 5341145: "CA",
        5341256: "CA", 5341430: "CA", 5341483: "CA", 5341531: "CA",
        5341704: "CA", 5342485: "CA", 5342710: "CA", 5342992: "CA",
        5343171: "CA", 5343303: "CA", 5343858: "CA", 5344147: "CA",
        5344157: "CA", 5344817: "CA", 5344942: "CA", 5344994: "CA",
        5345032: "CA", 5345529: "CA", 5345609: "CA", 5345623: "CA",
        5345679: "CA", 5345743: "CA", 5345860: "CA", 5346111: "CA",
        5346646: "CA", 5346649: "CA", 5346827: "CA", 5347287: "CA",
        5347335: "CA", 5347578: "CA", 5349613: "CA", 5349705: "CA",
        5349755: "CA", 5349803: "CA", 5350159: "CA", 5350207: "CA",
        5350734: "CA", 5350937: "CA", 5351247: "CA", 5351428: "CA",
        5351515: "CA", 5351549: "CA", 5352214: "CA", 5352350: "CA",
        5352423: "CA", 5352439: "CA", 5352963: "CA", 5353530: "CA",
        5354172: "CA", 5354819: "CA", 5355180: "CA", 5355828: "CA",
        5355933: "CA", 5356277: "CA", 5356451: "CA", 5356521: "CA",
        5356576: "CA", 5356868: "CA", 5357499: "CA", 5357527: "CA",
        5358705: "CA", 5358736: "CA", 5359052: "CA", 5359054: "CA",
        5359446: "CA", 5359488: "CA", 5359777: "CA", 5359864: "CA",
        5363748: "CA", 5363859: "CA", 5363922: "CA", 5363943: "CA",
        5363990: "CA", 5364007: "CA", 5364022: "CA", 5364059: "CA",
        5364066: "CA", 5364079: "CA", 5364134: "CA", 5364199: "CA",
        5364226: "CA", 5364271: "CA", 5364275: "CA", 5364306: "CA",
        5364329: "CA", 5364369: "CA", 5364499: "CA", 5364514: "CA",
        5364782: "CA", 5364855: "CA", 5364916: "CA", 5364940: "CA",
        5365425: "CA", 5365603: "CA", 5365893: "CA", 5365918: "CA",
        5365937: "CA", 5366375: "CA", 5366531: "CA", 5367314: "CA",
        5367440: "CA", 5367565: "CA", 5367696: "CA", 5367767: "CA",
        5367788: "CA", 5367929: "CA", 5368335: "CA", 5368361: "CA",
        5368453: "CA", 5368518: "CA", 5369367: "CA", 5369568: "CA",
        5370082: "CA", 5370164: "CA", 5370493: "CA", 5370868: "CA",
        5371261: "CA", 5371858: "CA", 5372205: "CA", 5372223: "CA",
        5372253: "CA", 5373129: "CA", 5373327: "CA", 5373492: "CA",
        5373497: "CA", 5373628: "CA", 5373763: "CA", 5373900: "CA",
        5374175: "CA", 5374232: "CA", 5374322: "CA", 5374361: "CA",
        5374406: "CA", 5374648: "CA", 5374671: "CA", 5374732: "CA",
        5374764: "CA", 5375480: "CA", 5375911: "CA", 5376095: "CA",
        5376200: "CA", 5376803: "CA", 5376890: "CA", 5377100: "CA",
        5377199: "CA", 5377640: "CA", 5377654: "CA", 5377985: "CA",
        5377995: "CA", 5378044: "CA", 5378500: "CA", 5378538: "CA",
        5378566: "CA", 5378771: "CA", 5378870: "CA", 5379439: "CA",
        5379513: "CA", 5379566: "CA", 5379609: "CA", 5379678: "CA",
        5379759: "CA", 5380184: "CA", 5380420: "CA", 5380437: "CA",
        5380626: "CA", 5380668: "CA", 5380698: "CA", 5380748: "CA",
        5381002: "CA", 5381110: "CA", 5381325: "CA", 5381396: "CA",
        5381438: "CA", 5381515: "CA", 5382146: "CA", 5382232: "CA",
        5382496: "CA", 5383187: "CA", 5383465: "CA", 5383527: "CA",
        5383720: "CA", 5383777: "CA", 5384170: "CA", 5384339: "CA",
        5384471: "CA", 5384690: "CA", 5385082: "CA", 5385793: "CA",
        5385941: "CA", 5385955: "CA", 5386015: "CA", 5386035: "CA",
        5386039: "CA", 5386053: "CA", 5386082: "CA", 5386754: "CA",
        5386785: "CA", 5386834: "CA", 5386984: "CA", 5387288: "CA",
        5387428: "CA", 5387494: "CA", 5387687: "CA", 5387749: "CA",
        5387844: "CA", 5387877: "CA", 5388319: "CA", 5388564: "CA",
        5388735: "CA", 5388867: "CA", 5388873: "CA", 5388881: "CA",
        5389126: "CA", 5389213: "CA", 5389489: "CA", 5391295: "CA",
        5391710: "CA", 5391749: "CA", 5391760: "CA", 5391791: "CA",
        5391811: "CA", 5391891: "CA", 5391945: "CA", 5391959: "CA",
        5392034: "CA", 5392090: "CA", 5392171: "CA", 5392229: "CA",
        5392263: "CA", 5392281: "CA", 5392323: "CA", 5392368: "CA",
        5392423: "CA", 5392508: "CA", 5392528: "CA", 5392567: "CA",
        5392593: "CA", 5392868: "CA", 5392900: "CA", 5392952: "CA",
        5393015: "CA", 5393049: "CA", 5393052: "CA", 5393128: "CA",
        5393180: "CA", 5393212: "CA", 5393245: "CA", 5393287: "CA",
        5393429: "CA", 5393485: "CA", 5394086: "CA", 5394136: "CA",
        5394329: "CA", 5394409: "CA", 5394842: "CA", 5395244: "CA",
        5396003: "CA", 5397018: "CA", 5397376: "CA", 5397603: "CA",
        5397664: "CA", 5397717: "CA", 5397765: "CA", 5397777: "CA",
        5397841: "CA", 5397851: "CA", 5398277: "CA", 5398630: "CA",
        5399020: "CA", 5399438: "CA", 5399629: "CA", 5399901: "CA",
        5399976: "CA", 5400075: "CA", 5401395: "CA", 5401469: "CA",
        5402405: "CA", 5403022: "CA", 5403191: "CA", 5403676: "CA",
        5403767: "CA", 5403783: "CA", 5404024: "CA", 5404119: "CA",
        5404122: "CA", 5404198: "CA", 5404476: "CA", 5404555: "CA",
        5404794: "CA", 5404915: "CA", 5405228: "CA", 5405288: "CA",
        5405326: "CA", 5405380: "CA", 5405693: "CA", 5405841: "CA",
        5405878: "CA", 5406222: "CA", 5406421: "CA", 5406567: "CA",
        5406602: "CA", 5406976: "CA", 5406990: "CA", 5407030: "CA",
        5407529: "CA", 5407908: "CA", 5407933: "CA", 5408076: "CA",
        5408191: "CA", 5408211: "CA", 5408406: "CA", 5408431: "CA",
        5409059: "CA", 5409260: "CA", 5409768: "CA", 5410004: "CA",
        5410129: "CA", 5410430: "CA", 5410438: "CA", 5410902: "CA",
        5411015: "CA", 5411046: "CA", 5411079: "CA", 5412199: "CO",
        5412347: "CO", 5414941: "CO", 5415035: "CO", 5416005: "CO",
        5416329: "CO", 5416357: "CO", 5416541: "CO", 5417041: "CO",
        5417258: "CO", 5417598: "CO", 5417657: "CO", 5417737: "CO",
        5419384: "CO", 5420241: "CO", 5421250: "CO", 5422191: "CO",
        5423294: "CO", 5423573: "CO", 5423908: "CO", 5425043: "CO",
        5427207: "CO", 5427771: "CO", 5427946: "CO", 5429032: "CO",
        5429522: "CO", 5431710: "CO", 5433124: "CO", 5434006: "CO",
        5435464: "CO", 5435477: "CO", 5438567: "CO", 5439752: "CO",
        5441492: "CO", 5443910: "CO", 5443948: "CO", 5445298: "KS",
        5445439: "KS", 5445820: "KS", 5454627: "NM", 5454711: "NM",
        5460459: "NM", 5462393: "NM", 5467328: "NM", 5468773: "NM",
        5471578: "NM", 5475352: "NM", 5476913: "NM", 5487811: "NM",
        5488441: "NM", 5490263: "NM", 5492450: "NM", 5493403: "NM",
        5500539: "NV", 5501344: "NV", 5503766: "NV", 5504003: "NV",
        5505411: "NV", 5506956: "NV", 5508180: "NV", 5509403: "NV",
        5509851: "NV", 5509952: "NV", 5511077: "NV", 5512827: "NV",
        5512862: "NV", 5512909: "NV", 5513307: "NV", 5513343: "NV",
        5515110: "NV", 5515345: "NV", 5516233: "TX", 5517061: "TX",
        5520076: "TX", 5520552: "TX", 5520677: "TX", 5520993: "TX",
        5523074: "TX", 5523369: "TX", 5525577: "TX", 5526337: "TX",
        5527554: "TX", 5527953: "TX", 5528450: "TX", 5530022: "TX",
        5530932: "TX", 5530937: "TX", 5533366: "TX", 5536630: "UT",
        5540831: "UT", 5546220: "UT", 5549222: "UT", 5550368: "CA",
        5551123: "AZ", 5551498: "AZ", 5551535: "AZ", 5552301: "AZ",
        5552450: "AZ", 5554072: "AK", 5558953: "CA", 5559320: "CA",
        5563397: "CA", 5567770: "CA", 5570160: "CA", 5572400: "CA",
        5574991: "CO", 5576859: "CO", 5576909: "CO", 5577147: "CO",
        5577592: "CO", 5579276: "CO", 5579368: "CO", 5583509: "CO",
        5586437: "ID", 5587698: "ID", 5589173: "ID", 5591778: "ID",
        5596475: "ID", 5597955: "ID", 5598538: "ID", 5598542: "ID",
        5600685: "ID", 5601538: "ID", 5601933: "ID", 5604045: "ID",
        5604353: "ID", 5605242: "ID", 5610810: "ID", 5640350: "MT",
        5641727: "MT", 5642934: "MT", 5655240: "MT", 5656882: "MT",
        5660340: "MT", 5666639: "MT", 5688025: "ND", 5688789: "ND",
        5690366: "ND", 5690532: "ND", 5692947: "ND", 5697939: "NE",
        5703670: "NV", 5710756: "OR", 5711099: "OR", 5711149: "OR",
        5711789: "OR", 5713376: "OR", 5713587: "OR", 5713759: "OR",
        5717758: "OR", 5718601: "OR", 5720495: "OR", 5720727: "OR",
        5722064: "OR", 5725846: "OR", 5727190: "OR", 5727382: "OR",
        5729080: "OR", 5729485: "OR", 5730183: "OR", 5730675: "OR",
        5731070: "OR", 5731371: "OR", 5734711: "OR", 5735238: "OR",
        5735724: "OR", 5736218: "OR", 5736378: "OR", 5739936: "OR",
        5740099: "OR", 5740900: "OR", 5742726: "OR", 5743731: "OR",
        5744253: "OR", 5745380: "OR", 5746545: "OR", 5747882: "OR",
        5749352: "OR", 5750162: "OR", 5751632: "OR", 5754005: "OR",
        5756304: "OR", 5756758: "OR", 5757477: "OR", 5757506: "OR",
        5760009: "OR", 5761287: "OR", 5761708: "OR", 5768233: "SD",
        5771826: "UT", 5771960: "UT", 5772654: "UT", 5772959: "UT",
        5773001: "UT", 5773304: "UT", 5774001: "UT", 5774215: "UT",
        5774301: "UT", 5774662: "UT", 5775782: "UT", 5775863: "UT",
        5776008: "UT", 5776715: "UT", 5776727: "UT", 5777107: "UT",
        5777224: "UT", 5777544: "UT", 5777793: "UT", 5778244: "UT",
        5778352: "UT", 5778755: "UT", 5779036: "UT", 5779068: "UT",
        5779206: "UT", 5779334: "UT", 5779548: "UT", 5779816: "UT",
        5780026: "UT", 5780557: "UT", 5780802: "UT", 5780993: "UT",
        5781061: "UT", 5781070: "UT", 5781087: "UT", 5781765: "UT",
        5781770: "UT", 5781783: "UT", 5781794: "UT", 5781860: "UT",
        5781993: "UT", 5782391: "UT", 5782476: "UT", 5783695: "UT",
        5784549: "UT", 5784607: "UT", 5785243: "WA", 5785657: "WA",
        5785868: "WA", 5785965: "WA", 5786485: "WA", 5786882: "WA",
        5786899: "WA", 5787776: "WA", 5787829: "WA", 5788054: "WA",
        5788516: "WA", 5788822: "WA", 5789683: "WA", 5790971: "WA",
        5791159: "WA", 5792244: "WA", 5793427: "WA", 5793639: "WA",
        5793933: "WA", 5794097: "WA", 5794245: "WA", 5794559: "WA",
        5795011: "WA", 5795906: "WA", 5796984: "WA", 5798487: "WA",
        5799587: "WA", 5799610: "WA", 5799625: "WA", 5799841: "WA",
        5800112: "WA", 5800317: "WA", 5800420: "WA", 5801617: "WA",
        5802049: "WA", 5802340: "WA", 5802493: "WA", 5802570: "WA",
        5803139: "WA", 5803457: "WA", 5803786: "WA", 5803990: "WA",
        5804127: "WA", 5804191: "WA", 5804306: "WA", 5804953: "WA",
        5805441: "WA", 5805687: "WA", 5805782: "WA", 5805815: "WA",
        5806253: "WA", 5806298: "WA", 5807212: "WA", 5807540: "WA",
        5807575: "WA", 5808079: "WA", 5808189: "WA", 5808276: "WA",
        5809333: "WA", 5809402: "WA", 5809805: "WA", 5809844: "WA",
        5810301: "WA", 5810490: "WA", 5811456: "WA", 5811581: "WA",
        5811696: "WA", 5811729: "WA", 5812604: "WA", 5812944: "WA",
        5814043: "WA", 5814095: "WA", 5814450: "WA", 5814616: "WA",
        5814916: "WA", 5815136: "WA", 5815342: "WA", 5815538: "WA",
        5815539: "WA", 5816320: "WA", 5816605: "WA", 5820705: "WY",
        5821086: "WY", 5826027: "WY", 5830062: "WY", 5836898: "WY",
        5838198: "WY", 5844096: "UT", 5847411: "HI", 5847486: "HI",
        5848189: "HI", 5849297: "HI", 5850554: "HI", 5851030: "HI",
        5852275: "HI", 5853992: "HI", 5854496: "HI", 5854686: "HI",
        5855070: "HI", 5855927: "HI", 5856195: "HI", 5861187: "AK",
        5861897: "AK", 5879400: "AK", 5879898: "AK", 6331908: "GA",
        6331909: "GA", 6332309: "FL", 6332422: "NY", 6332428: "NY",
        6332439: "FL", 6640032: "NY", 6690773: "CA", 6941080: "NC",
        6957263: "HI", 7160204: "CT", 7174365: "WA", 7176035: "WA",
        7176039: "CA", 7184841: "NY", 7250691: "NY", 7257422: "CT",
        7257540: "MA", 7257726: "MA", 7257991: "MD", 7257992: "MD",
        7258271: "NJ", 7258432: "MD", 7258633: "NY", 7258671: "MD",
        7258780: "VA", 7258832: "MD", 7258897: "MD", 7258965: "NY",
        7259034: "MD", 7259061: "MD", 7259265: "MO", 7259480: "MI",
        7259621: "MI", 7259705: "FL", 7259780: "FL", 7259823: "FL",
        7260019: "GA", 7260095: "FL", 7260219: "FL", 7260233: "FL",
        7260327: "FL", 7260548: "FL", 7260806: "CA", 7260966: "WA",
        7261029: "CA", 7261268: "CA", 7261291: "TX", 7261476: "WA",
        7261553: "CA", 7261759: "WA", 7262110: "WA", 7262349: "CO",
        7262428: "WA", 7262464: "CA", 7262518: "CA", 7262622: "NV",
        7262761: "HI", 7262790: "HI", 7310164: "AZ", 7315234: "FL",
        7315237: "FL", 7315245: "HI", 7315274: "MO", 7315409: "WA",
        7315412: "WA", 7315418: "WA", 7839240: "NM", 8030162: "CA",
        8062662: "OH", 8062667: "OH", 8096113: "FL", 8096217: "CA",
        8097131: "VA", 8299576: "NJ", 8299577: "NJ", 8347326: "IN",
        8436065: "IL", 8436083: "IL", 8436084: "IL", 8436473: "NY",
        8436486: "NY", 8436572: "RI", 8449772: "CA", 8449777: "CA",
        8449910: "IN", 8469295: "NJ", 8478941: "WI", 8479411: "FL",
        8480031: "CT", 8480062: "VA", 8531960: "RI", 8604682: "RI",
        8605040: "AR", 8605041: "AL", 8643098: "PA", 9958118: "PA",
        10104153:"CA", 10104154:"CA", 11704266:"FL", 11748973:"GA",
        11811570:"NY", 11838960:"CA", 11979227:"AZ", 11979238:"AZ",
    }
    return dct

