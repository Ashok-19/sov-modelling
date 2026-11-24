"""Loading the spreadsheet, tidying topic names and removing exact duplicates."""

import re

import pandas as pd

from . import config


def clean_topic(name):
    if not isinstance(name, str) or not name.strip():
        return None
    name = re.sub(r"\s+", " ", name.strip())
    return config.TOPIC_ALIASES.get(name, name)


def topic_number(name):
    m = re.match(r"\s*(\d+)\.", name or "")
    return int(m.group(1)) if m else None


def load(path=config.DATA_FILE):
    df = pd.read_excel(path, sheet_name=config.SHEET)
    df["topic_clean"] = df[config.TOPIC_COL].map(clean_topic)
    df["topic_num"] = df["topic_clean"].map(topic_number)
    return df


def list_topics(df):
    """Topics found in the data, as {number: name}, skipping the noise bucket."""
    topics = {}
    for name in df["topic_clean"].dropna().unique():
        if any(w in name.lower() for w in config.SKIP_TOPIC_WORDS):
            continue
        num = topic_number(name)
        if num is None:
            continue
        topics[num] = name
    return dict(sorted(topics.items()))


def dedupe(df):
    """Drop posts whose text is an exact copy of an earlier post in the same topic.

    Returns (kept, duplicates). The duplicates frame has a `duplicate_of`
    column pointing at the Post_ID we kept, so they can be counted back in later.
    """
    text = df[config.TEXT_COL].fillna("").astype(str)
    key = df["topic_clean"].fillna("") + "\x00" + text
    first_id = df.groupby(key)[config.ID_COL].transform("first")
    is_dup = key.duplicated(keep="first")
    dups = df[is_dup].copy()
    dups["duplicate_of"] = first_id[is_dup]
    return df[~is_dup].copy(), dups
