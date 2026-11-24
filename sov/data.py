"""Loading the spreadsheet and tidying topic names."""

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


def load(path=config.DATA_FILE, sheet=config.SHEET):
    """Read the posts from an Excel file (.xlsx/.xls) or a CSV."""
    path = str(path)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet)
    missing = [c for c in (config.ID_COL, config.TEXT_COL, config.TOPIC_COL) if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing column(s) {missing}. It needs the same columns "
                         f"as the original sheet: {config.ID_COL}, {config.TEXT_COL}, {config.TOPIC_COL}.")
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

