import re

import numpy as np
from lib.utils.constants import (
    COUNTRY_DOMAIN_LIST,
    SUBDOMAIN_LIST,
    TOP_LEVEL_DOMAIN_LIST,
)
from newspaper import Article
from newspaper.article import ArticleException


def extract_article_artifacts(url, language="en", null_val=np.nan):
    try:
        doc = Article(url, language=language)
        doc.download()
        doc.parse()
        doc.nlp()
        return {
            "DocAuthors": doc.authors,
            "DocPubDate": doc.publish_date,
            "DocText": doc.text,
            "DocKeywords": doc.keywords,
        }
    except ArticleException:
        return {
            "DocAuthors": null_val,
            "DocPubDate": null_val,
            "DocText": null_val,
            "DocKeywords": null_val,
        }


def extract_host(url):
    host = np.nan
    url = re.sub("HTTPS?://", "", url)  # Mask scheme
    url = re.sub("(?:\.COM|\.NET):\d{,3}", "", url)  # Mask ports
    pattern = re.compile(r"(?:WWW\.)?([^/?]+)")
    match = pattern.search(url)
    if match:
        full_domain = match.group(1)
        excluded_domains = TOP_LEVEL_DOMAIN_LIST + COUNTRY_DOMAIN_LIST + SUBDOMAIN_LIST
        domain = [
            domain
            for domain in full_domain.split(".")
            if domain not in excluded_domains
        ]
        if domain:
            host = domain[-1]
        else:
            stopwords = ["ENGLISH", "COM"]  # Based on manual inspection
            host = "-".join(
                [domain for domain in full_domain.split(".") if domain not in stopwords]
            )

    return host


# Note: null if url is similar to HTTP://GHORBANEWS.COM/NEWSDETAILS.ASPX?ID=56913
def extract_headline(url):
    headline = np.nan
    path_segments = url.split("/")
    path_segments = [
        path_segment
        for path_segment in path_segments
        if path_segment not in ["EN", "NEWS"]
    ]
    token_counts = [path_segment.count("-") for path_segment in path_segments]
    if max(token_counts) > 0:
        headline = path_segments[token_counts.index(max(token_counts))]
        headline = headline.replace("-", " ")
    else:
        headline = path_segments[-1]
    return headline
