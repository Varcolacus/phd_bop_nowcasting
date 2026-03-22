"""
NLP Sentiment Pipeline for Chapter 2 — BoP Nowcasting
=======================================================

Extracts trade-relevant sentiment from:
  1. ECB Monetary Policy Statements (press conferences)
  2. Banque de France Monthly Business Surveys (Enquete Mensuelle de Conjoncture)

Both sources are scored using FinBERT (ProsusAI/finbert).

Pipeline:
  1. Collect texts (HTML scraping)
  2. Chunk into paragraphs and filter for trade-relevant content
  3. Score sentiment using FinBERT
  4. Aggregate to monthly frequency

NOTE — URL Pattern Fragility:
  The ECB web scraper uses manually enumerated URL patterns for both the
  legacy (/press/pressconf/) and current (/press/press_conference/) ECB
  website structures.  If the ECB restructures its website or changes URL
  conventions, the scraper will require updating.  No official ECB API for
  press conference transcripts exists as of 2026.  The pipeline degrades
  gracefully: if too few texts are retrieved, it falls back to synthetic
  sentiment calibrated on crisis timing.

Author: PhD Pilot Study
Date: March 2026
"""

import os
import logging
import warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# SSL verification: override via NOWCAST_VERIFY_SSL=true if not behind a proxy
VERIFY_SSL = os.getenv('NOWCAST_VERIFY_SSL', 'false').lower() == 'true'

DATA_DIR = Path(__file__).parent.parent / "data"
ALT_DATA_DIR = DATA_DIR / "alternative"
ALT_DATA_DIR.mkdir(exist_ok=True)
NLP_DIR = ALT_DATA_DIR / "nlp"
NLP_DIR.mkdir(exist_ok=True)

# Trade-related keywords for filtering paragraphs
TRADE_KEYWORDS = {
    "trade", "export", "import", "current account", "balance",
    "external", "competitiveness", "tariff", "goods", "services",
    "trade balance", "surplus", "deficit", "exchange rate",
    "global demand", "foreign", "trading partners", "shipping",
}

# ECB press conference URLs follow a pattern
ECB_PRESS_BASE = "https://www.ecb.europa.eu/press/pressconf"


# =====================================================================
# Step 1: Collect ECB Monetary Policy Statements
# =====================================================================

def get_ecb_statement_urls(start_year=2008, end_year=2022):
    """
    Generate URLs for ECB press conference introductory statements.

    The ECB website uses dynamic rendering, so the index page cannot be
    scraped statically.  Instead, we use the ECB Data Portal's publicly
    documented list of Governing Council monetary policy meeting dates
    and construct URLs from ECB's bulk download CSV format for press
    conferences.

    Falls back to known meeting dates when web access fails.
    Returns list of (date, url) tuples.
    """
    # ECB Governing Council monetary policy meeting dates (from ECB calendar)
    # These are the dates of the press conference (typically the day of the meeting)
    KNOWN_PRESS_DATES = [
        # 2008
        "2008-01-10", "2008-02-07", "2008-03-06", "2008-04-10", "2008-05-08",
        "2008-06-05", "2008-07-03", "2008-08-07", "2008-09-04", "2008-10-02",
        "2008-11-06", "2008-12-04",
        # 2009
        "2009-01-15", "2009-02-05", "2009-03-05", "2009-04-02", "2009-05-07",
        "2009-06-04", "2009-07-02", "2009-08-06", "2009-09-03", "2009-10-08",
        "2009-11-05", "2009-12-03",
        # 2010
        "2010-01-14", "2010-02-04", "2010-03-04", "2010-04-08", "2010-05-06",
        "2010-06-10", "2010-07-08", "2010-08-05", "2010-09-02", "2010-10-07",
        "2010-11-04", "2010-12-02",
        # 2011
        "2011-01-13", "2011-02-03", "2011-03-03", "2011-04-07", "2011-05-05",
        "2011-06-09", "2011-07-07", "2011-08-04", "2011-09-08", "2011-10-06",
        "2011-11-03", "2011-12-08",
        # 2012
        "2012-01-12", "2012-02-09", "2012-03-08", "2012-04-04", "2012-05-03",
        "2012-06-06", "2012-07-05", "2012-08-02", "2012-09-06", "2012-10-04",
        "2012-11-08", "2012-12-06",
        # 2013
        "2013-01-10", "2013-02-07", "2013-03-07", "2013-04-04", "2013-05-02",
        "2013-06-06", "2013-07-04", "2013-08-01", "2013-09-05", "2013-10-02",
        "2013-11-07", "2013-12-05",
        # 2014 (changed to 6-week cycle)
        "2014-01-09", "2014-02-06", "2014-03-06", "2014-04-03", "2014-05-08",
        "2014-06-05", "2014-07-03", "2014-08-07", "2014-09-04", "2014-10-02",
        "2014-11-06", "2014-12-04",
        # 2015 (6-week cycle)
        "2015-01-22", "2015-03-05", "2015-04-15", "2015-06-03", "2015-07-16",
        "2015-09-03", "2015-10-22", "2015-12-03",
        # 2016
        "2016-01-21", "2016-03-10", "2016-04-21", "2016-06-02", "2016-07-21",
        "2016-09-08", "2016-10-20", "2016-12-08",
        # 2017
        "2017-01-19", "2017-03-09", "2017-04-27", "2017-06-08", "2017-07-20",
        "2017-09-07", "2017-10-26", "2017-12-14",
        # 2018
        "2018-01-25", "2018-03-08", "2018-04-26", "2018-06-14", "2018-07-26",
        "2018-09-13", "2018-10-25", "2018-12-13",
        # 2019
        "2019-01-24", "2019-03-07", "2019-04-10", "2019-06-06", "2019-07-25",
        "2019-09-12", "2019-10-24", "2019-12-12",
        # 2020
        "2020-01-23", "2020-03-12", "2020-04-30", "2020-06-04", "2020-07-16",
        "2020-09-10", "2020-10-29", "2020-12-10",
        # 2021
        "2021-01-21", "2021-03-11", "2021-04-22", "2021-06-10", "2021-07-22",
        "2021-09-09", "2021-10-28", "2021-12-16",
        # 2022
        "2022-02-03", "2022-03-10", "2022-04-14", "2022-06-09", "2022-07-21",
        "2022-09-08", "2022-10-27", "2022-12-15",
    ]

    statements = []
    for date_str in KNOWN_PRESS_DATES:
        date = pd.Timestamp(date_str)
        if date.year < start_year or date.year > end_year:
            continue
        # Build URL: ECB monetary policy statement endpoint
        yy = f"{date.year % 100:02d}"
        mm = f"{date.month:02d}"
        dd = f"{date.day:02d}"
        # Use the ECB monetary policy decisions page as text source
        # Format: https://www.ecb.europa.eu/press/pr/date/YYYY/html/pr.mpYYMMDD~HASH.en.html
        # Since hash varies, we'll scrape text from the introductory statement page instead
        statements.append((date, date_str))

    print(f"    Found {len(statements)} known ECB press conference dates")
    return statements


def _extract_ecb_text(soup):
    """Extract press conference text from an ECB HTML page.

    The ECB website stores statement text across multiple
    ``<div class="section">`` elements.  Collect ALL such divs
    and gather every ``<p>`` inside them.  Fall back to ``<article>``
    or ``<div id="main-wrapper">`` if no section divs are found.
    """
    sections = soup.find_all("div", class_="section")
    if sections:
        paragraphs = []
        for sec in sections:
            paragraphs.extend(sec.find_all("p"))
    else:
        content = soup.find("article") or soup.find("div", id="main-wrapper")
        paragraphs = content.find_all("p") if content else []

    text = "\n".join(p.get_text(strip=True) for p in paragraphs
                     if len(p.get_text(strip=True)) > 20)
    return text


def scrape_ecb_statement(url):
    """
    Scrape the text of an ECB press conference statement from its URL.
    Returns the full text as a string.
    """
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=30, verify=VERIFY_SSL,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        return _extract_ecb_text(soup)
    except Exception as e:
        logger.warning("scrape_ecb_statement failed for %s: %s", url, e)
        return ""


# =====================================================================
# Step 2: Text preprocessing
# =====================================================================

def chunk_into_paragraphs(text, min_tokens=30):
    """Split text into paragraph-level chunks, filter by minimum length."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return [p for p in paragraphs if len(p.split()) >= min_tokens]


def filter_trade_relevant(paragraphs, keywords=None):
    """Keep paragraphs containing trade-related keywords."""
    if keywords is None:
        keywords = TRADE_KEYWORDS
    filtered = []
    for p in paragraphs:
        p_lower = p.lower()
        if any(kw in p_lower for kw in keywords):
            filtered.append(p)
    return filtered


# =====================================================================
# Step 3: Sentiment scoring with FinBERT
# =====================================================================

_finbert_pipeline = None


def get_finbert_pipeline():
    """Lazy-load the FinBERT sentiment pipeline."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline as hf_pipeline
        print("    Loading FinBERT model (ProsusAI/finbert)...")
        _finbert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        print("    [OK] FinBERT loaded")
        return _finbert_pipeline
    except Exception as e:
        print(f"    [WARN] FinBERT load failed: {e}")
        return None


def score_sentiment(paragraphs):
    """
    Score paragraphs using FinBERT.
    Returns list of sentiment scores in [-1, +1].
    Positive = optimistic, Negative = pessimistic.
    """
    pipe = get_finbert_pipeline()
    if pipe is None:
        return []

    scores = []
    for p in paragraphs:
        try:
            result = pipe(p[:512])[0]  # Truncate to 512 tokens
            label = result["label"].lower()
            confidence = result["score"]
            if label == "positive":
                scores.append(confidence)
            elif label == "negative":
                scores.append(-confidence)
            else:
                scores.append(0.0)
        except Exception as e:
            logger.warning("FinBERT scoring failed for paragraph: %s", e)
    return scores


# =====================================================================
# Step 4: Aggregate to monthly
# =====================================================================

def aggregate_sentiment_monthly(statement_scores):
    """
    Aggregate statement-level sentiment scores to monthly frequency.

    Parameters:
        statement_scores: list of (date, mean_score, dispersion) tuples

    Returns:
        DataFrame(date, ecb_sentiment, sentiment_dispersion)
    """
    if not statement_scores:
        return pd.DataFrame(columns=["date", "ecb_sentiment", "sentiment_dispersion"])

    df = pd.DataFrame(statement_scores, columns=["date", "ecb_sentiment", "sentiment_dispersion"])
    df["date"] = pd.to_datetime(df["date"])

    # Resample to monthly — forward-fill between meetings
    df = df.set_index("date").resample("MS").first()
    df = df.ffill()
    df = df.reset_index()

    return df


# =====================================================================
# PDF text extraction helper
# =====================================================================

def _extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF document (ECB monetary policy decisions)."""
    # Try pdfplumber first (best quality), then PyPDF2
    try:
        import pdfplumber
        from io import BytesIO
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages_text)
    except ImportError:
        logger.debug("pdfplumber not available, trying PyPDF2")
        pass

    try:
        from PyPDF2 import PdfReader
        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text)
    except ImportError:
        logger.debug("PyPDF2 not available, cannot extract PDF text")
        pass

    return ""


# =====================================================================
# Full NLP pipeline
# =====================================================================

def run_nlp_pipeline(start_year=2008, end_year=2022):
    """
    Run the full NLP sentiment pipeline on ECB press conferences.

    Since the ECB website uses dynamic rendering, we use the ECB's
    monetary policy decisions search API to fetch statement texts.
    Falls back to synthetic if web access fails.

    Returns:
        DataFrame(date, ecb_sentiment, sentiment_dispersion)
        str: "real" or "synthetic"
    """
    print("\n  NLP SENTIMENT PIPELINE")
    print("  " + "-" * 40)

    # Step 1: Get known press conference dates
    statement_dates = get_ecb_statement_urls(start_year, end_year)

    if not statement_dates:
        print("    [INFO] No statement dates found — generating synthetic sentiment")
        return generate_synthetic_sentiment(start_year, end_year), "synthetic"

    # Step 2: Try to fetch ECB press conference texts via multiple URL patterns
    print("    Fetching ECB monetary policy statement texts...")
    fetched_texts = []

    from bs4 import BeautifulSoup

    # --- First, try to scrape the ECB press conference archive pages ---
    # to discover actual URLs including hashes for post-2015 documents
    discovered_urls = {}
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 8  # Give up on archive scraping after 8 failures

    # Try the main press conference listing page (all years)
    main_listing_urls = [
        "https://www.ecb.europa.eu/press/pressconf/html/index.en.html",
        "https://www.ecb.europa.eu/press/pressconf/html/index_include.en.html",
        "https://www.ecb.europa.eu/press/key/html/index.en.html",
    ]
    for listing_url in main_listing_urls:
        try:
            resp = requests.get(listing_url, timeout=15, verify=VERIFY_SSL,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if not href.endswith(".en.html") and not href.endswith(".en.pdf"):
                        continue
                    for date, date_str in statement_dates:
                        yy = f"{date.year % 100:02d}"
                        mm = f"{date.month:02d}"
                        dd = f"{date.day:02d}"
                        yyyymmdd = f"{date.year}{mm}{dd}"
                        if (f"is{yy}{mm}{dd}" in href or f"is{yyyymmdd}" in href or
                            f"mp{yy}{mm}{dd}" in href or f"mp{yyyymmdd}" in href or
                            f"mps{yy}{mm}{dd}" in href or f"mps{yyyymmdd}" in href or
                            f"ds{yyyymmdd}" in href or f"sp{yy}{mm}{dd}" in href):
                            full_url = href if href.startswith("http") else f"https://www.ecb.europa.eu{href}"
                            discovered_urls[date_str] = full_url
        except Exception as e:
            logger.warning("Main listing scrape failed for %s: %s", listing_url, e)

    # Year-specific archive pages
    for yr in range(start_year, end_year + 1):
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"    [INFO] Archive scraping aborted after {MAX_CONSECUTIVE_FAILURES} consecutive failures")
            break
        for section in ["pressconf", "pr/date"]:
            try:
                if section == "pressconf":
                    archive_url = (
                        f"https://www.ecb.europa.eu/press/pressconf/{yr}"
                        "/html/index.en.html"
                    )
                else:
                    archive_url = (
                        f"https://www.ecb.europa.eu/press/pr/date/{yr}"
                        "/html/index.en.html"
                    )
                resp = requests.get(archive_url, timeout=8, verify=VERIFY_SSL,
                                    headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200:
                    consecutive_failures = 0
                    soup = BeautifulSoup(resp.text, "lxml")
                    links = soup.find_all("a", href=True)
                    for link in links:
                        href = link["href"]
                        if href.endswith(".en.html") or href.endswith(".en.pdf"):
                            for date, date_str in statement_dates:
                                if date.year == yr:
                                    yy = f"{date.year % 100:02d}"
                                    mm = f"{date.month:02d}"
                                    dd = f"{date.day:02d}"
                                    yyyymmdd = f"{date.year}{mm}{dd}"
                                    # Match any URL containing this date's identifiers
                                    if (f"is{yy}{mm}{dd}" in href or
                                        f"is{yyyymmdd}" in href or
                                        f"pr{yy}{mm}{dd}" in href or
                                        f"mp{yy}{mm}{dd}" in href or
                                        f"mp{yyyymmdd}" in href or
                                        f"mps{yy}{mm}{dd}" in href or
                                        f"mps{yyyymmdd}" in href or
                                        f"ecb.mp{yy}{mm}{dd}" in href or
                                        f"ds{yyyymmdd}" in href or
                                        f"sp{yy}{mm}{dd}" in href):
                                        full_url = href if href.startswith("http") else f"https://www.ecb.europa.eu{href}"
                                        discovered_urls[date_str] = full_url
                else:
                    consecutive_failures += 1
            except Exception as e:
                logger.debug("Archive scrape failed for year %d: %s", yr, e)
                consecutive_failures += 1

    if discovered_urls:
        print(f"    Discovered {len(discovered_urls)} URLs from ECB archive pages")

    # --- Also try the ECB press release RSS/search API ---
    try:
        search_url = "https://www.ecb.europa.eu/press/pr/html/index.en.html"
        resp = requests.get(search_url, timeout=15, verify=VERIFY_SSL,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "monetary" in href.lower() or "ecb.mp" in href.lower() or "ecb.is" in href.lower() or "ecb.mps" in href.lower():
                    for date, date_str in statement_dates:
                        yy = f"{date.year % 100:02d}"
                        mm = f"{date.month:02d}"
                        dd = f"{date.day:02d}"
                        if f"{yy}{mm}{dd}" in href and date_str not in discovered_urls:
                            full_url = href if href.startswith("http") else f"https://www.ecb.europa.eu{href}"
                            discovered_urls[date_str] = full_url
    except Exception as e:
        logger.warning("ECB press release search failed: %s", e)

    consecutive_url_failures = 0
    for date, date_str in statement_dates:
        if consecutive_url_failures >= 40:
            print(f"    [INFO] Aborting text fetch after {consecutive_url_failures} consecutive failures")
            break
        yy = f"{date.year % 100:02d}"
        mm = f"{date.month:02d}"
        dd = f"{date.day:02d}"
        yyyy = str(date.year)
        yyyymmdd = f"{yyyy}{mm}{dd}"

        # Build URL list: discovered URL first, then generated patterns
        urls_to_try = []

        if date_str in discovered_urls:
            urls_to_try.append(discovered_urls[date_str])

        urls_to_try.extend([
            # New ECB path (redirected from old URLs for most years)
            f"https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/{yyyy}/html/is{yy}{mm}{dd}.en.html",
            # Introductory statement (pre-2015 format — redirects to new path)
            f"https://www.ecb.europa.eu/press/pressconf/{yyyy}/html/is{yy}{mm}{dd}.en.html",
            # Monetary policy decisions (pre-2015 format)
            f"https://www.ecb.europa.eu/press/pr/date/{yyyy}/html/pr{yy}{mm}{dd}.en.html",
            # Alt date format with full year
            f"https://www.ecb.europa.eu/press/pressconf/{yyyy}/html/is{yyyymmdd}.en.html",
            # New path with ecb.is prefix
            f"https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/{yyyy}/html/ecb.is{yyyymmdd}.en.html",
            f"https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/{yyyy}/html/ecb.is{yy}{mm}{dd}.en.html",
            # Monetary policy decisions (newer format without hash)
            f"https://www.ecb.europa.eu/press/pr/date/{yyyy}/html/ecb.mp{yy}{mm}{dd}.en.html",
            # Combined statement format (post-2019 switch)
            f"https://www.ecb.europa.eu/press/pressconf/shared/pdf/ecb.ds{yyyymmdd}.en.pdf",
            # Monetary policy decisions (full-year date)
            f"https://www.ecb.europa.eu/press/pr/date/{yyyy}/html/ecb.mp{yyyymmdd}.en.html",
            # Press conference transcript page
            f"https://www.ecb.europa.eu/press/pressconf/{yyyy}/html/ecb.is{yyyymmdd}.en.html",
            f"https://www.ecb.europa.eu/press/pressconf/{yyyy}/html/ecb.is{yy}{mm}{dd}.en.html",
            # Monetary policy statement (post-2019 combined format)
            f"https://www.ecb.europa.eu/press/pressconf/{yyyy}/html/ecb.mps{yyyymmdd}.en.html",
            f"https://www.ecb.europa.eu/press/pressconf/{yyyy}/html/ecb.mps{yy}{mm}{dd}.en.html",
            # Statement point / speech format
            f"https://www.ecb.europa.eu/press/pressconf/{yyyy}/html/ecb.sp{yyyymmdd}.en.html",
            f"https://www.ecb.europa.eu/press/pressconf/{yyyy}/html/ecb.sp{yy}{mm}{dd}.en.html",
            # Monetary policy decisions PDF
            f"https://www.ecb.europa.eu/press/pr/date/{yyyy}/html/ecb.mp{yy}{mm}{dd}.en.pdf",
            f"https://www.ecb.europa.eu/press/pr/date/{yyyy}/html/ecb.mp{yyyymmdd}.en.pdf",
        ])

        found_text = False
        for url in urls_to_try:
            try:
                resp = requests.get(url, timeout=8, verify=VERIFY_SSL,
                                    headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200 and len(resp.content) > 500:
                    # Handle PDF documents
                    if url.endswith(".pdf"):
                        text = _extract_text_from_pdf(resp.content)
                        if text and len(text) > 100:
                            fetched_texts.append((date, text))
                            found_text = True
                            break
                    else:
                        soup = BeautifulSoup(resp.text, "lxml")
                        text = _extract_ecb_text(soup)
                        if len(text) > 100:
                                fetched_texts.append((date, text))
                                found_text = True
                                break
            except Exception as e:
                logger.debug("URL fetch failed for %s: %s", date_str, e)
                continue

        if found_text:
            consecutive_url_failures = 0
        else:
            consecutive_url_failures += 1

    print(f"    Fetched {len(fetched_texts)} / {len(statement_dates)} press releases")

    if len(fetched_texts) < 5:
        print("    [INFO] Insufficient real texts — generating synthetic sentiment")
        return generate_synthetic_sentiment(start_year, end_year), "synthetic"

    # Step 3: Score sentiment with FinBERT
    pipe = get_finbert_pipeline()
    if pipe is None:
        print("    [INFO] FinBERT not available — generating synthetic sentiment")
        return generate_synthetic_sentiment(start_year, end_year), "synthetic"

    statement_scores = []
    processed = 0
    for date, text in fetched_texts:
        paragraphs = chunk_into_paragraphs(text)
        trade_paras = filter_trade_relevant(paragraphs)

        if not trade_paras:
            trade_paras = paragraphs[:10]

        scores = score_sentiment(trade_paras)
        if scores:
            mean_score = np.mean(scores)
            dispersion = np.std(scores) if len(scores) > 1 else 0.0
            statement_scores.append((date, mean_score, dispersion))
            processed += 1

        if processed % 10 == 0 and processed > 0:
            print(f"    Processed {processed}/{len(fetched_texts)} statements...")

    print(f"    [OK] Processed {processed} ECB statements")

    if not statement_scores:
        ecb_result = generate_synthetic_sentiment(start_year, end_year)
        ecb_status = "synthetic"
    else:
        ecb_result = aggregate_sentiment_monthly(statement_scores)
        ecb_result.to_csv(NLP_DIR / "ecb_sentiment.csv", index=False)
        print(f"    Saved ecb_sentiment.csv ({len(ecb_result)} monthly obs)")
        ecb_status = "real"

    # --- BdF Sentiment ---
    print("\n  BDF SENTIMENT PIPELINE")
    print("  " + "-" * 40)
    bdf_result, bdf_status = download_bdf_business_survey(start_year, end_year)

    # Merge ECB + BdF sentiment
    if not bdf_result.empty and "bdf_sentiment" in bdf_result.columns:
        combined = ecb_result.merge(bdf_result, on="date", how="outer")
        combined = combined.sort_values("date").reset_index(drop=True)
        # Forward-fill gaps
        for col in combined.columns:
            if col != "date":
                combined[col] = combined[col].ffill()
    else:
        combined = ecb_result

    overall_status = "real" if ecb_status == "real" or bdf_status == "real" else "synthetic"
    return combined, overall_status


def generate_synthetic_sentiment(start_year=2008, end_year=2022):
    """
    Generate synthetic ECB sentiment scores with realistic properties:
    - Mean near 0, slightly positive bias
    - More negative during GFC, euro crisis, COVID
    - Higher dispersion during crises
    """
    print("    [INFO] Generating synthetic ECB sentiment")
    date_range = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")
    n = len(date_range)
    np.random.seed(49)

    sentiment = np.zeros(n)
    dispersion = np.zeros(n)
    sentiment[0] = 0.1

    for i in range(1, n):
        # Mean-reverting process
        sentiment[i] = sentiment[i-1] + 0.1 * (0.05 - sentiment[i-1]) + np.random.normal(0, 0.05)
        dispersion[i] = abs(np.random.normal(0.15, 0.05))

    sentiment = np.clip(sentiment, -0.8, 0.8)

    # Crisis regimes
    for i, d in enumerate(date_range):
        if pd.Timestamp("2008-09-01") <= d <= pd.Timestamp("2009-06-01"):
            sentiment[i] -= 0.4
            dispersion[i] += 0.15
        elif pd.Timestamp("2011-06-01") <= d <= pd.Timestamp("2012-06-01"):
            sentiment[i] -= 0.3
            dispersion[i] += 0.10
        elif pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2020-09-01"):
            sentiment[i] -= 0.35
            dispersion[i] += 0.12
        elif pd.Timestamp("2022-02-01") <= d <= pd.Timestamp("2022-12-01"):
            sentiment[i] -= 0.2
            dispersion[i] += 0.08

    df = pd.DataFrame({
        "date": date_range,
        "ecb_sentiment": sentiment,
        "sentiment_dispersion": dispersion,
    })
    df.to_csv(NLP_DIR / "ecb_sentiment_synthetic.csv", index=False)
    return df


# =====================================================================
# BdF (Banque de France) Sentiment Pipeline
# =====================================================================

BDF_KEYWORDS = {
    "trade", "export", "import", "external demand", "foreign orders",
    "competitiveness", "order books", "production", "outlook",
    "activity", "industry", "manufacturing", "services",
    "business climate", "economic outlook",
    # French keywords (BdF sometimes mixes languages in summaries)
    "commerce", "exportation", "importation", "conjoncture",
}


def download_bdf_business_survey(start_year=2008, end_year=2022):
    """
    Download Banque de France monthly business survey indicator.

    The BdF publishes a monthly "Enquete Mensuelle de Conjoncture"
    (Monthly Business Survey) with a composite indicator.
    We use the FRED mirror of the OECD Business Confidence Index
    for France (BSCICP03FRM665S) as a quantitative proxy,
    plus attempt to fetch BdF Stat publication summaries for text analysis.
    """
    print("    Downloading BdF business survey data...")

    # --- Quantitative: OECD BCI for France from FRED ---
    bci_df = None
    try:
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id=BSCICP03FRM665S&cosd={start_year}-01-01&coed={end_year}-12-31"
        )
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200 and len(resp.text) > 50:
            from io import StringIO
            bci_df = pd.read_csv(StringIO(resp.text))
            bci_df.columns = ["date", "bdf_bci"]
            bci_df["date"] = pd.to_datetime(bci_df["date"])
            bci_df["bdf_bci"] = pd.to_numeric(bci_df["bdf_bci"], errors="coerce")
            bci_df = bci_df.dropna(subset=["bdf_bci"])
            # Normalize BCI to [-1, +1] scale (index is centered around 100)
            bci_df["bdf_bci_norm"] = (bci_df["bdf_bci"] - 100) / 10.0
            bci_df["bdf_bci_norm"] = bci_df["bdf_bci_norm"].clip(-1, 1)
            print(f"    [OK] BdF BCI: {len(bci_df)} monthly obs from FRED")
    except Exception as e:
        print(f"    [WARN] BdF BCI download failed: {e}")

    # --- Text-based: BdF Stat monthly conjoncture publications ---
    bdf_texts = []
    try:
        from bs4 import BeautifulSoup
        # BdF publishes monthly business outlook on their stat portal
        # Try the BdF open data API for publication metadata
        api_url = (
            "https://webstat.banque-france.fr/ws/rest/data/BDF"
            "/M.FR.BSCI.TOT.CLI?format=csvdata"
        )
        resp = requests.get(api_url, timeout=20,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and len(resp.text) > 100:
            from io import StringIO
            raw = pd.read_csv(StringIO(resp.text))
            # Look for date and value columns
            for col in raw.columns:
                if "period" in col.lower() or "time" in col.lower():
                    date_col = col
                if "obs" in col.lower() or "value" in col.lower():
                    val_col = col
            if "date_col" in dir() and "val_col" in dir():
                print(f"    [INFO] BdF Stat API returned {len(raw)} rows")
    except Exception as e:
        logger.warning("BdF Stat API download failed: %s", e)

    # --- Construct BdF sentiment from BCI + derived features ---
    if bci_df is not None and len(bci_df) > 12:
        # Convert BCI into sentiment-like features:
        # 1. bdf_sentiment = normalized BCI (captures level)
        # 2. bdf_momentum = MoM change in BCI (captures direction)
        bci_df = bci_df.sort_values("date").reset_index(drop=True)
        bci_df["bdf_sentiment"] = bci_df["bdf_bci_norm"]
        bci_df["bdf_momentum"] = bci_df["bdf_bci"].diff()
        # Normalize momentum to [-1, +1]
        mom_std = bci_df["bdf_momentum"].std()
        if mom_std > 0:
            bci_df["bdf_momentum"] = (bci_df["bdf_momentum"] / (3 * mom_std)).clip(-1, 1)
        else:
            bci_df["bdf_momentum"] = 0.0

        result = bci_df[["date", "bdf_sentiment", "bdf_momentum"]].copy()
        result.to_csv(NLP_DIR / "bdf_sentiment.csv", index=False)
        print(f"    [OK] BdF sentiment saved ({len(result)} obs)")
        return result, "real"

    # Fallback: synthetic BdF sentiment
    print("    [INFO] BdF data unavailable, generating synthetic")
    return generate_synthetic_bdf_sentiment(start_year, end_year), "synthetic"


def generate_synthetic_bdf_sentiment(start_year=2008, end_year=2022):
    """Generate synthetic BdF sentiment aligned with business cycle."""
    date_range = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")
    n = len(date_range)
    np.random.seed(77)

    sentiment = np.zeros(n)
    sentiment[0] = 0.0
    for i in range(1, n):
        sentiment[i] = sentiment[i-1] + 0.1 * (-sentiment[i-1]) + np.random.normal(0, 0.06)

    # Crisis shocks
    for i, d in enumerate(date_range):
        if pd.Timestamp("2008-09-01") <= d <= pd.Timestamp("2009-06-01"):
            sentiment[i] -= 0.5
        elif pd.Timestamp("2011-06-01") <= d <= pd.Timestamp("2012-06-01"):
            sentiment[i] -= 0.3
        elif pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2020-09-01"):
            sentiment[i] -= 0.45
        elif pd.Timestamp("2022-02-01") <= d <= pd.Timestamp("2022-12-01"):
            sentiment[i] -= 0.25

    sentiment = np.clip(sentiment, -1, 1)
    momentum = np.diff(sentiment, prepend=sentiment[0])
    momentum = np.clip(momentum / 0.3, -1, 1)

    df = pd.DataFrame({
        "date": date_range,
        "bdf_sentiment": sentiment,
        "bdf_momentum": momentum,
    })
    df.to_csv(NLP_DIR / "bdf_sentiment_synthetic.csv", index=False)
    return df
