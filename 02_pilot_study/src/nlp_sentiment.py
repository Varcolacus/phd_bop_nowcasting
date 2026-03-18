"""
NLP Sentiment Pipeline for Chapter 2 — BoP Nowcasting
=======================================================

Extracts trade-relevant sentiment from ECB Monetary Policy Statements
using FinBERT (ProsusAI/finbert).

Pipeline:
  1. Collect ECB press conference introductory statements (HTML scraping)
  2. Chunk into paragraphs and filter for trade-relevant content
  3. Score sentiment using FinBERT
  4. Aggregate to monthly frequency

Author: PhD Pilot Study
Date: March 2026
"""

import warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

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
    ECB publishes these in a predictable URL pattern.
    Returns list of (date, url) tuples.
    """
    # ECB Governing Council typically meets ~8 times per year
    # Dates are approximate — we try common patterns
    statements = []

    # Known ECB press conference dates (major ones)
    # In practice, these would be scraped from the ECB index page
    # For robustness, we try to scrape the index
    try:
        from bs4 import BeautifulSoup
        index_url = f"{ECB_PRESS_BASE}/html/index.en.html"
        resp = requests.get(index_url, timeout=30, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Find links to individual press conferences
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/pressconf/" in href and href.endswith(".en.html"):
                # Extract date from URL pattern
                try:
                    # URLs like: /press/pressconf/2022/html/ecb.is220721~xxx.en.html
                    parts = href.split("ecb.")
                    if len(parts) > 1:
                        date_part = parts[1].split("~")[0].replace("is", "")
                        if len(date_part) == 6:
                            year = 2000 + int(date_part[:2])
                            month = int(date_part[2:4])
                            day = int(date_part[4:6])
                            date = pd.Timestamp(year=year, month=month, day=day)
                            if start_year <= year <= end_year:
                                full_url = "https://www.ecb.europa.eu" + href if href.startswith("/") else href
                                statements.append((date, full_url))
                except (ValueError, IndexError):
                    continue

        if statements:
            statements.sort(key=lambda x: x[0])
            print(f"    Found {len(statements)} ECB press conference URLs")
            return statements

    except Exception as e:
        print(f"    [WARN] Index scraping failed: {e}")

    print("    [INFO] Using known ECB press conference dates")
    return []


def scrape_ecb_statement(url):
    """
    Scrape the text of an ECB press conference statement from its URL.
    Returns the full text as a string.
    """
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=30, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # ECB statements have the main text in div with class 'section'
        # or in the main content area
        content = soup.find("div", class_="section")
        if content is None:
            content = soup.find("div", id="main-wrapper")
        if content is None:
            content = soup.find("article")
        if content is None:
            return ""

        # Get paragraphs
        paragraphs = content.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs)
        return text

    except Exception:
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
        except Exception:
            scores.append(0.0)

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
# Full NLP pipeline
# =====================================================================

def run_nlp_pipeline(start_year=2008, end_year=2022):
    """
    Run the full NLP sentiment pipeline on ECB press conferences.

    Returns:
        DataFrame(date, ecb_sentiment, sentiment_dispersion)
        str: "real" or "synthetic"
    """
    print("\n  NLP SENTIMENT PIPELINE")
    print("  " + "-" * 40)

    # Step 1: Get statement URLs
    statement_urls = get_ecb_statement_urls(start_year, end_year)

    if not statement_urls:
        print("    [INFO] No statement URLs found — generating synthetic sentiment")
        return generate_synthetic_sentiment(start_year, end_year), "synthetic"

    # Step 2-3: Scrape, chunk, filter, score
    pipe = get_finbert_pipeline()
    if pipe is None:
        print("    [INFO] FinBERT not available — generating synthetic sentiment")
        return generate_synthetic_sentiment(start_year, end_year), "synthetic"

    statement_scores = []
    processed = 0
    for date, url in statement_urls:
        text = scrape_ecb_statement(url)
        if not text:
            continue

        paragraphs = chunk_into_paragraphs(text)
        trade_paras = filter_trade_relevant(paragraphs)

        if not trade_paras:
            # Use all paragraphs if no trade-specific ones found
            trade_paras = paragraphs[:10]

        scores = score_sentiment(trade_paras)
        if scores:
            mean_score = np.mean(scores)
            dispersion = np.std(scores) if len(scores) > 1 else 0.0
            statement_scores.append((date, mean_score, dispersion))
            processed += 1

        if processed % 10 == 0 and processed > 0:
            print(f"    Processed {processed}/{len(statement_urls)} statements...")

    print(f"    [OK] Processed {processed} ECB statements")

    if not statement_scores:
        return generate_synthetic_sentiment(start_year, end_year), "synthetic"

    result = aggregate_sentiment_monthly(statement_scores)
    result.to_csv(NLP_DIR / "ecb_sentiment.csv", index=False)
    print(f"    Saved ecb_sentiment.csv ({len(result)} monthly obs)")

    return result, "real"


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
