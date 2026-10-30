"""
Financial News Sentiment & Macro Event Intelligence Module
QuantEdge 1.0 - Serendib University Endowment Risk Engine (Phase 3)

Implements a transparent, deterministic rule-based financial sentiment analysis engine
inspired by the Loughran-McDonald institutional financial lexicon.
Identifies:
- Directional sentiment (Positive / Negative / Neutral)
- Related ETF assets & asset classes
- Critical macroeconomic events (FOMC, Inflation, Employment, Sovereign Debt)
- Potential market impact assessment (High / Medium / Low)
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

# Financial Sentiment Lexicon (Loughran-McDonald inspired financial dictionary)
FINANCIAL_POSITIVE_TERMS = {
    "gain", "gains", "gained", "gaining", "rally", "rallies", "rallied", "rallying",
    "surge", "surges", "surged", "surging", "profit", "profits", "profitable",
    "bullish", "growth", "grow", "growing", "beat", "beats", "beating", "outperform",
    "outperformed", "outperforming", "record", "high", "highs", "higher", "rebound",
    "rebounds", "rebounded", "upgrade", "upgraded", "upgrades", "dividend", "dividends",
    "expansion", "expanding", "jump", "jumps", "jumped", "soar", "soars", "soared",
    "boost", "boosts", "boosted", "strong", "stronger", "robust", "optimistic",
    "optimism", "recovery", "recover", "recovering", "advance", "advances", "advanced",
    "breakthrough", "stimulus", "tailwind", "tailwinds", "green", "climb", "climbs",
}

FINANCIAL_NEGATIVE_TERMS = {
    "loss", "losses", "lost", "losing", "drop", "drops", "dropped", "dropping",
    "fall", "falls", "fell", "falling", "slump", "slumps", "slumped", "slumping",
    "plunge", "plunges", "plunged", "plunging", "crash", "crashes", "crashed",
    "bearish", "decline", "declines", "declined", "declining", "miss", "misses",
    "missed", "downgrade", "downgrades", "downgraded", "risk", "risks", "crisis",
    "crises", "recession", "recessionary", "inflation", "inflationary", "debt",
    "deficit", "cut", "cuts", "cutting", "default", "defaults", "warning", "warnings",
    "down", "lower", "lows", "weak", "weaker", "weakness", "headwind", "headwinds",
    "sluggish", "negative", "bankrupt", "bankruptcy", "lawsuit", "sanction", "sanctions",
    "tightening", "hike", "hikes", "hikes", "selloff", "sell-off", "dump", "dumped",
}

NEGATION_TERMS = {"not", "no", "never", "without", "hardly", "barely", "failed", "fails", "failing"}
INTENSIFIERS = {"strongly", "significantly", "sharply", "dramatically", "substantially", "heavily", "massively"}

# Macroeconomic & Institutional Event Patterns
MACRO_EVENT_PATTERNS = {
    "Monetary Policy / FOMC": [r"\bfed\b", r"\bfomc\b", r"\bcentral bank\b", r"\brate cut\b", r"\brate hike\b", r"\bpowell\b"],
    "Inflation & Prices": [r"\bcpi\b", r"\bpce\b", r"\binflation\b", r"\bproducer price\b", r"\bstagflation\b"],
    "Labor & Employment": [r"\bunemployment\b", r"\bnonfarm\b", r"\bjobless\b", r"\blabor market\b", r"\bhiring\b"],
    "Sovereign Debt & Yields": [r"\btreasury\b", r"\byield curve\b", r"\bbond market\b", r"\b10-year\b", r"\bdebt ceiling\b"],
    "Geopolitics & Energy": [r"\bopec\b", r"\bcrude\b", r"\boil\b", r"\bwar\b", r"\bsanctions\b", r"\bmiddle east\b"],
    "Corporate Earnings": [r"\bearnings\b", r"\brevenue\b", r"\bguidance\b", r"\bquarterly results\b"],
}

# Mapping keywords to Endowment Asset Classes
ASSET_KEYWORD_MAPPING = {
    "SPY": ["s&p", "sp500", "spy", "equities", "stocks", "wall street", "us market"],
    "EFA": ["europe", "asia", "developed market", "japan", "uk", "eafe", "international stocks"],
    "EEM": ["emerging market", "china", "india", "brazil", "developing market", "taiwan"],
    "AGG": ["bonds", "fixed income", "treasuries", "treasury", "credit", "aggregate bond", "yield"],
    "GLD": ["gold", "precious metals", "bullion", "safe haven", "sovereign hedge"],
    "XLE": ["energy", "oil", "crude", "natural gas", "refinery", "exxon", "chevron", "petroleum"],
    "XLK": ["technology", "tech", "semiconductor", "chips", "apple", "microsoft", "nvidia", "software", "ai"],
}


@dataclass
class AnalyzedArticle:
    """Structured representation of an analyzed financial news article."""
    headline: str
    summary: str
    source: str
    published_time: str
    url: str
    sentiment_label: str  # "Positive", "Negative", "Neutral"
    sentiment_score: float  # -1.0 to +1.0
    related_etfs: List[str]
    macro_events: List[str]
    market_impact: str  # "High", "Medium", "Low"
    sentiment_drivers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Headline": self.headline,
            "Sentiment": self.sentiment_label,
            "Score": f"{self.sentiment_score:+.2f}",
            "Related ETFs": ", ".join(self.related_etfs) if self.related_etfs else "Broad Market",
            "Macro Events": ", ".join(self.macro_events) if self.macro_events else "General Corporate",
            "Impact": self.market_impact,
            "Source": self.source,
            "Published": self.published_time,
            "URL": self.url,
        }


def score_text_sentiment(text: str) -> Tuple[float, str, List[str]]:
    """
    Score sentiment of text using financial lexicon with negation and intensifier handling.
    
    Returns
    -------
    Tuple[float, str, List[str]]
        (compound_score, label, driver_words)
    """
    words = re.findall(r"\b[a-zA-Z\-]+\b", text.lower())
    if not words:
        return 0.0, "Neutral", []

    pos_score = 0.0
    neg_score = 0.0
    drivers: List[str] = []

    for i, word in enumerate(words):
        # Check window of 2 words prior for negation
        is_negated = False
        if i > 0 and words[i - 1] in NEGATION_TERMS:
            is_negated = True
        elif i > 1 and words[i - 2] in NEGATION_TERMS:
            is_negated = True

        # Check intensifiers
        multiplier = 1.0
        if i > 0 and words[i - 1] in INTENSIFIERS:
            multiplier = 1.5

        if word in FINANCIAL_POSITIVE_TERMS:
            if is_negated:
                neg_score += 1.0 * multiplier
                drivers.append(f"negated-{word}")
            else:
                pos_score += 1.0 * multiplier
                drivers.append(word)
        elif word in FINANCIAL_NEGATIVE_TERMS:
            if is_negated:
                pos_score += 1.0 * multiplier
                drivers.append(f"negated-{word}")
            else:
                neg_score += 1.0 * multiplier
                drivers.append(word)

    total_words = len(words)
    net = (pos_score - neg_score)
    # Normalized score between -1.0 and 1.0
    denom = (pos_score + neg_score + 1.0)
    compound = round(net / denom, 2)

    if compound >= 0.15:
        label = "Positive"
    elif compound <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    return compound, label, drivers[:5]


def detect_macro_events(text: str) -> List[str]:
    """Detect presence of critical economic / policy events in text."""
    text_lower = text.lower()
    events: List[str] = []
    for event_name, patterns in MACRO_EVENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                events.append(event_name)
                break
    return events


def detect_related_etfs(text: str, default_symbol: Optional[str] = None) -> List[str]:
    """Detect which endowment ETFs are relevant to the article."""
    text_lower = text.lower()
    related: List[str] = []

    if default_symbol and default_symbol.upper() in ASSET_KEYWORD_MAPPING:
        related.append(default_symbol.upper())

    for sym, keywords in ASSET_KEYWORD_MAPPING.items():
        if sym in related:
            continue
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                related.append(sym)
                break
    return related


def analyze_article(raw_article: Dict[str, Any], default_symbol: Optional[str] = None) -> AnalyzedArticle:
    """
    Perform complete sentiment, entity, and event analysis on a single news article.
    """
    headline = raw_article.get("headline", "No headline")
    summary = raw_article.get("summary", "")
    source = raw_article.get("source", "Finnhub")
    pub_time = raw_article.get("datetime", "N/A")
    url = raw_article.get("url", "#")

    combined_text = f"{headline} {summary}"

    score, sentiment_label, drivers = score_text_sentiment(combined_text)
    events = detect_macro_events(combined_text)
    related_etfs = detect_related_etfs(combined_text, default_symbol)

    # Determine impact severity: High if macro/rates/FOMC/inflation, Medium if sector, Low if company
    if any(e in events for e in ["Monetary Policy / FOMC", "Inflation & Prices", "Sovereign Debt & Yields"]):
        impact = "High"
    elif len(related_etfs) >= 2 or len(events) >= 1:
        impact = "Medium"
    else:
        impact = "Low"

    return AnalyzedArticle(
        headline=headline,
        summary=summary,
        source=source,
        published_time=pub_time,
        url=url,
        sentiment_label=sentiment_label,
        sentiment_score=score,
        related_etfs=related_etfs,
        macro_events=events,
        market_impact=impact,
        sentiment_drivers=drivers,
    )


def analyze_news_feed(articles: List[Dict[str, Any]], default_symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a collection of raw Finnhub news articles.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing aggregate sentiment metrics and analyzed articles list.
    """
    if not articles:
        return {
            "total_articles": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "average_sentiment": 0.0,
            "sentiment_stance": "Balanced / Neutral Sentiment",
            "articles": [],
            "high_impact_events": [],
        }

    analyzed: List[AnalyzedArticle] = [analyze_article(a, default_symbol) for a in articles]

    pos_c = sum(1 for a in analyzed if a.sentiment_label == "Positive")
    neg_c = sum(1 for a in analyzed if a.sentiment_label == "Negative")
    neu_c = sum(1 for a in analyzed if a.sentiment_label == "Neutral")
    tot = len(analyzed)

    avg_score = round(sum(a.sentiment_score for a in analyzed) / tot, 2) if tot > 0 else 0.0

    if avg_score >= 0.10:
        stance = "Bullish / Positive Sentiment"
    elif avg_score <= -0.10:
        stance = "Bearish / Negative Sentiment"
    else:
        stance = "Balanced / Neutral Sentiment"

    high_impact = [a for a in analyzed if a.market_impact == "High"]

    return {
        "total_articles": tot,
        "positive_count": pos_c,
        "negative_count": neg_c,
        "neutral_count": neu_c,
        "positive_ratio": round(pos_c / tot, 2) if tot > 0 else 0.0,
        "negative_ratio": round(neg_c / tot, 2) if tot > 0 else 0.0,
        "average_sentiment": avg_score,
        "sentiment_stance": stance,
        "articles": analyzed,
        "high_impact_events": high_impact,
    }
