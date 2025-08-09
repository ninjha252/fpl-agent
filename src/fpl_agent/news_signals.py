from __future__ import annotations

import re
import time
from typing import Dict, List, Tuple, Optional

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import process, fuzz

# Reddit is optional; the app still works if PRAW isn't installed/usable
try:
    import praw  # type: ignore
except Exception:
    praw = None

# -----------------------------
# Defaults (overridable from app)
# -----------------------------
DEFAULT_REDDIT_SUBS: List[str] = [
    "FantasyPL",
    "FantasyPremierLeague",
    "PremierLeague",
    "soccer",
]

DEFAULT_RSS_FEEDS: List[str] = [
    "https://www.premierleague.com/news/rss",
    "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "https://www.skysports.com/rss/12040",  # football
]

# Keyword hints to push sentiment up/down beyond VADER
INJURY_WORDS = r"(injured|knock|doubt|hamstring|groin|calf|ankle|ill|sick|out|ruled out|scan|setback|return date|injury)"
NAILED_WORDS = r"(first[- ]?choice|nailed|starter|90 mins|on pens|penalties|free[- ]?kicks|corners|set[- ]?pieces)"
ROTATION_WORDS = r"(rotation|bench(ed)?|rest|minutes managed|cameo|dropped|not starting)"

ANALYZER = SentimentIntensityAnalyzer()


# -----------------------------
# Helpers
# -----------------------------
def _make_name_index(player_names: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return a list of unique canonical names and alias map (lower -> canonical)."""
    canon: Dict[str, str] = {}
    uniq: List[str] = []
    for n in player_names:
        ln = str(n).strip()
        if not ln:
            continue
        key = ln.lower()
        if key not in canon:
            canon[key] = ln
            uniq.append(ln)
    return uniq, canon


def _best_name_matches(text: str, name_list: List[str], limit: int = 5, cutoff: int = 85) -> List[str]:
    """Return canonical names that closely match the text (fuzzy)."""
    matches = process.extract(text, name_list, scorer=fuzz.WRatio, limit=limit)
    return [m[0] for m in matches if int(m[1]) >= cutoff]


def _sentiment_score(s: str) -> float:
    """VADER compound score in [-1, 1]."""
    return ANALYZER.polarity_scores(s)["compound"]


def _keyword_boost(s: str) -> float:
    """Hand-tuned keyword nudges."""
    s_low = s.lower()
    score = 0.0
    if re.search(INJURY_WORDS, s_low):
        score -= 0.8
    if re.search(ROTATION_WORDS, s_low):
        score -= 0.4
    if re.search(NAILED_WORDS, s_low):
        score += 0.4
    return score


# -----------------------------
# Reddit
# -----------------------------
def _build_reddit_client(st_secrets) -> Optional["praw.Reddit"]:
    if praw is None:
        return None
    if not isinstance(st_secrets, dict) or "reddit" not in st_secrets:
        return None
    cfg = st_secrets.get("reddit", {})
    cid = cfg.get("client_id")
    csec = cfg.get("client_secret")
    ua = cfg.get("user_agent", "fpl-agent/0.1")
    if not cid or not csec:
        return None
    try:
        return praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)
    except Exception:
        return None


def fetch_reddit_buzz(
    st_secrets,
    player_names: List[str],
    limit_per_sub: int = 40,
    subs: Optional[List[str]] = None,
    sleep_per_sub: float = 0.4,
) -> Dict[str, float]:
    """
    Aggregate name mentions + sentiment from Reddit subs.
    Returns: dict {canonical_player_name: score}
    """
    reddit = _build_reddit_client(st_secrets)
    if not reddit:
        return {}

    use_subs = subs if subs else (
        st_secrets.get("reddit", {}).get("subs", DEFAULT_REDDIT_SUBS)  # allow list in secrets
        if isinstance(st_secrets, dict) else DEFAULT_REDDIT_SUBS
    )

    uniq, _ = _make_name_index(player_names)
    scores: Dict[str, float] = {}

    for sub in use_subs:
        try:
            sr = reddit.subreddit(sub)
            for post in sr.hot(limit=limit_per_sub):
                title = getattr(post, "title", "") or ""
                body = getattr(post, "selftext", "") or ""
                text = f"{title} {body}"
                sent = _sentiment_score(text) + _keyword_boost(text)  # -? .. +?
                # Each mention contributes a base + sentiment
                for name in _best_name_matches(text, uniq):
                    scores[name] = scores.get(name, 0.0) + (0.5 * sent + 0.5)
            time.sleep(sleep_per_sub)
        except Exception:
            # ignore sub failures so one bad sub doesn't kill the whole pass
            continue

    return scores


# -----------------------------
# RSS news
# -----------------------------
def fetch_rss_buzz(
    player_names: List[str],
    limit_total: int = 150,
    feeds: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Aggregate name mentions from RSS feeds.
    Returns: dict {canonical_player_name: score}
    """
    use_feeds = feeds or DEFAULT_RSS_FEEDS
    uniq, _ = _make_name_index(player_names)
    scores: Dict[str, float] = {}
    seen = 0

    for url in use_feeds:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:100]:
                title = str(e.get("title", "") or "")
                summary = str(e.get("summary", "") or "")
                text = f"{title} {summary}"
                sent = _sentiment_score(text) + _keyword_boost(text)
                for name in _best_name_matches(text, uniq):
                    # RSS gets a slightly stronger base because items are curated
                    scores[name] = scores.get(name, 0.0) + (0.6 * sent + 0.6)
                seen += 1
                if seen >= limit_total:
                    return scores
        except Exception:
            continue

    return scores


# -----------------------------
# Combine & normalize
# -----------------------------
def combine_buzz(
    reddit_scores: Dict[str, float],
    rss_scores: Dict[str, float],
    w_reddit: float = 0.6,
    w_rss: float = 0.4,
) -> Dict[str, float]:
    """
    Merge two score dicts and normalize to roughly [-1, +1].
    """
    keys = set(reddit_scores) | set(rss_scores)
    merged: Dict[str, float] = {}
    for k in keys:
        merged[k] = w_reddit * reddit_scores.get(k, 0.0) + w_rss * rss_scores.get(k, 0.0)

    if not merged:
        return {}

    vals = list(merged.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    # scale to [-1, +1]
    for k in list(merged.keys()):
        merged[k] = -1.0 + 2.0 * ((merged[k] - lo) / span)

    return merged


# -----------------------------
# Convenience: one-call builder
# -----------------------------
def build_buzz_map(
    st_secrets,
    player_names: List[str],
    reddit_limit_per_sub: int = 40,
    reddit_subs: Optional[List[str]] = None,
    rss_limit_total: int = 150,
    rss_feeds: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    One convenience call used by the app:
      -> fetch RSS
      -> fetch Reddit (if secrets configured)
      -> combine and return buzz map
    """
    rss = fetch_rss_buzz(player_names, limit_total=rss_limit_total, feeds=rss_feeds)
    reddit = fetch_reddit_buzz(
        st_secrets,
        player_names,
        limit_per_sub=reddit_limit_per_sub,
        subs=reddit_subs,
    )
    return combine_buzz(reddit, rss)
