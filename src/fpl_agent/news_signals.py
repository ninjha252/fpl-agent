from __future__ import annotations
import re, time
from typing import Dict, List, Tuple
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import process, fuzz

try:
    import praw  # optional
except Exception:
    praw = None

ANALYZER = SentimentIntensityAnalyzer()

REDDIT_SUBS = ["FantasyPL", "FantasyPremierLeague", "soccer"]
RSS_FEEDS = [
    "https://www.premierleague.com/news/rss",
    "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "https://www.skysports.com/rss/12040",
]

INJURY_WORDS = r"(injured|knock|doubt|hamstring|groin|calf|ankle|ill|sick|out|ruled out|scan|setback|return date)"
NAILED_WORDS = r"(first[- ]?choice|nailed|90 mins|starter|on pens|penalties|free[- ]kicks|corners|set[- ]pieces)"
ROTATION_WORDS = r"(rotation|bench(ed)?|rest|minutes managed|cameo|dropped)"

def _make_name_index(player_names: List[str]) -> Tuple[List[str], Dict[str, str]]:
    canon = {}
    uniq = []
    for n in player_names:
        ln = n.lower()
        if ln not in canon:
            canon[ln] = n
            uniq.append(n)
    return uniq, canon

def _best_name_match(text: str, name_list: List[str]) -> List[Tuple[str, int]]:
    matches = process.extract(text, name_list, scorer=fuzz.WRatio, limit=5)
    return [(m[0], int(m[1])) for m in matches if m[1] >= 85]

def _sentiment(s: str) -> float:
    vs = ANALYZER.polarity_scores(s)
    return vs["compound"]

def _keyword_boost(s: str) -> float:
    score = 0.0
    s_low = s.lower()
    if re.search(INJURY_WORDS, s_low):
        score -= 0.8
    if re.search(ROTATION_WORDS, s_low):
        score -= 0.4
    if re.search(NAILED_WORDS, s_low):
        score += 0.4
    return score

def fetch_reddit_buzz(st_secrets, player_names: List[str], limit_per_sub: int = 40) -> Dict[str, float]:
    if praw is None or "reddit" not in st_secrets:
        return {}
    cfg = st_secrets["reddit"]
    try:
        reddit = praw.Reddit(
            client_id=cfg.get("client_id"),
            client_secret=cfg.get("client_secret"),
            user_agent=cfg.get("user_agent", "fpl-agent"),
        )
    except Exception:
        return {}

    uniq, _ = _make_name_index(player_names)
    scores: Dict[str, float] = {}
    for sub in REDDIT_SUBS:
        try:
            for post in reddit.subreddit(sub).hot(limit=limit_per_sub):
                text = f"{post.title} {getattr(post, 'selftext', '') or ''}"
                sent = _sentiment(text) + _keyword_boost(text)
                for name, _sim in _best_name_match(text, uniq):
                    scores[name] = scores.get(name, 0.0) + (0.5 * sent + 0.5)
            time.sleep(0.4)
        except Exception:
            continue
    return scores

def fetch_rss_buzz(player_names: List[str], limit_total: int = 120) -> Dict[str, float]:
    uniq, _ = _make_name_index(player_names)
    scores: Dict[str, float] = {}
    seen = 0
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:60]:
                text = f"{e.get('title','')} {e.get('summary','')}"
                sent = _sentiment(text) + _keyword_boost(text)
                for name, _sim in _best_name_match(text, uniq):
                    scores[name] = scores.get(name, 0.0) + (0.6 * sent + 0.4)
                seen += 1
                if seen >= limit_total:
                    return scores
        except Exception:
            continue
    return scores

def combine_buzz(reddit_scores: Dict[str, float], rss_scores: Dict[str, float]) -> Dict[str, float]:
    keys = set(reddit_scores) | set(rss_scores)
    out = {}
    for k in keys:
        out[k] = 0.6 * reddit_scores.get(k, 0.0) + 0.4 * rss_scores.get(k, 0.0)
    if not out:
        return out
    vals = list(out.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    for k in out:
        out[k] = -1.0 + 2.0 * ((out[k] - lo) / span)
    return out
