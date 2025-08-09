from __future__ import annotations
from typing import Dict, List
import feedparser
import re
import requests

# Very light signal builder: Reddit RSS + simple news RSS + fuzzy match on names.
# The app caches the returned dict for 15 minutes; keep this lean.

SUBREDDIT_RSS = "https://www.reddit.com/r/{sub}/new/.rss"
NEWS_FEEDS = [
    "https://www.theguardian.com/football/rss",
    "https://www.skysports.com/rss/12040",
]

UA = {"User-Agent": "Mozilla/5.0"}

def _score_from_text(text: str) -> float:
    t = text.lower()
    score = 0.0
    if any(k in t for k in ["brace", "hat trick", "form", "man of the match", "motm"]):
        score += 0.7
    if any(k in t for k in ["injury", "doubt", "knock", "out for", "ruled out"]):
        score -= 0.8
    if any(k in t for k in ["transfer", "signs", "agrees", "loan"]):
        score += 0.3
    return score

def _fetch_rss(url: str) -> List[str]:
    try:
        f = feedparser.parse(url)
        out = []
        for e in f.entries[:50]:
            out.append(f"{e.get('title','')} {e.get('summary','')}")
        return out
    except Exception:
        return []

def build_buzz_map(secrets: dict, player_names: List[str], reddit_subs: List[str]) -> Dict[str, float]:
    texts: List[str] = []
    for sub in reddit_subs:
        texts += _fetch_rss(SUBREDDIT_RSS.format(sub=sub))
    for feed in NEWS_FEEDS:
        texts += _fetch_rss(feed)

    # Very cheap name match: whole word or capitalized fragments
    buzz = {name: 0.0 for name in player_names}
    for name in player_names:
        n = re.escape(name)
        pat = re.compile(rf"\b{n}\b", flags=re.I)
        for t in texts:
            if pat.search(t or ""):
                buzz[name] += _score_from_text(t)

    # Normalize to ~0..1
    if buzz:
        vals = list(buzz.values())
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        for k in list(buzz.keys()):
            buzz[k] = (buzz[k] - lo) / rng
    return buzz
