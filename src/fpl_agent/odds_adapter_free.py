from __future__ import annotations
import re
import time
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ---------- helpers ----------
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

TEAM_NAME_ALIASES: Dict[str, str] = {
    # keep this small; add mappings only if you see mismatches
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham",
    "Newcastle Utd": "Newcastle",
    "Wolves": "Wolverhampton",
    "West Ham": "West Ham",
    "Brighton": "Brighton",
    "Leeds": "Leeds",
}

def _alias(name: str) -> str:
    return TEAM_NAME_ALIASES.get(name, name)

def _decimal_to_prob(odds: float) -> float:
    try:
        if odds and odds > 1e-9:
            return 1.0 / float(odds)
    except Exception:
        pass
    return 0.0

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _extract_first_float(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)", text or "")
    return float(m.group(1)) if m else None

# ---------- site scrapers ----------
def _scrape_oddsportal_match(url: str) -> Optional[Tuple[str, str, float, float, float]]:
    """
    Return: (home, away, home_dec, draw_dec, away_dec)
    """
    try:
        r = requests.get(url, headers=UA, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")

        # Title often: "Arsenal - Liverpool Odds | ..."
        title = soup.find("title").get_text(" ", strip=True) if soup.find("title") else ""
        tm = re.search(r"([A-Za-z' .-]+)\s*-\s*([A-Za-z' .-]+)", title)
        if not tm:
            # fallback from header
            h = soup.select_one("h1")
            tm = re.search(r"([A-Za-z' .-]+)\s*-\s*([A-Za-z' .-]+)", h.get_text(" ", strip=True) if h else "")
        if not tm:
            return None
        home, away = _norm_space(tm.group(1)), _norm_space(tm.group(2))

        # Get three-way odds – oddsportal renders many tables; pick first 3 price cells we can parse
        prices = []
        for cell in soup.find_all(text=re.compile(r"^\d+(\.\d+)?$")):
            try:
                val = float(cell)
                if 1.2 <= val <= 100.0:
                    prices.append(val)
            except Exception:
                continue
            if len(prices) >= 3:
                break
        if len(prices) < 3:
            return None
        return home, away, float(prices[0]), float(prices[1]), float(prices[2])
    except Exception:
        return None

def _scrape_bettingodds_search(q: str) -> Optional[str]:
    """
    Finds a match page on bettingodds.com via site search.
    Returns the first link if any.
    """
    try:
        url = f"https://duckduckgo.com/html/?q={requests.utils.quote(q)}"
        r = requests.get(url, headers=UA, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        a = soup.select_one("a.result__a")
        if a and a.get("href"):
            return a["href"]
    except Exception:
        return None
    return None

def _scrape_bettingodds_match(url: str) -> Optional[Tuple[str, str, float, float, float]]:
    """
    Attempt to parse a BettingOdds.com match page for 1x2 prices.
    """
    try:
        r = requests.get(url, headers=UA, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        # Team names:
        h = soup.find(["h1", "h2"])
        if not h:
            return None
        m = re.search(r"([A-Za-z' .-]+)\s+v\s+([A-Za-z' .-]+)", h.get_text(" ", strip=True))
        if not m:
            return None
        home, away = _norm_space(m.group(1)), _norm_space(m.group(2))

        # Find “Match Result” section — grab best odds (decimal) for Home/Draw/Away
        # Many pages show odds in buttons/spans; scrape the first three that look like numbers.
        prices: List[float] = []
        for elm in soup.find_all(text=re.compile(r"\d+(?:\.\d+)?")):
            num = _extract_first_float(elm)
            if num and 1.1 <= num <= 100:
                prices.append(num)
            if len(prices) >= 3:
                break
        if len(prices) < 3:
            return None
        return home, away, prices[0], prices[1], prices[2]
    except Exception:
        return None

def _search_oddsportal(team_name: str) -> Optional[str]:
    q = f"{team_name} odds site:oddsportal.com"
    try:
        url = f"https://duckduckgo.com/html/?q={requests.utils.quote(q)}"
        r = requests.get(url, headers=UA, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        link = soup.select_one("a.result__a")
        if link and link.get("href"):
            return link["href"]
    except Exception:
        return None
    return None

# ---------- public entry point ----------
def fetch_match_odds(teams_df: pd.DataFrame, sleep_between: float = 0.2) -> pd.DataFrame:
    """
    Free odds fetcher (no API keys). For each PL team, we try to find the
    *next* match odds on public sites and compute implied probabilities.

    Returns columns: team_name, win_prob, over25_prob (best-effort; over2.5 may be 0 if not found)
    """
    team_names = [str(x) for x in teams_df.get("team_name", []) if pd.notna(x)]
    out_rows: List[Dict] = []

    for team in team_names:
        alias = _alias(team)

        # 1) Try OddsPortal via search
        url = _search_oddsportal(alias)
        row = None
        if url:
            parsed = _scrape_oddsportal_match(url)
            if parsed:
                home, away, h, d, a = parsed
                total = sum([_decimal_to_prob(x) for x in (h, d, a)]) or 1.0
                probs = {
                    home: (_decimal_to_prob(h) / total),
                    "Draw": (_decimal_to_prob(d) / total),
                    away: (_decimal_to_prob(a) / total)
                }
                win_prob = probs.get(alias, probs.get(team, 0.0))
                row = {"team_name": team, "win_prob": float(win_prob), "over25_prob": 0.0}

        # 2) Fallback: BettingOdds.com via search
        if row is None:
            link = _scrape_bettingodds_search(f"{alias} odds site:bettingodds.com")
            if link:
                parsed = _scrape_bettingodds_match(link)
                if parsed:
                    home, away, h, d, a = parsed
                    total = sum([_decimal_to_prob(x) for x in (h, d, a)]) or 1.0
                    probs = {
                        home: (_decimal_to_prob(h) / total),
                        "Draw": (_decimal_to_prob(d) / total),
                        away: (_decimal_to_prob(a) / total)
                    }
                    win_prob = probs.get(alias, probs.get(team, 0.0))
                    row = {"team_name": team, "win_prob": float(win_prob), "over25_prob": 0.0}

        if row is not None:
            out_rows.append(row)
        time.sleep(sleep_between)

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows)
    out["win_prob"] = out["win_prob"].fillna(0.0).clip(0, 1)
    out["over25_prob"] = out["over25_prob"].fillna(0.0).clip(0, 1)
    return out
