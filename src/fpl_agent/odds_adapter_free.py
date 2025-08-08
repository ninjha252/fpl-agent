from __future__ import annotations
import re
import time
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

TEAM_NAME_ALIASES: Dict[str, str] = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham",
    "Newcastle Utd": "Newcastle",
    "Wolves": "Wolverhampton",
    "West Ham": "West Ham",
    "Brighton": "Brighton",
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

def _search_ddg(q: str) -> Optional[str]:
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

def _scrape_oddsportal_match(url: str) -> Optional[Tuple[str, str, float, float, float]]:
    try:
        r = requests.get(url, headers=UA, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        title = soup.find("title").get_text(" ", strip=True) if soup.find("title") else ""
        tm = re.search(r"([A-Za-z' .-]+)\s*-\s*([A-Za-z' .-]+)", title)
        if not tm:
            h = soup.select_one("h1")
            tm = re.search(r"([A-Za-z' .-]+)\s*-\s*([A-Za-z' .-]+)", h.get_text(" ", strip=True) if h else "")
        if not tm:
            return None
        home, away = _norm_space(tm.group(1)), _norm_space(tm.group(2))
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

def _scrape_bettingodds_match(url: str) -> Optional[Tuple[str, str, float, float, float]]:
    try:
        r = requests.get(url, headers=UA, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        h = soup.find(["h1", "h2"])
        if not h:
            return None
        m = re.search(r"([A-Za-z' .-]+)\s+v\s+([A-Za-z' .-]+)", h.get_text(" ", strip=True))
        if not m:
            return None
        home, away = _norm_space(m.group(1)), _norm_space(m.group(2))
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

def fetch_match_odds(teams_df: pd.DataFrame, sleep_between: float = 0.2) -> pd.DataFrame:
    """
    Free odds fetcher (best-effort). Returns columns: team_name, win_prob, over25_prob (over25 may be 0).
    """
    team_names = [str(x) for x in teams_df.get("team_name", []) if pd.notna(x)]
    out_rows: List[Dict] = []

    for team in team_names:
        alias = _alias(team)
        row = None

        # OddsPortal search & scrape
        url = _search_ddg(f"{alias} odds site:oddsportal.com")
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

        # BettingOdds.com as fallback
        if row is None:
            link = _search_ddg(f"{alias} odds site:bettingodds.com")
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
