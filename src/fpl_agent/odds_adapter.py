from __future__ import annotations
import os
from typing import Dict, List
import requests
import pandas as pd

# Map your FPL team names to bookmaker team names if needed
TEAM_NAME_ALIASES: Dict[str, str] = {
    # "Man City": "Manchester City",
    # "Nott'm Forest": "Nottingham Forest",
}

def _alias(name: str) -> str:
    return TEAM_NAME_ALIASES.get(name, name)

def fetch_match_odds(teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-team odds features: win_prob, draw_prob, lose_prob, over25_prob, cs_prob (if available).
    Requires ODDS_API_KEY in env or Streamlit secrets forwarded to os.environ.
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return pd.DataFrame()

    try:
        # Example for The Odds API (you can swap provider easily)
        # Docs: https://the-odds-api.com/
        url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
        params = {"apiKey": api_key, "regions": "uk,eu", "markets": "h2h,totals", "oddsFormat": "decimal"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    # Build a tidy odds table keyed by home/away team names
    rows: List[Dict] = []
    for match in data:
        try:
            home = match["home_team"]
            away = match["away_team"]
            # flatten first bookmaker only for simplicity
            books = match.get("bookmakers", [])
            if not books:
                continue
            mkts = {m["key"]: m for m in books[0].get("markets", [])}
            # h2h three-way probs (convert decimal odds to implied probs 1/odds and normalize)
            win_p = draw_p = lose_p = None
            if "h2h" in mkts:
                o = mkts["h2h"]["outcomes"]
                # outcomes: list of {name: team/draw, price: decimal}
                price = {x["name"]: float(x["price"]) for x in o if "price" in x and "name" in x}
                imp = {k: (1.0 / v) for k, v in price.items() if v > 0}
                total = sum(imp.values()) or 1.0
                imp = {k: v/total for k, v in imp.items()}
                win_p = imp.get(home)
                draw_p = imp.get("Draw")
                lose_p = imp.get(away)

            over25 = None
            if "totals" in mkts:
                # pick the line closest to 2.5 and take the Over prob
                lines = mkts["totals"]["outcomes"]
                # outcomes have "name": "Over 2.5", "price": ...
                best = None
                for x in lines:
                    name = str(x.get("name",""))
                    if "Over 2.5" in name:
                        best = x
                        break
                if best and best.get("price"):
                    over25 = 1.0 / float(best["price"])

            rows.append({"home_team": home, "away_team": away,
                         "home_win_prob": win_p, "draw_prob": draw_p, "away_win_prob": lose_p,
                         "over25_prob": over25})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    odds = pd.DataFrame(rows)

    # explode to per-team rows
    home = odds.rename(columns={
        "home_team":"team_name", "home_win_prob":"win_prob"
    })[["team_name","win_prob","over25_prob"]].copy()
    away = odds.rename(columns={
        "away_team":"team_name", "away_win_prob":"win_prob"
    })[["team_name","win_prob","over25_prob"]].copy()
    out = pd.concat([home, away], ignore_index=True)

    # aliasing to match FPL team names
    out["team_name"] = out["team_name"].map(_alias)
    # safe clip
    out["win_prob"] = out["win_prob"].fillna(0.0).clip(0,1)
    out["over25_prob"] = out["over25_prob"].fillna(0.0).clip(0,1)
    return out
