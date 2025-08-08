from __future__ import annotations
import pandas as pd
from typing import Tuple
from .fpl_api import FPL
from .utils import POSITION_MAP, TEAM_ID_TO_NAME

def load_bootstrap() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    js = FPL.bootstrap()
    players = pd.DataFrame(js["elements"])
    teams   = pd.DataFrame(js["teams"])
    events  = pd.DataFrame(js.get("events", []))

    # --- Teams: always provide team_id and team_name for display ---
    teams = teams.rename(columns={"id": "team_id"})  # keep original 'name' too
    if "team_name" not in teams.columns:
        teams["team_name"] = teams.get("name", "")

    # --- Players: normalize ---
    players = players.rename(columns={
        "id": "player_id",
        "element_type": "pos_id",
        "now_cost": "cost_tenths",
        "team": "team_id"
    })
    players["position"] = players["pos_id"].map(POSITION_MAP)
    players["cost"] = players["cost_tenths"] / 10.0

    # --- team id -> name map (optional for other modules) ---
    TEAM_ID_TO_NAME.clear()
    for _, r in teams.iterrows():
        TEAM_ID_TO_NAME[int(r.team_id)] = r.team_name

    return players, teams, events

def load_fixtures() -> pd.DataFrame:
    fx = FPL.fixtures()
    df = pd.DataFrame(fx)
    # Keep only scheduled/unplayed
    if "finished" in df.columns:
        df = df[df["finished"] == False]
    return df
