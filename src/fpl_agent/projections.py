from __future__ import annotations
import numpy as np
import pandas as pd

ROLE_W = {"GK": 0.65, "DEF": 0.80, "MID": 1.00, "FWD": 1.05}
TEAM_KEYS = [
    "strength_overall_home", "strength_overall_away",
    "strength_attack_home",  "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
]

def make_fixture_table(players: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if "event" in fixtures.columns and fixtures["event"].notna().any():
        next_event = int(fixtures["event"].dropna().min())
    else:
        next_event = 1
    gw_fx = fixtures[fixtures["event"] == next_event].copy()
    rows = []
    for _, m in gw_fx.iterrows():
        rows.append({"team_id": m["team_h"], "opp_id": m["team_a"], "was_home": 1})
        rows.append({"team_id": m["team_a"], "opp_id": m["team_h"], "was_home": 0})
    return pd.DataFrame(rows)

def normalize_team_strength(teams: pd.DataFrame) -> pd.DataFrame:
    st = teams[["team_id"] + TEAM_KEYS].copy()
    for k in TEAM_KEYS:
        mu, sd = st[k].mean(), st[k].std(ddof=0) or 1.0
        st[k] = (st[k] - mu) / sd
    return st

def expected_points_next_gw(players: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    team_strength = normalize_team_strength(teams)
    fx_tbl = make_fixture_table(players, teams, fixtures)

    df = players[["player_id","web_name","position","cost","team_id","ict_index","selected_by_percent","status","news","chance_of_playing_next_round"]].copy()
    df = df.merge(team_strength, on="team_id", how="left")
    df = df.merge(fx_tbl, on="team_id", how="left")

    opp_strength = team_strength.add_suffix("_opp").rename(columns={"team_id_opp": "opp_id"})
    df = df.merge(opp_strength, on="opp_id", how="left")

    cop = df["chance_of_playing_next_round"].fillna(100)
    minutes_factor = np.clip(cop / 100.0, 0.0, 1.0)

    role_w = df["position"].map(ROLE_W).fillna(0.9)

    ict = pd.to_numeric(df["ict_index"], errors="coerce").fillna(0.0)
    ict_z = (ict - ict.mean()) / (ict.std(ddof=0) or 1.0)

    home = df["was_home"].fillna(0)
    att = df[["strength_attack_home", "strength_attack_away"]].mean(axis=1)
    opp_def = df[["strength_defence_home_opp", "strength_defence_away_opp"]].mean(axis=1)
    fixture_adj = att - opp_def + 0.15 * home

    team_def = df[["strength_defence_home", "strength_defence_away"]].mean(axis=1)
    opp_att = df[["strength_attack_home_opp", "strength_attack_away_opp"]].mean(axis=1)
    cs_proxy = (team_def - opp_att).clip(-2, 2) * df["position"].isin(["GK","DEF"]).astype(float)

    sel = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0.0)
    sel_norm = (sel - sel.min()) / ((sel.max() - sel.min()) or 1.0)
    bonus_proxy = 0.1 * sel_norm

    xpts = minutes_factor * (role_w * (0.9 * ict_z) + 0.6 * fixture_adj + 0.4 * cs_proxy + bonus_proxy)
    xpts = 4.0 + 2.0 * xpts

    out = df[["player_id","web_name","position","team_id","cost","status"]].copy()
    out["xPts"] = xpts.clip(lower=0.0)
    out["price_momentum"] = sel_norm - 0.5
    return out
# projections.py placeholder
