from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

# ==== knobs (tweak from UI via setters) ====
NAILEDNESS_SCALE: float = 1.0
STARTER_HIDE_THRESHOLD: float = 0.35
GK_BACKUP_MULT: float = 0.10
BACKUP_MULT: float = 0.25
NEWS_BUZZ_WEIGHT: float = 0.5

# Manual overrides
MANUAL_ROLE_OVERRIDES: Dict[str, str] = {
    "Arrizabalaga": "backup",
    "Kepa": "backup",
}

ROLE_W = {"GK": 0.65, "DEF": 0.80, "MID": 1.00, "FWD": 1.05}
TEAM_KEYS = [
    "strength_overall_home", "strength_overall_away",
    "strength_attack_home",  "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
]

def set_nailedness_scale(v: float):
    global NAILEDNESS_SCALE
    NAILEDNESS_SCALE = float(v)

def set_starter_hide_threshold(v: float):
    global STARTER_HIDE_THRESHOLD
    STARTER_HIDE_THRESHOLD = float(v)

def set_manual_role_override(name: str, role: str):
    key = str(name).strip()
    if not key:
        return
    MANUAL_ROLE_OVERRIDES[key] = role.lower().strip()

# ---------- helpers ----------
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

# ---------- starter logic ----------
def _first_choice_gk_flags(players: pd.DataFrame) -> pd.Series:
    cols = ["player_id", "team_id", "cost", "minutes", "selected_by_percent"]
    gk = players[players["position"] == "GK"][cols].copy()
    gk["minutes"] = pd.to_numeric(gk.get("minutes", 0), errors="coerce").fillna(0.0)
    gk["selected_by_percent"] = pd.to_numeric(gk.get("selected_by_percent", 0), errors="coerce").fillna(0.0)
    gk = gk.sort_values(["team_id", "minutes", "selected_by_percent", "cost"],
                        ascending=[True, False, False, False]).copy()
    gk["rank"] = gk.groupby("team_id").cumcount()
    first_flags = (gk["rank"] == 0).astype(float)
    return first_flags.reindex(players["player_id"]).fillna(0.0)

def _starter_probability(players: pd.DataFrame) -> pd.Series:
    df = players.copy()
    base = pd.to_numeric(df.get("chance_of_playing_next_round", 100), errors="coerce").fillna(100.0) / 100.0

    sel = pd.to_numeric(df.get("selected_by_percent", 0.0), errors="coerce").fillna(0.0)
    sel_norm = (sel - sel.min()) / ((sel.max() - sel.min()) or 1.0)

    mins = pd.to_numeric(df.get("minutes", 0.0), errors="coerce").fillna(0.0)
    z = df[["team_id", "position"]].copy()
    z["minutes"] = mins
    z["min_rank"] = z.groupby(["team_id", "position"])["minutes"] \
                     .transform(lambda s: (s - s.min()) / ((s.max() - s.min()) or 1.0))
    nailed = (0.6 * z["min_rank"] + 0.4 * sel_norm)
    nailed = (NAILEDNESS_SCALE * nailed).clip(0.0, 1.5)

    first_gk = _first_choice_gk_flags(df)
    gk_adj = pd.Series(0.0, index=df.index)
    is_gk = (df["position"] == "GK").astype(float)
    gk_adj += -0.60 * is_gk * (1 - first_gk.reindex(df["player_id"]).fillna(0.0))
    gk_adj += +0.10 * is_gk * first_gk.reindex(df["player_id"]).fillna(0.0)

    names = df["web_name"].astype(str)
    manual = names.map(lambda n: MANUAL_ROLE_OVERRIDES.get(n, MANUAL_ROLE_OVERRIDES.get(n.strip(), None)))
    is_manual_backup = (manual == "backup").astype(float)
    is_manual_starter = (manual == "starter").astype(float)

    status = df.get("status", "a").astype(str)
    bad_status = status.isin(["i", "o", "n", "s"]).astype(float)
    clamp = 1.0 - 0.85 * bad_status

    prob = (0.55 * base + 0.35 * nailed + gk_adj).clip(0.0, 1.0)
    prob = np.where(is_manual_backup > 0, np.minimum(prob, 0.10), prob)
    prob = np.where(is_manual_starter > 0, np.maximum(prob, 0.80), prob)
    prob = (prob * clamp).clip(0.0, 1.0)

    return pd.Series(prob, index=df.index)

# ---------- main projections ----------
def expected_points_next_gw(players: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    team_strength = normalize_team_strength(teams)
    fx_tbl = make_fixture_table(players, teams, fixtures)

    df = players[[
        "player_id","web_name","position","cost","team_id","minutes",
        "ict_index","selected_by_percent","status","news","chance_of_playing_next_round"
    ]].copy()
    df = df.merge(team_strength, on="team_id", how="left")
    df = df.merge(fx_tbl, on="team_id", how="left")

    opp_strength = team_strength.add_suffix("_opp").rename(columns={"team_id_opp": "opp_id"})
    df = df.merge(opp_strength, on="opp_id", how="left")

    # starter prob (as minutes factor)
    starter_prob = _starter_probability(df)
    minutes_factor = starter_prob

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
    out["starter_prob"] = starter_prob.clip(0.0, 1.0)

    # demote backups
    is_gk = out["position"].eq("GK")
    gk_backup_mask = is_gk & (out["starter_prob"] < 0.5)
    of_backup_mask = ~is_gk & (out["starter_prob"] < 0.35)
    xpts = np.where(gk_backup_mask, xpts * GK_BACKUP_MULT, xpts)
    xpts = np.where(of_backup_mask, xpts * BACKUP_MULT, xpts)
    out["xPts"] = np.maximum(xpts, 0.0)

    # manual role string for UI
    out["manual_role"] = out["web_name"].map(lambda n: MANUAL_ROLE_OVERRIDES.get(str(n), ""))

    # attach team_name for display
    if "team_name" in teams.columns:
        out = out.merge(teams[["team_id","team_name"]], on="team_id", how="left")

    # integrate news buzz if available
    try:
        from . import news_state
        buzz_map = getattr(news_state, "BUZZ", {})
    except Exception:
        buzz_map = {}
    if buzz_map:
        out["buzz"] = out["web_name"].map(lambda n: buzz_map.get(str(n), 0.0))
        out["xPts"] = (out["xPts"] + NEWS_BUZZ_WEIGHT * out["buzz"]).clip(lower=0.0)
    else:
        out["buzz"] = 0.0

    return out
