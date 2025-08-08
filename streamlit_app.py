from __future__ import annotations

# --- Path fix so imports work on Streamlit Cloud ---
import os, sys
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd

from src.fpl_agent.data import load_bootstrap, load_fixtures
from src.fpl_agent.projections import expected_points_next_gw
from src.fpl_agent.optimizer import SquadOptimizer
from src.fpl_agent.utils import SquadRules

# knobs & buzz
from src.fpl_agent import projections as proj_mod
from src.fpl_agent import news_state
from src.fpl_agent.news_signals import build_buzz_map
# Optional FREE odds (best-effort; can be toggled off)
from src.fpl_agent.odds_adapter_free import fetch_match_odds as fetch_odds_free

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="FPL AI Agent", layout="wide")
st.title("🏆 FPL AI Agent – MVP")

with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Initial Budget (for new squad)", 95.0, 110.0, 100.0, 0.5)
    build_mode = st.radio("Mode", ["Build initial 15", "Optimize my 15"], horizontal=True)

    st.markdown("### Starters & penalties")
    nailedness = st.slider("Nailedness scale", 0.5, 1.5, 1.0, 0.05)
    hide_thresh = st.slider("Hide backups below prob", 0.0, 0.8, 0.35, 0.05)
    manual_backups = st.text_input("Manual backups (comma-separated)", "Arrizabalaga,Kepa")

    st.markdown("### Form & Odds")
    form_n = st.slider("Form lookback (games)", 3, 10, 6, 1)
    odds_on = st.checkbox("Use FREE bookmaker odds (experimental)", value=False)  # optional now
    odds_w = st.slider("Odds weight", 0.0, 1.5, 0.6, 0.05)

    st.markdown("### News & Buzz")
    use_buzz = st.checkbox("Incorporate Reddit & news buzz", value=False)
    buzz_weight = st.slider("Buzz weight added to xPts", 0.0, 1.5, 0.5, 0.05)
    subs_text = st.text_input(
        "Subreddits (comma-separated)",
        "FantasyPL,FantasyPremierLeague,PremierLeague,soccer"
    )
    subs = [s.strip() for s in subs_text.split(",") if s.strip()]

st.info(
    "Projections use official FPL data + starter probability, team form, optional FREE odds & news buzz. "
    "XI obeys FPL rules (1 GK, ≥3 DEF, ≥3 MID, ≥1 FWD)."
)

# ---------------- LOAD DATA ----------------
with st.spinner("Loading FPL data..."):
    players, teams, events = load_bootstrap()
    fixtures = load_fixtures()

# ---------------- APPLY KNOBS ----------------
proj_mod.set_nailedness_scale(nailedness)
proj_mod.set_starter_hide_threshold(hide_thresh)
proj_mod.set_form_lookback(form_n)
proj_mod.set_odds_weight(odds_w)
proj_mod.NEWS_BUZZ_WEIGHT = buzz_weight

# Manual backups injection
for name in [n.strip() for n in manual_backups.split(",") if n.strip()]:
    proj_mod.set_manual_role_override(name, "backup")

# Buzz (optional)
news_state.BUZZ = {}
if use_buzz:
    st.toast("Fetching buzz…", icon="📰")
    names = players["web_name"].astype(str).tolist()
    news_state.BUZZ = build_buzz_map(st.secrets, names, reddit_subs=subs)

# FREE odds prefetch (optional)
if odds_on:
    with st.spinner("Fetching free odds (best-effort)…"):
        try:
            st.session_state["_free_odds_df"] = fetch_odds_free(teams[["team_id","team_name"]])
        except Exception:
            st.session_state["_free_odds_df"] = pd.DataFrame()
else:
    st.session_state["_free_odds_df"] = pd.DataFrame()

# ---------------- PROJECTIONS ----------------
proj = expected_points_next_gw(players, teams, fixtures)
# Merge free odds just to display (projections may already have used some odds internally)
if isinstance(st.session_state.get("_free_odds_df"), pd.DataFrame) and not st.session_state["_free_odds_df"].empty:
    proj = proj.merge(st.session_state["_free_odds_df"], on="team_name", how="left")

if hide_thresh > 0:
    sp = proj.get("starter_prob", 1.0)
    proj = proj[sp >= hide_thresh].copy()

def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

# ---------------- SEARCH ----------------
with st.expander("🔎 Search players / teams"):
    q = st.text_input("Type a player or team name")
    if q.strip():
        ql = q.lower()
        mask = (
            proj["web_name"].astype(str).str.lower().str.contains(ql, na=False)
            | proj.get("team_name", pd.Series("", index=proj.index)).astype(str).str.lower().str.contains(ql, na=False)
        )
        res = proj.loc[mask].sort_values("xPts", ascending=False)
        st.dataframe(
            res[safe_cols(res, ["player_id","web_name","position","team_name","cost","starter_prob","win_prob","xPts"])].head(100),
            use_container_width=True
        )

# ---------------- MAIN TABLE ----------------
st.subheader("Projected points (next GW)")
display_cols = safe_cols(
    proj,
    ["player_id", "web_name", "position", "team_name", "cost", "starter_prob", "win_prob", "buzz", "xPts"]
)
st.dataframe(
    proj.sort_values("xPts", ascending=False)[display_cols].head(50),
    use_container_width=True
)

# ---------------- SQUAD BUILDER / OPTIMIZER ----------------
rules = SquadRules(budget=budget)
opt = SquadOptimizer(rules)

if build_mode == "Build initial 15":
    st.subheader("Initial Squad Builder")
    pool = proj.copy()
    if "status" in pool.columns:
        pool = pool[~pool.status.isin(["i", "o"])].copy()

    if st.button("Build Squad"):
        res = opt.pick_initial_squad(pool)
        chosen = proj.merge(pd.DataFrame({"player_id": res["squad15"]}), on="player_id")
        xi = chosen[chosen.player_id.isin(res["xi"])].copy()
        bench = chosen[chosen.player_id.isin(res["bench"])].copy()
        cap = chosen[chosen.player_id == res["captain"]].iloc[0]
        vice = chosen[chosen.player_id == res["vice"]].iloc[0]

        xi = xi.sort_values(["team_name", "xPts"], ascending=[True, False])
        bench = bench.sort_values(["team_name", "xPts"], ascending=[True, False])

        xi_cols = safe_cols(xi, ["web_name", "position", "team_name", "cost", "xPts"])
        bench_cols = safe_cols(bench, ["web_name", "position", "team_name", "cost", "xPts"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Starting XI + C/VC")
            st.dataframe(xi[xi_cols], use_contai]()
