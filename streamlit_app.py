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
st.title("🏆 FPL AI Agent – MVP (faster)")

# --------- helpers ---------
def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

@st.cache_data(ttl=1800)  # 30 minutes
def _get_fpl_data():
    players, teams, events = load_bootstrap()
    fixtures = load_fixtures()
    return players, teams, events, fixtures

@st.cache_data(ttl=900)  # 15 minutes
def _get_free_odds(teams_df: pd.DataFrame):
    try:
        return fetch_odds_free(teams_df[["team_id", "team_name"]])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)  # 15 minutes
def _get_buzz(names: list[str], subs: list[str], secrets: dict):
    try:
        return build_buzz_map(secrets, names, reddit_subs=subs)
    except Exception:
        return {}

@st.cache_data(ttl=900)  # 15 minutes
def _compute_proj(players: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame,
                  nailedness: float, hide_thresh: float, form_n: int,
                  odds_w: float, buzz_w: float,
                  odds_df: pd.DataFrame | None, buzz_map: dict | None):
    # set knobs (kept module-level for simplicity)
    proj_mod.set_nailedness_scale(nailedness)
    proj_mod.set_starter_hide_threshold(hide_thresh)
    proj_mod.set_form_lookback(form_n)
    proj_mod.set_odds_weight(odds_w)
    proj_mod.NEWS_BUZZ_WEIGHT = buzz_w

    # pass optional odds/buzz directly so projections doesn’t scrape
    df = expected_points_next_gw(players, teams, fixtures, odds_df=odds_df, buzz_map=buzz_map)
    return df

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Initial Budget (for new squad)", 95.0, 110.0, 100.0, 0.5)
    mode = st.radio("Mode", ["Build initial 15", "Optimize my 15"], horizontal=True)

    st.markdown("### Starters & penalties")
    nailedness = st.slider("Nailedness scale", 0.5, 1.5, 1.0, 0.05)
    hide_thresh = st.slider("Hide backups below prob", 0.0, 0.8, 0.35, 0.05)
    manual_backups = st.text_input("Manual backups (comma-separated)", "Arrizabalaga,Kepa")

    st.markdown("### Form & Odds")
    form_n = st.slider("Form lookback (games)", 3, 10, 6, 1)
    odds_on = st.checkbox("Use FREE bookmaker odds (experimental)", value=False)  # optional
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
    "Faster build: heavy stuff is cached; odds & buzz are optional. "
    "XI obeys FPL rules (1 GK, ≥3 DEF, ≥3 MID, ≥1 FWD)."
)

# ---------------- LOAD DATA (cached) ----------------
with st.spinner("Loading FPL data..."):
    players, teams, events, fixtures = _get_fpl_data()

# manual backup overrides
for name in [n.strip() for n in manual_backups.split(",") if n.strip()]:
    proj_mod.set_manual_role_override(name, "backup")

# Optional buzz (cached)
buzz_map = {}
if use_buzz:
    st.toast("Fetching buzz…", icon="📰")
    names = players["web_name"].astype(str).tolist()
    buzz_map = _get_buzz(names, subs, st.secrets)

# Optional free odds (cached)
free_odds_df = pd.DataFrame()
if odds_on:
    with st.spinner("Fetching free odds (best-effort)…"):
        free_odds_df = _get_free_odds(teams)

# ---------------- PROJECTIONS (cached) ----------------
proj = _compute_proj(
    players, teams, fixtures,
    nailedness=nailedness, hide_thresh=hide_thresh, form_n=form_n,
    odds_w=odds_w, buzz_w=buzz_weight,
    odds_df=free_odds_df if odds_on else None,
    buzz_map=buzz_map if use_buzz else None
)

# Merge odds into display (nice to see, even if weights used internally)
if odds_on and not free_odds_df.empty and "team_name" in proj.columns:
    proj = proj.merge(free_odds_df, on="team_name", how="left")

# Hide obvious backups from table view if desired
if hide_thresh > 0 and "starter_prob" in proj.columns:
    proj = proj[proj["starter_prob"] >= hide_thresh].copy()

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

if mode == "Build initial 15":
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
            st.dataframe(xi[xi_cols], use_container_width=True)
            st.success(f"Captain: {cap.web_name} | Vice: {vice.web_name}")
        with col2:
            st.markdown("### Bench")
            st.dataframe(bench[bench_cols], use_container_width=True)

else:
    st.subheader("Optimize my 15 (paste current squad)")
    st.caption("Paste your **15 player IDs** (comma-separated). Use the `player_id` column from the table above.")
    ids_text = st.text_area("Your 15 player IDs", "")

    action = st.radio("Choose action", ["Best XI", "Suggest transfers"], horizontal=True)

    with st.expander("⚙️ Transfer settings (used for Suggest transfers)"):
        bank = st.number_input("Bank (£m)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        free_transfers = st.number_input("Free transfers", min_value=0, max_value=8, value=1, step=1)
        max_changes = st.number_input("Max changes to consider", min_value=1, max_value=15, value=2, step=1)
        hit_cost = st.number_input("Hit cost per extra transfer", min_value=0, max_value=8, value=4, step=1)

    if action == "Best XI":
        if st.button("Optimize XI + C/VC"):
            try:
                ids = [int(x.strip()) for x in ids_text.split(",") if x.strip()]
                assert len(ids) == 15, "You must provide exactly 15 player IDs."
                chosen = proj[proj.player_id.isin(ids)].copy()
                xi, bench, captain, vice = opt._pick_xi_captain(chosen)

                xi_df = chosen[chosen.player_id.isin(xi)].sort_values(["team_name", "xPts"], ascending=[True, False])
                bench_df = chosen[chosen.player_id.isin(bench)].sort_values(["team_name", "xPts"], ascending=[True, False])

                xi_cols = safe_cols(xi_df, ["web_name", "position", "team_name", "cost", "xPts"])
                bench_cols = safe_cols(bench_df, ["web_name", "position", "team_name", "cost", "xPts"])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Starting XI + C/VC")
                    st.dataframe(xi_df[xi_cols], use_container_width=True)
                    cap = chosen[chosen.player_id == captain].iloc[0]
                    v = chosen[chosen.player_id == vice].iloc[0]
                    st.success(f"Captain: {cap.web_name} | Vice: {v.web_name}")
                with col2:
                    st.markdown("### Bench")
                    st.dataframe(bench_df[bench_cols], use_container_width=True)
            except Exception as e:
                st.error(str(e))

    else:
        if st.button("💹 Suggest transfers"):
            try:
                ids = [int(x.strip()) for x in ids_text.split(",") if x.strip()]
                assert len(ids) == 15, "You must provide exactly 15 player IDs."
                current = proj[proj.player_id.isin(ids)].copy()

                result = opt.suggest_transfers(
                    current15=current,
                    pool=proj,
                    bank=float(bank),
                    max_changes=int(max_changes),
                    free_transfers=int(free_transfers),
                    hit_cost=int(hit_cost),
                    team_cap=3,
                )
                sugg = result["suggestions"]
                if not sugg:
                    st.info("No profitable transfers found under the current settings.")
                else:
                    st.markdown("### Suggested transfers")
                    df = pd.DataFrame([{
                        "Out": f"{s['out_name']} ({s['out_team']})",
                        "In":  f"{s['in_name']} ({s['in_team']})",
                        "Cost Δ": round(s["cost_diff"], 1),
                        "Raw gain": round(s["raw_gain"], 3),
                        "Net gain (after hits)": round(s["net_gain"], 3),
                    } for s in sugg])
                    st.dataframe(df, use_container_width=True)
                    st.success(
                        f"Projected XI points after transfers: {result['final_points']:.3f} | "
                        f"Remaining bank: £{result['remaining_bank']:.1f}m | "
                        f"Changes applied: {result['changes']}"
                    )
            except Exception as e:
                st.error(str(e))

st.divider()
st.caption(
    "Speed tips: odds/news are cached & optional. Changing sliders invalidates only what’s needed; "
    "base FPL data is reused for 30 min."
)
