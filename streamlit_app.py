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
    odds_on = st.checkbox("Use bookmaker odds (needs ODDS_API_KEY)", value=False)
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
    "Projections use official FPL data + starter probability, team form, optional odds & news buzz. "
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

# Manual backups injection (quick override)
for name in [n.strip() for n in manual_backups.split(",") if n.strip()]:
    proj_mod.set_manual_role_override(name, "backup")

# Pass ODDS_API_KEY from secrets to env so odds_adapter can see it
if odds_on and "odds" in st.secrets and "api_key" in st.secrets["odds"]:
    os.environ["ODDS_API_KEY"] = st.secrets["odds"]["api_key"]

# Build news buzz map (optional)
news_state.BUZZ = {}
if use_buzz:
    st.toast("Fetching buzz…", icon="📰")
    names = players["web_name"].astype(str).tolist()
    news_state.BUZZ = build_buzz_map(st.secrets, names, reddit_subs=subs)

# ---------------- PROJECTIONS ----------------
proj = expected_points_next_gw(players, teams, fixtures)
if hide_thresh > 0:
    # Filter extreme backups out of the table view
    sp = proj.get("starter_prob", 1.0)
    proj = proj[sp >= hide_thresh].copy()

# ---- Safe column selection helper ----
def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

display_cols = safe_cols(
    proj,
    ["player_id", "web_name", "position", "team_name", "cost", "starter_prob", "buzz", "xPts"]
)

st.subheader("Projected points (next GW)")
st.dataframe(
    proj.sort_values("xPts", ascending=False)[display_cols].head(50),
    use_container_width=True
)

# ---------------- SQUAD BUILDER / XI OPT ----------------
rules = SquadRules(budget=budget)
opt = SquadOptimizer(rules)

if build_mode == "Build initial 15":
    st.subheader("Initial Squad Builder")
    pool = proj.copy()
    # Drop players ruled out/injured for sanity
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
    st.caption("Paste your 15 player IDs (comma-separated). Find IDs in the table above (column: player_id).")
    ids_text = st.text_area("Your 15 player IDs", "")

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

st.divider()
st.markdown(
    "#### Notes\n"
    "- Backups (esp GK) get penalized; use **Manual backups** box to force known backups.\n"
    "- Toggle **Buzz** to include Reddit/RSS sentiment; add your subs in the sidebar.\n"
    "- **Odds** require an API key; set `odds.api_key` in Streamlit Secrets.\n"
)
