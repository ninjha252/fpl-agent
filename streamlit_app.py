from __future__ import annotations

import os, sys
# Put repo root (contains "src") on path so imports work on Streamlit Cloud
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
from src.fpl_agent.news_signals import fetch_reddit_buzz, fetch_rss_buzz, combine_buzz

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

    st.markdown("### News & Buzz")
    use_buzz = st.checkbox("Incorporate Reddit & news buzz", value=False)
    buzz_weight = st.slider("Buzz weight added to xPts", 0.0, 1.5, 0.5, 0.05)

st.info("Projections use official FPL data + starter probability + optional buzz from Reddit/News.")

with st.spinner("Loading FPL data..."):
    players, teams, events = load_bootstrap()
    fixtures = load_fixtures()

# Apply knobs to projection module
proj_mod.set_nailedness_scale(nailedness)
proj_mod.set_starter_hide_threshold(hide_thresh)
proj_mod.NEWS_BUZZ_WEIGHT = buzz_weight

# Manual backups injection
for name in [n.strip() for n in manual_backups.split(",") if n.strip()]:
    proj_mod.set_manual_role_override(name, "backup")

# Fetch buzz if requested
news_state.BUZZ = {}
if use_buzz:
    st.toast("Fetching buzz…", icon="📰")
    names = players["web_name"].astype(str).tolist()
    rss = fetch_rss_buzz(names)
    reddit = fetch_reddit_buzz(st.secrets, names) if "reddit" in st.secrets else {}
    news_state.BUZZ = combine_buzz(reddit, rss)

# Compute projections
proj = expected_points_next_gw(players, teams, fixtures)
if hide_thresh > 0:
    proj = proj[proj.get("starter_prob", 1.0) >= hide_thresh].copy()

# ---- Safe column selection (prevents KeyError) ----
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

# -------- Squad builder / optimizer --------
rules = SquadRules(budget=budget)
opt = SquadOptimizer(rules)

if build_mode == "Build initial 15":
    st.subheader("Initial Squad Builder")
    pool = proj.copy()
    # Drop players marked out/injured for initial squad sanity
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
    st.caption("Paste your 15 player IDs (comma-separated). Find IDs in the table above.")
    ids_text = st.text_area("Your 15 player IDs", "")

    if st.button("Optimize XI + C/VC"):
        try:
            ids = [int(x.strip()) for x in ids_text.split(",") if x.strip()]
            assert len(ids) == 15, "You must provide exactly 15 player IDs."
            chosen = proj[proj.player_id.isin(ids)].copy()
            xi, bench, captain, vice = opt._pick_xi_captain(chosen)
            xi_df = chosen[chosen.player_id.isin(xi)]
            bench_df = chosen[chosen.player_id.isin(bench)]
            cap = chosen[chosen.player_id == captain].iloc[0]
            v = chosen[chosen.player_id == vice].iloc[0]

            xi_df = xi_df.sort_values(["team_name", "xPts"], ascending=[True, False])
            bench_df = bench_df.sort_values(["team_name", "xPts"], ascending=[True, False])

            xi_cols = safe_cols(xi_df, ["web_name", "position", "team_name", "cost", "xPts"])
            bench_cols = safe_cols(bench_df, ["web_name", "position", "team_name", "cost", "xPts"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Starting XI + C/VC")
                st.dataframe(xi_df[xi_cols], use_container_width=True)
                st.success(f"Captain: {cap.web_name} | Vice: {v.web_name}")
            with col2:
                st.markdown("### Bench")
                st.dataframe(bench_df[bench_cols], use_container_width=True)
        except Exception as e:
            st.error(str(e))

st.divider()
st.markdown(
    "#### Notes\n"
    "- Backups (esp GK) are heavily penalized.\n"
    "- Toggle Reddit/News buzz in the sidebar to nudge xPts.\n"
    "- Team names are shown for quick stack checks.\n"
)
