from __future__ import annotations

import os, sys
# Add the *src* folder to Python's search path
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import streamlit as st
import pandas as pd

from fpl_agent.data import load_bootstrap, load_fixtures
from fpl_agent.projections import expected_points_next_gw
from fpl_agent.optimizer import SquadOptimizer
from fpl_agent.utils import SquadRules



st.set_page_config(page_title="FPL AI Agent", layout="wide")
st.title("🏆 FPL AI Agent – MVP")

with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Initial Budget (for new squad)", 95.0, 110.0, 100.0, 0.5)
    allow_hits = st.checkbox("Allow points hits? (visual only for now)", True)
    max_transfers = st.slider("Max transfers this week (UI only for now)", 0, 4, 1)
    price_weight = st.slider("λ: price momentum weight (UI only for now)", 0.0, 1.0, 0.2, 0.05)
    build_mode = st.radio("Mode", ["Build initial 15", "Optimize my 15"], horizontal=True)

st.info("Data loads from the official FPL API. Projections are a fast proxy; you can upgrade the model later.")

with st.spinner("Loading FPL data..."):
    players, teams, events = load_bootstrap()
    fixtures = load_fixtures()

proj = expected_points_next_gw(players, teams, fixtures)

st.subheader("Projected points (next GW)")
st.dataframe(proj.sort_values("xPts", ascending=False).head(50), use_container_width=True)

rules = SquadRules(budget=budget)
opt = SquadOptimizer(rules)

if build_mode == "Build initial 15":
    st.subheader("Initial Squad Builder")
    pool = proj.copy()
    # Drop out/injured if you want a cleaner initial team
    pool = pool[~pool.status.isin(["i", "o"])].copy()

    if st.button("Build Squad"):
        res = opt.pick_initial_squad(pool)
        chosen = proj.merge(pd.DataFrame({"player_id": res["squad15"]}), on="player_id")
        xi = chosen[chosen.player_id.isin(res["xi"])].copy()
        bench = chosen[chosen.player_id.isin(res["bench"])].copy()
        cap = chosen[chosen.player_id == res["captain"]].iloc[0]
        vice = chosen[chosen.player_id == res["vice"]].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Starting XI + C/VC")
            st.dataframe(xi[["web_name", "position", "team_id", "cost", "xPts"]].sort_values("position"))
            st.success(f"Captain: {cap.web_name} | Vice: {vice.web_name}")
        with col2:
            st.markdown("### Bench")
            st.dataframe(bench[["web_name", "position", "team_id", "cost", "xPts"]])

else:
    st.subheader("Optimize my 15 (paste current squad)")
    st.caption("Paste your 15 player IDs (comma-separated). IDs are in the table above (column: player_id).")
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

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Starting XI + C/VC")
                st.dataframe(xi_df[["web_name", "position", "team_id", "cost", "xPts"]].sort_values("position"))
                st.success(f"Captain: {cap.web_name} | Vice: {v.web_name}")
            with col2:
                st.markdown("### Bench")
                st.dataframe(bench_df[["web_name", "position", "team_id", "cost", "xPts"]])
        except Exception as e:
            st.error(str(e))

st.divider()
st.markdown(
    "#### Coming next: transfer planner with hits & price-chasing λ\n"
    "- Compare 0-hit vs 1-hit plans and expected net gain (points minus hit).\n"
    "- Optional value building via price momentum.\n"
)
# Streamlit main app placeholder
