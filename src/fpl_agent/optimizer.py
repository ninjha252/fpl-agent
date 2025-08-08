from __future__ import annotations
import pulp as pl
import pandas as pd
from typing import Dict, List, Tuple
from .utils import SquadRules

class SquadOptimizer:
    def __init__(self, rules: SquadRules):
        self.rules = rules

    def pick_initial_squad(self, pool: pd.DataFrame) -> Dict[str, List[int]]:
        # pool: player_id, position, cost, xPts, team_id
        prob = pl.LpProblem("initial_squad", pl.LpMaximize)
        ids = pool.player_id.tolist()

        y = pl.LpVariable.dicts("y", ids, lowBound=0, upBound=1, cat=pl.LpBinary)

        # Objective: maximize projected points with a tiny nudge toward cheaper value
        prob += pl.lpSum([
            y[i] * (float(pool.loc[pool.player_id==i, "xPts"].values[0]) +
                    0.02 * float(pool.loc[pool.player_id==i, "cost"].values[0]))
            for i in ids
        ])

        # Budget
        prob += pl.lpSum([y[i] * float(pool.loc[pool.player_id==i, "cost"].values[0]) for i in ids]) <= self.rules.budget

        # Squad size by position
        for pos, need in {"GK": self.rules.gk, "DEF": self.rules.df, "MID": self.rules.md, "FWD": self.rules.fw}.items():
            pos_ids = pool[pool.position==pos].player_id.tolist()
            prob += pl.lpSum([y[i] for i in pos_ids]) == need

        # Max 3 per club
        for t in pool.team_id.unique():
            t_ids = pool[pool.team_id==t].player_id.tolist()
            prob += pl.lpSum([y[i] for i in t_ids]) <= self.rules.max_from_team

        prob.solve(pl.PULP_CBC_CMD(msg=False))
        chosen = [i for i in ids if y[i].value() == 1]

        sub_pool = pool[pool.player_id.isin(chosen)].copy()
        xi, bench, captain, vice = self._pick_xi_captain(sub_pool)
        return {"squad15": chosen, "xi": xi, "bench": bench, "captain": captain, "vice": vice}

    def _pick_xi_captain(self, squad15: pd.DataFrame) -> Tuple[List[int], List[int], int, int]:
        prob = pl.LpProblem("xi_and_cap", pl.LpMaximize)
        ids = squad15.player_id.tolist()
        s = pl.LpVariable.dicts("s", ids, lowBound=0, upBound=1, cat=pl.LpBinary)
        c = pl.LpVariable.dicts("c", ids, lowBound=0, upBound=1, cat=pl.LpBinary)
        v = pl.LpVariable.dicts("v", ids, lowBound=0, upBound=1, cat=pl.LpBinary)

        xpts = {i: float(squad15.loc[squad15.player_id==i, "xPts"].values[0]) for i in ids}

        prob += pl.lpSum([s[i] * xpts[i] for i in ids]) + pl.lpSum([c[i] * xpts[i] for i in ids])

        prob += pl.lpSum([s[i] for i in ids]) == 11

        for pos, min_need in {"DEF": 3, "MID": 2, "FWD": 1}.items():
            pos_ids = squad15[squad15.position==pos].player_id.tolist()
            prob += pl.lpSum([s[i] for i in pos_ids]) >= min_need

        prob += pl.lpSum([c[i] for i in ids]) == 1
        prob += pl.lpSum([v[i] for i in ids]) == 1
        for i in ids:
            prob += c[i] <= s[i]
            prob += v[i] <= s[i]

        prob.solve(pl.PULP_CBC_CMD(msg=False))
        xi = [i for i in ids if s[i].value() == 1]
        bench = [i for i in ids if i not in xi]
        captain = next(i for i in ids if c[i].value() == 1)
        vice = next(i for i in ids if v[i].value() == 1)
        return xi, bench, captain, vice
# optimizer.py placeholder
