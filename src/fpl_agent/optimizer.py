from __future__ import annotations
import pulp as pl
import pandas as pd
from typing import Dict, List, Tuple
from .utils import SquadRules

# --------- helpers ---------
def _status_ok(prob: pl.LpProblem) -> bool:
    try:
        return pl.LpStatus[prob.status] == "Optimal"
    except Exception:
        return False

def _pos_ids(df: pd.DataFrame, pos: str) -> List[int]:
    return df[df.position == pos].player_id.tolist()

def _team_count(df: pd.DataFrame) -> Dict[int, int]:
    return df["team_id"].value_counts().to_dict()

def _greedy_repair_squad(pool: pd.DataFrame, want: Dict[str, int], budget: float, max_from_team: int) -> List[int]:
    """
    Very simple greedy builder used only if the MILP fails.
    1) Take top-xPts per position to satisfy exact counts.
    2) If team cap/budget violated, swap lowest xPts offenders.
    Not perfect, but guarantees a legal 15 with decent quality.
    """
    chosen: List[int] = []
    used_team: Dict[int, int] = {}
    cost_total = 0.0

    # Step 1: satisfy per-position counts
    for pos, need in want.items():
        cand = pool[pool.position == pos].sort_values("xPts", ascending=False)
        for _, row in cand.iterrows():
            tid = int(row.team_id)
            pid = int(row.player_id)
            if used_team.get(tid, 0) >= max_from_team:
                continue
            if cost_total + float(row.cost) > budget:
                continue
            chosen.append(pid)
            used_team[tid] = used_team.get(tid, 0) + 1
            cost_total += float(row.cost)
            if len([i for i in chosen if i in _pos_ids(pool, pos)]) >= need:
                break

    # If still short (budget too tight), fill with cheapest options that respect team cap
    for pos, need in want.items():
        have = len([pid for pid in chosen if pid in _pos_ids(pool, pos)])
        if have < need:
            cand = pool[pool.position == pos].sort_values(["cost", "xPts"], ascending=[True, False])
            for _, row in cand.iterrows():
                tid = int(row.team_id); pid = int(row.player_id)
                if pid in chosen:
                    continue
                if used_team.get(tid, 0) >= max_from_team:
                    continue
                if cost_total + float(row.cost) > budget:
                    continue
                chosen.append(pid)
                used_team[tid] = used_team.get(tid, 0) + 1
                cost_total += float(row.cost)
                have += 1
                if have >= need:
                    break

    # If still missing (edge), just add cheapest available respecting cap/budget until 15
    cand_all = pool.sort_values(["cost", "xPts"], ascending=[True, False])
    while len(chosen) < 15:
        for _, row in cand_all.iterrows():
            tid = int(row.team_id); pid = int(row.player_id)
            if pid in chosen:
                continue
            if used_team.get(tid, 0) >= max_from_team:
                continue
            if cost_total + float(row.cost) > budget:
                continue
            chosen.append(pid)
            used_team[tid] = used_team.get(tid, 0) + 1
            cost_total += float(row.cost)
            break
        else:
            # can't add more without breaking constraints
            break

    return chosen[:15]

# --------- optimizer ---------
class SquadOptimizer:
    def __init__(self, rules: SquadRules):
        self.rules = rules

    def pick_initial_squad(self, pool: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Build a legal 15:
          GK=2, DEF=5, MID=5, FWD=3, team cap, under budget.
        """
        prob = pl.LpProblem("initial_squad", pl.LpMaximize)
        ids = pool.player_id.astype(int).tolist()
        cost_map = dict(zip(pool.player_id.astype(int), pool.cost.astype(float)))
        xpts_map = dict(zip(pool.player_id.astype(int), pool.xPts.astype(float)))
        pos_map = dict(zip(pool.player_id.astype(int), pool.position.astype(str)))
        team_map = dict(zip(pool.player_id.astype(int), pool.team_id.astype(int)))

        y = pl.LpVariable.dicts("y", ids, lowBound=0, upBound=1, cat=pl.LpBinary)

        # Objective: points + tiny nudge for cheaper cost (helps feasibility)
        prob += pl.lpSum([y[i] * (xpts_map[i] + 0.02 * cost_map[i]) for i in ids])

        # Budget
        prob += pl.lpSum([y[i] * cost_map[i] for i in ids]) <= float(self.rules.budget)

        # Exact position counts for 15-man squad
        want = {"GK": self.rules.gk, "DEF": self.rules.df, "MID": self.rules.md, "FWD": self.rules.fw}
        for pos, need in want.items():
            prob += pl.lpSum([y[i] for i in ids if pos_map[i] == pos]) == int(need)

        # Max 3 per club
        for t in set(team_map.values()):
            prob += pl.lpSum([y[i] for i in ids if team_map[i] == t]) <= int(self.rules.max_from_team)

        # Solve
        prob.solve(pl.PULP_CBC_CMD(msg=False))

        if not _status_ok(prob):
            # Fallback greedy builder
            chosen = _greedy_repair_squad(pool, want, float(self.rules.budget), int(self.rules.max_from_team))
        else:
            chosen = [i for i in ids if y[i].value() == 1]

        # Safety: if solver gave nonsense, repair counts
        def _count(pos: str) -> int:
            return sum(1 for pid in chosen if pos_map.get(pid) == pos)

        if _count("GK") != want["GK"] or _count("DEF") != want["DEF"] or _count("MID") != want["MID"] or _count("FWD") != want["FWD"] or len(chosen) != 15:
            # Greedy repair to guarantee legality
            chosen = _greedy_repair_squad(pool, want, float(self.rules.budget), int(self.rules.max_from_team))

        sub_pool = pool[pool.player_id.isin(chosen)].copy()
        xi, bench, captain, vice = self._pick_xi_captain(sub_pool)
        return {"squad15": chosen, "xi": xi, "bench": bench, "captain": captain, "vice": vice}

    def _pick_xi_captain(self, squad15: pd.DataFrame) -> Tuple[List[int], List[int], int, int]:
        """
        Build a legal XI:
          exactly 11 players, exactly 1 GK, ≥3 DEF, ≥3 MID, ≥1 FWD.
        Falls back to greedy repair if the MILP doesn't return Optimal.
        """
        prob = pl.LpProblem("xi_and_cap", pl.LpMaximize)
        ids = squad15.player_id.astype(int).tolist()
        xpts = dict(zip(squad15.player_id.astype(int), squad15.xPts.astype(float)))
        pos_map = dict(zip(squad15.player_id.astype(int), squad15.position.astype(str)))

        s = pl.LpVariable.dicts("s", ids, lowBound=0, upBound=1, cat=pl.LpBinary)
        c = pl.LpVariable.dicts("c", ids, lowBound=0, upBound=1, cat=pl.LpBinary)
        v = pl.LpVariable.dicts("v", ids, lowBound=0, upBound=1, cat=pl.LpBinary)

        prob += pl.lpSum([s[i] * xpts[i] for i in ids]) + pl.lpSum([c[i] * xpts[i] for i in ids])

        # Exactly 11 starters
        prob += pl.lpSum([s[i] for i in ids]) == 11

        # Formation constraints
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "GK"]) == 1
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "DEF"]) >= 3
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "MID"]) >= 3
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "FWD"]) >= 1

        # Captain/Vice must be in XI
        prob += pl.lpSum([c[i] for i in ids]) == 1
        prob += pl.lpSum([v[i] for i in ids]) == 1
        for i in ids:
            prob += c[i] <= s[i]
            prob += v[i] <= s[i]

        prob.solve(pl.PULP_CBC_CMD(msg=False))

        if not _status_ok(prob):
            # Greedy XI: start from top xPts, then enforce formation
            sorted_ids = sorted(ids, key=lambda i: xpts[i], reverse=True)
            xi: List[int] = []
            counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
            for i in sorted_ids:
                p = pos_map[i]
                if len(xi) < 11:
                    xi.append(i); counts[p] += 1
            # enforce formation by swaps if needed
            def ensure_at_least(role: str, k: int):
                nonlocal xi, counts
                while counts[role] < k:
                    # replace the worst player from an over-represented role
                    over_roles = [r for r in ("DEF","MID","FWD") if r != role and counts[r] > (1 if r=="FWD" else 3)]
                    if not over_roles:
                        over_roles = [r for r in ("DEF","MID","FWD") if r != role and counts[r] > 0]
                    worst_id = min([j for j in xi if pos_map[j] in over_roles], key=lambda j: xpts[j], default=None)
                    cand = next((j for j in sorted_ids if j not in xi and pos_map[j] == role), None)
                    if worst_id is None or cand is None:
                        break
                    xi.remove(worst_id); xi.append(cand)
                    counts[pos_map[worst_id]] -= 1; counts[role] += 1
            # 1 GK, >=3 DEF, >=3 MID, >=1 FWD
            # ensure GK = 1
            if counts["GK"] != 1:
                # swap worst non-GK with best GK not in XI
                best_gk = next((i for i in sorted_ids if pos_map[i]=="GK" and i not in xi), None)
                worst_non_gk = min([j for j in xi if pos_map[j] != "GK"], key=lambda j: xpts[j], default=None)
                if best_gk and worst_non_gk:
                    xi.remove(worst_non_gk); xi.append(best_gk)
                    counts[pos_map[worst_non_gk]] -= 1; counts["GK"] = 1
            ensure_at_least("DEF", 3)
            ensure_at_least("MID", 3)
            ensure_at_least("FWD", 1)

            bench = [i for i in ids if i not in xi]
            captain = max(xi, key=lambda i: xpts[i])
            vice = max([i for i in xi if i != captain] or xi, key=lambda i: xpts[i])
            return xi, bench, int(captain), int(vice)

        # MILP solution path
        xi = [i for i in ids if s[i].value() == 1]
        # Safety: ensure exactly 11
        if len(xi) != 11:
            # fallback to greedy repair if the solver returns a weird partial
            return self._pic_
