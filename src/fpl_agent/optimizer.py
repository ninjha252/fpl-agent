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

def _coerce_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns and dtypes exist."""
    need = {"player_id", "position", "team_id", "cost", "xPts"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Optimizer pool missing columns: {sorted(missing)}")

    out = df.copy()
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["team_id"]   = pd.to_numeric(out["team_id"],   errors="coerce").astype("Int64")
    out["cost"]      = pd.to_numeric(out["cost"],      errors="coerce").astype(float)
    out["xPts"]      = pd.to_numeric(out["xPts"],      errors="coerce").astype(float)
    out["position"]  = out["position"].astype(str)
    out = out.dropna(subset=["player_id", "team_id", "cost", "xPts"]).copy()
    out["player_id"] = out["player_id"].astype(int)
    out["team_id"]   = out["team_id"].astype(int)
    return out

def _pos_ids(df: pd.DataFrame, pos: str) -> List[int]:
    return df[df.position == pos].player_id.astype(int).tolist()

def _greedy_repair_squad(pool: pd.DataFrame, want: Dict[str, int], budget: float, max_from_team: int) -> List[int]:
    """Greedy builder used only if MILP fails; guarantees legal 15 or raises if impossible."""
    pool = _coerce_pool(pool)
    chosen: List[int] = []
    used_team: Dict[int, int] = {}
    cost_total = 0.0

    # Phase 1: fill per position with best xPts
    for pos, need in want.items():
        cand = pool[pool.position == pos].sort_values("xPts", ascending=False)
        if cand.empty:
            raise ValueError(f"No candidates found for position {pos}")
        for _, row in cand.iterrows():
            tid, pid, pcost = int(row.team_id), int(row.player_id), float(row.cost)
            if used_team.get(tid, 0) >= max_from_team:
                continue
            if cost_total + pcost > budget:
                continue
            if pid in chosen:
                continue
            chosen.append(pid)
            used_team[tid] = used_team.get(tid, 0) + 1
            cost_total += pcost
            if len([i for i in chosen if i in _pos_ids(pool, pos)]) >= need:
                break

    # Phase 2: top up to 15 by cheapest decent picks
    cand_all = pool.sort_values(["cost", "xPts"], ascending=[True, False])
    for _, row in cand_all.iterrows():
        if len(chosen) >= 15:
            break
        tid, pid, pcost = int(row.team_id), int(row.player_id), float(row.cost)
        if pid in chosen:
            continue
        if used_team.get(tid, 0) >= max_from_team:
            continue
        if cost_total + pcost > budget:
            continue
        chosen.append(pid)
        used_team[tid] = used_team.get(tid, 0) + 1
        cost_total += pcost

    if len(chosen) < 15:
        raise ValueError("Could not build a legal 15-man squad under constraints. Try increasing budget or pool size.")
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
        pool = _coerce_pool(pool)
        ids = pool.player_id.astype(int).tolist()
        cost_map = dict(zip(pool.player_id, pool.cost))
        xpts_map = dict(zip(pool.player_id, pool.xPts))
        pos_map  = dict(zip(pool.player_id, pool.position))
        team_map = dict(zip(pool.player_id, pool.team_id))

        prob = pl.LpProblem("initial_squad", pl.LpMaximize)
        y = pl.LpVariable.dicts("y", ids, lowBound=0, upBound=1, cat=pl.LpBinary)

        # Objective
        prob += pl.lpSum([y[i] * (xpts_map[i] + 0.02 * cost_map[i]) for i in ids])

        # Budget
        prob += pl.lpSum([y[i] * cost_map[i] for i in ids]) <= float(self.rules.budget)

        # Exact 15-man composition
        want = {"GK": self.rules.gk, "DEF": self.rules.df, "MID": self.rules.md, "FWD": self.rules.fw}
        for pos, need in want.items():
            prob += pl.lpSum([y[i] for i in ids if pos_map[i] == pos]) == int(need)

        # Max 3 per team
        for t in set(team_map.values()):
            prob += pl.lpSum([y[i] for i in ids if team_map[i] == t]) <= int(self.rules.max_from_team)

        prob.solve(pl.PULP_CBC_CMD(msg=False))

        if not _status_ok(prob):
            chosen = _greedy_repair_squad(pool, want, float(self.rules.budget), int(self.rules.max_from_team))
        else:
            chosen = [i for i in ids if y[i].value() == 1]

        # Final sanity
        if len(chosen) != 15:
            chosen = _greedy_repair_squad(pool, want, float(self.rules.budget), int(self.rules.max_from_team))

        sub_pool = pool[pool.player_id.isin(chosen)].copy()
        xi, bench, captain, vice = self._pick_xi_captain(sub_pool)
        return {"squad15": chosen, "xi": xi, "bench": bench, "captain": captain, "vice": vice}

    def _pick_xi_captain(self, squad15: pd.DataFrame) -> Tuple[List[int], List[int], int, int]:
        """Legal XI: exactly 11, GK=1, DEF>=3, MID>=3, FWD>=1."""
        s15 = _coerce_pool(squad15)
        ids = s15.player_id.astype(int).tolist()
        xpts = dict(zip(s15.player_id, s15.xPts))
        pos_map = dict(zip(s15.player_id, s15.position))

        # Guard: ensure at least one GK exists
        if not any(p == "GK" for p in pos_map.values()):
            raise ValueError("Your 15-man squad has no goalkeeper; cannot build a legal XI.")

        prob = pl.LpProblem("xi_and_cap", pl.LpMaximize)
        s = pl.LpVariable.dicts("s", ids, lowBound=0, upBound=1, cat=pl.LpBinary)
        c = pl.LpVariable.dicts("c", ids, lowBound=0, upBound=1, cat=pl.LpBinary)
        v = pl.LpVariable.dicts("v", ids, lowBound=0, upBound=1, cat=pl.LpBinary)

        prob += pl.lpSum([s[i] * xpts[i] for i in ids]) + pl.lpSum([c[i] * xpts[i] for i in ids])

        # Exactly 11 starters
        prob += pl.lpSum([s[i] for i in ids]) == 11
        # Formation
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "GK"]) == 1
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "DEF"]) >= 3
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "MID"]) >= 3
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "FWD"]) >= 1
        # Captain/Vice on starters
        prob += pl.lpSum([c[i] for i in ids]) == 1
        prob += pl.lpSum([v[i] for i in ids]) == 1
        for i in ids:
            prob += c[i] <= s[i]
            prob += v[i] <= s[i]

        prob.solve(pl.PULP_CBC_CMD(msg=False))

        if not _status_ok(prob):
            # Greedy fallback
            sorted_ids = sorted(ids, key=lambda i: xpts[i], reverse=True)
            xi: List[int] = []
            counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
            for i in sorted_ids:
                if len(xi) == 11:
                    break
                p = pos_map[i]
                xi.append(i)
                counts[p] = counts.get(p, 0) + 1

            # Ensure formation by swaps
            def swap_in_role(role: str, k: int):
                nonlocal xi, counts
                while counts.get(role, 0) < k:
                    # worst from an over-represented role
                    over_roles = [r for r in ("DEF","MID","FWD") if r != role and counts.get(r,0) > (1 if r=="FWD" else 3)]
                    worst_id = min([j for j in xi if pos_map[j] in over_roles], key=lambda j: xpts[j], default=None)
                    cand = next((j for j in sorted_ids if j not in xi and pos_map[j] == role), None)
                    if worst_id is None or cand is None:
                        break
                    xi.remove(worst_id); xi.append(cand)
                    counts[pos_map[worst_id]] -= 1; counts[role] = counts.get(role,0) + 1

            # 1 GK
            if counts.get("GK", 0) != 1:
                best_gk = next((i for i in sorted_ids if pos_map[i] == "GK" and i not in xi), None)
                worst_non_gk = min([j for j in xi if pos_map[j] != "GK"], key=lambda j: xpts[j], default=None)
                if best_gk and worst_non_gk:
                    xi.remove(worst_non_gk); xi.append(best_gk)

            swap_in_role("DEF", 3)
            swap_in_role("MID", 3)
            swap_in_role("FWD", 1)

            bench = [i for i in ids if i not in xi]
            captain = max(xi, key=lambda i: xpts[i])
            vice = max([i for i in xi if i != captain] or xi, key=lambda i: xpts[i])
            return xi, bench, int(captain), int(vice)

        # MILP solution path
        xi = [i for i in ids if s[i].value() == 1]
        if len(xi) != 11:
            # weird solver result; try again via greedy
            return self._pick_xi_captain(squad15)

        bench = [i for i in ids if i not in xi]
        captain = next(i for i in ids if c[i].value() == 1)
        vice = next(i for i in ids if v[i].value() == 1)
        return xi, bench, int(captain), int(vice)
