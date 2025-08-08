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
    """Ensure required columns/dtypes and normalize positions."""
    need = {"player_id", "position", "team_id", "cost", "xPts"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Optimizer pool missing columns: {sorted(missing)}")

    out = df.copy()
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["team_id"]   = pd.to_numeric(out["team_id"],   errors="coerce").astype("Int64")
    out["cost"]      = pd.to_numeric(out["cost"],      errors="coerce").astype(float)
    out["xPts"]      = pd.to_numeric(out["xPts"],      errors="coerce").astype(float)
    out["position"]  = out["position"].astype(str).str.upper().str.strip()  # <—— normalize!
    out = out.dropna(subset=["player_id", "team_id", "cost", "xPts"]).copy()
    out["player_id"] = out["player_id"].astype(int)
    out["team_id"]   = out["team_id"].astype(int)
    return out

def _pos_ids(df: pd.DataFrame, pos: str) -> List[int]:
    return df[df.position == pos].player_id.astype(int).tolist()

def _greedy_repair_squad(pool: pd.DataFrame, want: Dict[str, int], budget: float, max_from_team: int) -> List[int]:
    pool = _coerce_pool(pool)
    chosen: List[int] = []
    used_team: Dict[int, int] = {}
    cost_total = 0.0

    for pos, need in want.items():
        cand = pool[pool.position == pos].sort_values("xPts", ascending=False)
        if cand.empty:
            raise ValueError(f"No candidates found for position {pos}")
        for _, row in cand.iterrows():
            tid, pid, pcost = int(row.team_id), int(row.player_id), float(row.cost)
            if used_team.get(tid, 0) >= max_from_team or cost_total + pcost > budget or pid in chosen:
                continue
            chosen.append(pid); used_team[tid] = used_team.get(tid, 0) + 1; cost_total += pcost
            if len([i for i in chosen if i in _pos_ids(pool, pos)]) >= need:
                break

    cand_all = pool.sort_values(["cost", "xPts"], ascending=[True, False])
    for _, row in cand_all.iterrows():
        if len(chosen) >= 15: break
        tid, pid, pcost = int(row.team_id), int(row.player_id), float(row.cost)
        if pid in chosen or used_team.get(tid, 0) >= max_from_team or cost_total + pcost > budget:
            continue
        chosen.append(pid); used_team[tid] = used_team.get(tid, 0) + 1; cost_total += pcost

    if len(chosen) < 15:
        raise ValueError("Could not build a legal 15-man squad under constraints.")
    return chosen[:15]

def _repair_xi_to_formation(xi: List[int], all_ids: List[int], pos_map: Dict[int,str], xpts: Dict[int,float]) -> List[int]:
    """Ensure 1 GK, ≥3 DEF, ≥3 MID, ≥1 FWD by swapping in best candidates."""
    counts = {"GK":0,"DEF":0,"MID":0,"FWD":0}
    for i in xi:
        counts[pos_map[i]] = counts.get(pos_map[i],0)+1

    sorted_ids = sorted(all_ids, key=lambda i: xpts[i], reverse=True)

    def swap_in(role: str, k: int):
        nonlocal xi, counts
        while counts.get(role,0) < k:
            # pick best candidate for the needed role not already in XI
            cand = next((i for i in sorted_ids if pos_map[i]==role and i not in xi), None)
            if cand is None: break
            # remove worst from an over-represented role
            over_roles = []
            if role != "DEF": over_roles += ["DEF"]  # DEF min is 3
            if role != "MID": over_roles += ["MID"]  # MID min is 3
            if role != "FWD": over_roles += ["FWD"]  # FWD min is 1
            victims = [j for j in xi if pos_map[j] in over_roles and not (pos_map[j]=="FWD" and counts["FWD"]<=1)
                       and not (pos_map[j]=="DEF" and counts["DEF"]<=3)
                       and not (pos_map[j]=="MID" and counts["MID"]<=3)]
            if not victims: break
            worst = min(victims, key=lambda j: xpts[j])
            xi.remove(worst); xi.append(cand)
            counts[pos_map[worst]] -= 1; counts[role] = counts.get(role,0)+1

    # enforce formation
    # exactly 1 GK:
    while counts["GK"] != 1:
        if counts["GK"] > 1:
            # drop worst GK for best non-GK
            gks = [j for j in xi if pos_map[j]=="GK"]
            worst_gk = min(gks, key=lambda j: xpts[j])
            repl = next((i for i in sorted_ids if pos_map[i]!="GK" and i not in xi), None)
            if repl is None: break
            xi.remove(worst_gk); xi.append(repl)
            counts["GK"] -= 1; counts[pos_map[repl]] = counts.get(pos_map[repl],0)+1
        else:
            # add best GK, remove worst non-GK
            best_gk = next((i for i in sorted_ids if pos_map[i]=="GK" and i not in xi), None)
            worst_non = min([j for j in xi if pos_map[j]!="GK"], key=lambda j: xpts[j], default=None)
            if best_gk is None or worst_non is None: break
            xi.remove(worst_non); xi.append(best_gk)
            counts[pos_map[worst_non]] -= 1; counts["GK"] += 1

    swap_in("DEF", 3)
    swap_in("MID", 3)
    swap_in("FWD", 1)
    return xi

# --------- optimizer ---------
class SquadOptimizer:
    def __init__(self, rules: SquadRules):
        self.rules = rules

    def pick_initial_squad(self, pool: pd.DataFrame) -> Dict[str, List[int]]:
        pool = _coerce_pool(pool)
        ids = pool.player_id.tolist()
        cost_map = dict(zip(pool.player_id, pool.cost))
        xpts_map = dict(zip(pool.player_id, pool.xPts))
        pos_map  = dict(zip(pool.player_id, pool.position))
        team_map = dict(zip(pool.player_id, pool.team_id))

        prob = pl.LpProblem("initial_squad", pl.LpMaximize)
        y = pl.LpVariable.dicts("y", ids, lowBound=0, upBound=1, cat=pl.LpBinary)

        prob += pl.lpSum([y[i] * (xpts_map[i] + 0.02 * cost_map[i]) for i in ids])
        prob += pl.lpSum([y[i] * cost_map[i] for i in ids]) <= float(self.rules.budget)

        want = {"GK": self.rules.gk, "DEF": self.rules.df, "MID": self.rules.md, "FWD": self.rules.fw}
        for pos, need in want.items():
            prob += pl.lpSum([y[i] for i in ids if pos_map[i] == pos]) == int(need)
        for t in set(team_map.values()):
            prob += pl.lpSum([y[i] for i in ids if team_map[i] == t]) <= int(self.rules.max_from_team)

        prob.solve(pl.PULP_CBC_CMD(msg=False))
        if not _status_ok(prob):
            chosen = _greedy_repair_squad(pool, want, float(self.rules.budget), int(self.rules.max_from_team))
        else:
            chosen = [i for i in ids if y[i].value() == 1]
        if len(chosen) != 15:
            chosen = _greedy_repair_squad(pool, want, float(self.rules.budget), int(self.rules.max_from_team))

        sub_pool = pool[pool.player_id.isin(chosen)].copy()
        xi, bench, captain, vice = self._pick_xi_captain(sub_pool)
        return {"squad15": chosen, "xi": xi, "bench": bench, "captain": captain, "vice": vice}

    def _pick_xi_captain(self, squad15: pd.DataFrame) -> Tuple[List[int], List[int], int, int]:
        s15 = _coerce_pool(squad15)
        ids = s15.player_id.tolist()
        xpts = dict(zip(s15.player_id, s15.xPts))
        pos_map = dict(zip(s15.player_id, s15.position))

        # must have at least one GK in the 15
        if not any(p == "GK" for p in pos_map.values()):
            raise ValueError("Your 15-man squad has no goalkeeper; cannot build a legal XI.")

        prob = pl.LpProblem("xi_and_cap", pl.LpMaximize)
        s = pl.LpVariable.dicts("s", ids, lowBound=0, upBound=1, cat=pl.LpBinary)
        c = pl.LpVariable.dicts("c", ids, lowBound=0, upBound=1, cat=pl.LpBinary)
        v = pl.LpVariable.dicts("v", ids, lowBound=0, upBound=1, cat=pl.LpBinary)

        prob += pl.lpSum([s[i] * xpts[i] for i in ids]) + pl.lpSum([c[i] * xpts[i] for i in ids])
        prob += pl.lpSum([s[i] for i in ids]) == 11
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "GK"]) == 1
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "DEF"]) >= 3
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "MID"]) >= 3
        prob += pl.lpSum([s[i] for i in ids if pos_map[i] == "FWD"]) >= 1
        prob += pl.lpSum([c[i] for i in ids]) == 1
        prob += pl.lpSum([v[i] for i in ids]) == 1
        for i in ids: prob += c[i] <= s[i]; prob += v[i] <= s[i]

        prob.solve(pl.PULP_CBC_CMD(msg=False))

        xi = [i for i in ids if s[i].value() == 1] if _status_ok(prob) else []
        # If solver returned nonsense, start greedy with top xPts
        if len(xi) != 11:
            xi = []
            for i in sorted(ids, key=lambda j: xpts[j], reverse=True):
                if len(xi) < 11: xi.append(i)

        # **Always** validate/repair formation after MILP
        xi = _repair_xi_to_formation(xi, ids, pos_map, xpts)

        bench = [i for i in ids if i not in xi]
        captain = max(xi, key=lambda i: xpts[i])
        vice = max([i for i in xi if i != captain] or xi, key=lambda i: xpts[i])
        return xi, bench, int(captain), int(vice)
