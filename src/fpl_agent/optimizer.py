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

def _repair_xi_to_formation(xi: List[int], all_ids: List[int],
                            pos_map: Dict[int, str], xpts: Dict[int, float]) -> List[int]:
    """
    Ensure XI is legal:
      - exactly 1 GK
      - DEF >= 3, MID >= 3, FWD >= 1
      - total = 11
    Never shrink below 11; only swap when a replacement exists.
    """
    xi = list(dict.fromkeys(xi))  # dedupe
    # If somehow <11, pad with best remaining
    if len(xi) < 11:
        for i in sorted(all_ids, key=lambda j: xpts[j], reverse=True):
            if i not in xi:
                xi.append(i)
                if len(xi) == 11:
                    break

    def recount():
        return {
            "GK": sum(pos_map[i] == "GK" for i in xi),
            "DEF": sum(pos_map[i] == "DEF" for i in xi),
            "MID": sum(pos_map[i] == "MID" for i in xi),
            "FWD": sum(pos_map[i] == "FWD" for i in xi),
        }

    counts = recount()
    sorted_ids = sorted(all_ids, key=lambda i: xpts[i], reverse=True)

    # --- enforce exactly 1 GK ---
    while counts["GK"] > 1:
        # candidate non-GK replacement available?
        repl = next((i for i in sorted_ids if pos_map[i] != "GK" and i not in xi), None)
        if repl is None:
            break  # can't fix without shrinking XI; abort
        # remove worst GK
        gks = [j for j in xi if pos_map[j] == "GK"]
        worst_gk = min(gks, key=lambda j: xpts[j])
        xi.remove(worst_gk); xi.append(repl)
        counts = recount()

    while counts["GK"] < 1:
        # need one GK: find best GK not in XI AND a non-GK we can drop
        add_gk = next((i for i in sorted_ids if pos_map[i] == "GK" and i not in xi), None)
        drop = min([j for j in xi if pos_map[j] != "GK"], key=lambda j: xpts[j], default=None)
        if add_gk is None or drop is None:
            break
        xi.remove(drop); xi.append(add_gk)
        counts = recount()

    # --- helper to raise a role to minimum by swapping the worst over-represented role ---
    def raise_min(role: str, k: int):
        nonlocal xi, counts
        tries = 0
        while counts.get(role, 0) < k and tries < 20:
            tries += 1
            cand = next((i for i in sorted_ids if pos_map[i] == role and i not in xi), None)
            if cand is None:
                break
            # pick cheapest (lowest xPts) victim from roles that are above their mins
            victims = []
            for r, rmin in (("DEF", 3), ("MID", 3), ("FWD", 1)):
                if r != role and counts.get(r, 0) > rmin:
                    victims += [j for j in xi if pos_map[j] == r]
            if not victims:
                break
            worst = min(victims, key=lambda j: xpts[j])
            xi.remove(worst); xi.append(cand)
            counts = recount()

    # enforce mins
    raise_min("DEF", 3)
    raise_min("MID", 3)
    raise_min("FWD", 1)

    # If length drifted (shouldn’t), pad back to 11 with best remaining that doesn’t break GK=1
    while len(xi) < 11:
        add = next((i for i in sorted_ids if i not in xi and not (pos_map[i] == "GK" and counts["GK"] >= 1)), None)
        if add is None:
            break
        xi.append(add)
        counts = recount()

    # If >11 somehow (shouldn’t), drop the lowest xPts non-essential role
    while len(xi) > 11:
        victims = [j for j in xi if not (pos_map[j] == "GK" and counts["GK"] == 1)]
        worst = min(victims, key=lambda j: xpts[j])
        xi.remove(worst)
        counts = recount()

    return xi


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
