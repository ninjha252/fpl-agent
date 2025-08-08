from __future__ import annotations
import pulp as pl
import pandas as pd
from typing import Dict, List, Tuple

from .utils import SquadRules

# --------- small utils ---------
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
    out["position"]  = out["position"].astype(str).str.upper().str.strip()
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
    """Safe repair: enforce 1 GK, DEF>=3, MID>=3, FWD>=1, total=11 without shrinking below 11."""
    xi = list(dict.fromkeys(xi))
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

    # exactly 1 GK
    while counts["GK"] > 1:
        repl = next((i for i in sorted_ids if pos_map[i] != "GK" and i not in xi), None)
        if repl is None:
            break
        gks = [j for j in xi if pos_map[j] == "GK"]
        worst_gk = min(gks, key=lambda j: xpts[j])
        xi.remove(worst_gk); xi.append(repl)
        counts = recount()

    while counts["GK"] < 1:
        add_gk = next((i for i in sorted_ids if pos_map[i] == "GK" and i not in xi), None)
        drop = min([j for j in xi if pos_map[j] != "GK"], key=lambda j: xpts[j], default=None)
        if add_gk is None or drop is None:
            break
        xi.remove(drop); xi.append(add_gk)
        counts = recount()

    def raise_min(role: str, k: int):
        nonlocal xi, counts
        tries = 0
        while counts.get(role, 0) < k and tries < 20:
            tries += 1
            cand = next((i for i in sorted_ids if pos_map[i] == role and i not in xi), None)
            if cand is None:
                break
            victims = []
            for r, rmin in (("DEF", 3), ("MID", 3), ("FWD", 1)):
                if r != role and counts.get(r, 0) > rmin:
                    victims += [j for j in xi if pos_map[j] == r]
            if not victims:
                break
            worst = min(victims, key=lambda j: xpts[j])
            xi.remove(worst); xi.append(cand)
            counts = recount()

    raise_min("DEF", 3)
    raise_min("MID", 3)
    raise_min("FWD", 1)

    while len(xi) < 11:
        add = next((i for i in sorted_ids if i not in xi and not (pos_map[i] == "GK" and counts["GK"] >= 1)), None)
        if add is None: break
        xi.append(add); counts = recount()

    while len(xi) > 11:
        victims = [j for j in xi if not (pos_map[j] == "GK" and counts["GK"] == 1)]
        worst = min(victims, key=lambda j: xpts[j])
        xi.remove(worst); counts = recount()

    return xi

class SquadOptimizer:
    def __init__(self, rules: SquadRules):
        self.rules = rules

    # ---------- building 15 ----------
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

    # ---------- legal XI ----------
    def _pick_xi_captain(self, squad15: pd.DataFrame) -> Tuple[List[int], List[int], int, int]:
        s15 = _coerce_pool(squad15)
        ids = s15.player_id.tolist()
        xpts = dict(zip(s15.player_id, s15.xPts))
        pos_map = dict(zip(s15.player_id, s15.position))

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
        for i in ids:
            prob += c[i] <= s[i]
            prob += v[i] <= s[i]

        prob.solve(pl.PULP_CBC_CMD(msg=False))

        xi = [i for i in ids if s[i].value() == 1] if _status_ok(prob) else []
        if len(xi) != 11:
            xi = []
            for i in sorted(ids, key=lambda j: xpts[j], reverse=True):
                if len(xi) < 11:
                    xi.append(i)

        xi = _repair_xi_to_formation(xi, ids, pos_map, xpts)
        bench = [i for i in ids if i not in xi]
        captain = max(xi, key=lambda i: xpts[i])
        vice = max([i for i in xi if i != captain] or xi, key=lambda i: xpts[i])
        return xi, bench, int(captain), int(vice)

    # ---------- transfer suggester ----------
    def _xi_points(self, squad_df: pd.DataFrame) -> float:
        """Approx XI expectation: sum(xPts XI) + extra xPts for captain."""
        xi, _, cap, _ = self._pick_xi_captain(squad_df)
        x = squad_df[squad_df.player_id.isin(xi)]["xPts"].sum()
        xc = float(squad_df.loc[squad_df.player_id == cap, "xPts"].iloc[0])
        return float(x + xc)  # rough captain boost

    def suggest_transfers(
        self,
        current15: pd.DataFrame,
        pool: pd.DataFrame,
        bank: float,
        max_changes: int = 2,
        free_transfers: int = 1,
        hit_cost: int = 4,
        team_cap: int = 3,
    ) -> Dict[str, object]:
        """
        Greedy, formation-aware transfer suggestions.
        Returns dict with 'suggestions' list and 'final_points'.
        Each suggestion: {out_id,in_id,out_name,in_name,delta_pts,cost_diff,net_gain}
        """
        squad = _coerce_pool(current15)
        pool  = _coerce_pool(pool)

        if len(squad) != 15:
            raise ValueError("Provide exactly 15 players from your current squad.")

        team_counts = squad["team_id"].value_counts().to_dict()
        have_ids = set(squad["player_id"].tolist())
        base_pts = self._xi_points(squad)

        suggestions = []
        changes = 0
        remaining_bank = float(bank)

        while changes < max_changes:
            best = None
            best_gain = 0.0

            for _, row_out in squad.iterrows():
                pos = row_out["position"]
                out_id = int(row_out["player_id"])
                out_team = int(row_out["team_id"])
                out_cost = float(row_out["cost"])

                candidates = pool[(pool["position"] == pos) & (~pool["player_id"].isin(have_ids))] \
                                .sort_values("xPts", ascending=False).head(80)

                for _, row_in in candidates.iterrows():
                    in_id = int(row_in["player_id"])
                    in_team = int(row_in["team_id"])
                    in_cost = float(row_in["cost"])

                    next_count_in = team_counts.get(in_team, 0) + (0 if in_team == out_team else 1)
                    if next_count_in > team_cap:
                        continue

                    if remaining_bank + out_cost - in_cost < -1e-9:
                        continue

                    tmp = squad.copy()
                    tmp.loc[tmp.player_id == out_id, ["player_id","team_id","cost","xPts","position","web_name","team_name"]] = [
                        in_id, in_team, in_cost, float(row_in["xPts"]), pos,
                        str(row_in.get("web_name", in_id)),
                        str(row_in.get("team_name", in_team)),
                    ]

                    try:
                        new_pts = self._xi_points(tmp)
                    except Exception:
                        continue

                    raw_gain = float(new_pts - base_pts)
                    hit = 0 if changes < free_transfers else hit_cost
                    net_gain = raw_gain - hit
                    if net_gain > best_gain + 1e-9:
                        best_gain = net_gain
                        best = {
                            "out_id": out_id, "in_id": in_id,
                            "out_name": row_out.get("web_name", str(out_id)),
                            "in_name": row_in.get("web_name", str(in_id)),
                            "out_team": row_out.get("team_name", out_team),
                            "in_team": row_in.get("team_name", in_team),
                            "cost_diff": float(in_cost - out_cost),
                            "raw_gain": raw_gain,
                            "net_gain": net_gain,
                            "new_pts": new_pts,
                        }

            if not best or best_gain <= 0:
                break

            changes += 1
            remaining_bank -= best["cost_diff"]
            base_pts = float(best["new_pts"])
            have_ids.remove(best["out_id"])
            have_ids.add(best["in_id"])
            team_counts[best["in_team"]] = team_counts.get(best["in_team"], 0) + (0 if best["in_team"] == best["out_team"] else 1)
            if best["in_team"] != best["out_team"]:
                team_counts[best["out_team"]] = team_counts.get(best["out_team"], 1) - 1

            squad.loc[squad.player_id == best["out_id"], ["player_id","team_id","cost","xPts","position","web_name","team_name"]] = [
                best["in_id"], best["in_team"],
                float(pool.loc[pool.player_id == best["in_id"], "cost"].iloc[0]),
                float(pool.loc[pool.player_id == best["in_id"], "xPts"].iloc[0]),
                str(pool.loc[pool.player_id == best["in_id"], "position"].iloc[0]),
                str(pool.loc[pool.player_id == best["in_id"], "web_name"].iloc[0]) if "web_name" in pool.columns else str(best["in_id"]),
                str(pool.loc[pool.player_id == best["in_id"], "team_name"].iloc[0]) if "team_name" in pool.columns else str(best["in_team"]),
            ]

            suggestions.append(best)

        return {
            "suggestions": suggestions,
            "final_points": base_pts,
            "remaining_bank": remaining_bank,
            "changes": changes,
        }
