from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

@dataclass
class SquadRules:
    budget: float = 100.0  # initial build
    bank: float = 0.0      # weekly bank override
    max_from_team: int = 3
    gk: int = 2
    df: int = 5
    md: int = 5
    fw: int = 3

@dataclass
class HitPolicy:
    max_transfers: int = 1
    free_transfers: int = 1
    hit_cost: int = 4
    allow_hits: bool = True

@dataclass
class PlanningKnobs:
    horizon: int = 1
    price_weight: float = 0.0
    risk: str = "medium"

TEAM_ID_TO_NAME: Dict[int, str] = {}
# utils.py placeholder
