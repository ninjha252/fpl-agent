from __future__ import annotations
import requests
from typing import Dict, Any, List

BASE = "https://fantasy.premierleague.com/api"

class FPL:
    @staticmethod
    def bootstrap() -> Dict[str, Any]:
        r = requests.get(f"{BASE}/bootstrap-static/")
        r.raise_for_status()
        return r.json()

    @staticmethod
    def fixtures() -> List[Dict[str, Any]]:
        r = requests.get(f"{BASE}/fixtures/")
        r.raise_for_status()
        return r.json()
# fpl_api.py placeholder
