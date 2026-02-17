from typing import Dict, List, Optional, Tuple
import math

def _score_num(a: Optional[float], b: Optional[float], pct_tol: float) -> float:
    """Score numeric proximity. pct_tol is relative tolerance."""
    if a is None or b is None:
        return 0.55
    try:
        a = float(a); b = float(b)
    except Exception:
        return 0.55
    if a <= 0 or b <= 0:
        return 0.55
    diff = abs(a - b) / max(abs(a), abs(b))
    if diff <= pct_tol:
        return 1.0 - 0.6 * (diff / pct_tol)
    return max(0.0, 0.4 - (diff - pct_tol) * 0.8)

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def candidate_score(row: Dict, obs: Dict) -> float:
    """
    Higher is better. Roughly in [0, ~4].
    Obs keys:
      year, diameter_mm, thickness_mm, mass_g, hole_mm, density_gcm3, country, denomination
    """
    s = 0.0

    # Year match
    y = obs.get("year")
    y0, y1 = row.get("year_start"), row.get("year_end")
    if y and y0 and y1:
        try:
            y = int(y); y0 = int(y0); y1 = int(y1)
            s += 1.0 if (y0 <= y <= y1) else 0.0
        except Exception:
            s += 0.4
    else:
        s += 0.4

    # Physical specs
    s += _score_num(row.get("diameter_mm"), obs.get("diameter_mm"), 0.03)
    s += _score_num(row.get("thickness_mm"), obs.get("thickness_mm"), 0.10)
    s += _score_num(row.get("mass_g"), obs.get("mass_g"), 0.06)

    # Hole
    hole_obs = obs.get("hole_mm") or 0.0
    hole_row = row.get("hole_mm") or 0.0
    if (hole_obs <= 0 and hole_row <= 0):
        s += 0.3
    elif (hole_obs > 0 and hole_row > 0):
        s += _score_num(hole_row, hole_obs, 0.08)
    else:
        s += 0.0

    # Country / denomination
    if obs.get("country") and row.get("country"):
        s += 0.7 if _norm(obs["country"]) == _norm(row["country"]) else 0.0
    if obs.get("denomination") and row.get("denomination"):
        o = _norm(obs["denomination"]); r = _norm(row["denomination"])
        s += 0.5 if (o in r or r in o) else 0.0

    return float(s)

def rank_candidates(db_rows: List[Dict], obs: Dict, top_n: int = 10) -> List[Tuple[Dict, float]]:
    scored = [(r, candidate_score(r, obs)) for r in db_rows]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]
