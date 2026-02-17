from typing import Dict, Optional
import requests

TROY_OUNCE_TO_GRAM = 31.1034768

# NOTE:
# - We intentionally avoid yfinance (disabled / incompatible with some Python versions).
# - We try a lightweight public endpoint (metals.live). If unavailable (offline, blocked),
#   we fall back to reasonable defaults so the app remains usable.

_DEFAULTS_PER_OZ_USD = {
    "gold": 2000.0,
    "silver": 25.0,
    "platinum": 900.0,
    "palladium": 950.0,
}

_METALS_LIVE_ENDPOINT = "https://api.metals.live/v1/spot/{metal}"


def _metals_live_spot(metal: str, timeout: float = 6.0) -> Optional[float]:
    """Fetch spot price (USD per troy oz) from metals.live.

    Returns None on any failure (network, parsing, unexpected shape).
    """
    try:
        url = _METALS_LIVE_ENDPOINT.format(metal=metal.lower())
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        js = r.json()
        # Expected shape: [[<timestamp_ms>, <price>]] (sometimes multiple rows)
        if not isinstance(js, list) or not js:
            return None
        last = js[-1]
        if not (isinstance(last, list) or isinstance(last, tuple)) or len(last) < 2:
            return None
        price = float(last[1])
        if price <= 0:
            return None
        return price
    except Exception:
        return None


def _spot_or_default(metal: str) -> float:
    val = _metals_live_spot(metal)
    if val is None:
        val = float(_DEFAULTS_PER_OZ_USD.get(metal.lower(), 0.0))
    return float(val)


def spot_prices() -> Dict[str, Dict[str, float]]:
    """Return a mapping of metal -> {per_oz, per_g}.

    Values are USD per troy ounce and USD per gram.
    """
    out: Dict[str, Dict[str, float]] = {}
    for metal in ["gold", "silver", "platinum", "palladium"]:
        per_oz = _spot_or_default(metal)
        out[metal] = {"per_oz": per_oz, "per_g": per_oz / TROY_OUNCE_TO_GRAM}
    return out
