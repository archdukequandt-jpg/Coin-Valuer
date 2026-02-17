from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

DENSITY = {
    "gold": 19.32, "silver": 10.49, "copper": 8.96, "nickel": 8.90, "zinc": 7.14,
    "brass": 8.5, "bronze": 8.8, "cupro-nickel": 8.9, "steel": 7.85, "stainless": 7.95,
    "aluminum": 2.70, "tin": 7.31, "lead": 11.34, "bi-metal": 8.6,
}

COLOR_PRIORS = {
    "gold": ["gold","yellow","brass"],
    "silver": ["silver","grey","gray","white"],
    "coppery": ["red","brown","copper","bronze"],
    "black": ["black","dark"],
}

@dataclass
class Observations:
    country: Optional[str]
    year: Optional[int]
    color: Optional[str]
    mass_g: Optional[float]
    diameter_mm: Optional[float]
    thickness_mm: Optional[float]
    hole_mm: Optional[float]

def _color_family(color: Optional[str]) -> Optional[str]:
    if not color:
        return None
    c = color.strip().lower()
    for fam, keys in COLOR_PRIORS.items():
        for k in keys:
            if k in c:
                return fam
    return None

def _gauss(x, mu, sigma):
    if sigma <= 0:
        sigma = 0.5
    return math.exp(-0.5*((x-mu)/sigma)**2)

def _likelihood_density(obs_rho, target_rho):
    if obs_rho is None or (isinstance(obs_rho, float) and (obs_rho!=obs_rho)):
        return 1.0
    sigma = max(0.5, 0.06 * target_rho)
    return _gauss(obs_rho, target_rho, sigma)

def _year_rule_bonus(country: Optional[str], year: Optional[int], metal_key: str) -> float:
    if not year:
        return 1.0
    y = int(year)
    c = (country or '').lower()
    # USA
    if c in ('usa','us','united states','united states of america','u.s.'):
        if metal_key == 'steel' and y == 1943:
            return 2.5
        if metal_key in ('zinc','copper') and y >= 1982:
            return 1.4 if metal_key=='zinc' else 1.1
        if metal_key == 'silver' and y <= 1964:
            return 1.8
        if metal_key == 'silver' and 1965 <= y <= 1970:
            return 1.3
        if metal_key in ('cupro-nickel','copper','nickel') and y >= 1965:
            return 1.2
    # UK
    if c in ('uk','united kingdom','great britain','england','britain'):
        if metal_key == 'silver' and y <= 1919:
            return 1.8
        if metal_key == 'silver' and 1920 <= y <= 1946:
            return 1.4
        if metal_key in ('cupro-nickel','steel') and y >= 1947:
            return 1.2
    # Canada
    if c in ('canada',):
        if metal_key == 'silver' and y <= 1966:
            return 1.6
        if metal_key == 'silver' and 1967 <= y <= 1968:
            return 1.3
    # Japan
    if c in ('japan',):
        if metal_key == 'aluminum' and y >= 1955:
            return 2.0
        if metal_key in ('brass','cupro-nickel') and y >= 1950:
            return 1.2
    # Euro area (generic)
    if 'euro' in c or c in ('eu','european union'):
        if metal_key in ('cupro-nickel','bi-metal','brass','steel','stainless'):
            return 1.2
    if metal_key in ('cupro-nickel','brass','bronze','steel','stainless','aluminum','zinc'):
        return 1.05
    return 1.0

def _color_prior_multiplier(color_family: Optional[str], metal_key: str) -> float:
    if color_family is None:
        return 1.0
    if color_family == 'gold' and metal_key in ('gold','brass','bronze'):
        return 1.3
    if color_family == 'silver' and metal_key in ('silver','cupro-nickel','steel','stainless','aluminum'):
        return 1.25
    if color_family == 'coppery' and metal_key in ('copper','bronze','brass'):
        return 1.3
    if color_family == 'black' and metal_key in ('steel','stainless'):
        return 1.1
    return 0.9

def classify_metal(ob: Observations, measured_density: Optional[float]) -> Dict[str, float]:
    '''Return dict metal/alloy -> normalized probability.'''
    candidates = ['gold','silver','copper','nickel','zinc','brass','bronze','cupro-nickel','steel','stainless','aluminum','lead','tin','bi-metal']
    colorfam = _color_family(ob.color)
    scores = {}
    for m in candidates:
        like = _likelihood_density(measured_density, DENSITY.get(m, 8.5))
        like *= _color_prior_multiplier(colorfam, m)
        like *= _year_rule_bonus(ob.country, ob.year, m)
        scores[m] = like
    tot = sum(scores.values()) or 1.0
    return {k: v/tot for k, v in scores.items()}
