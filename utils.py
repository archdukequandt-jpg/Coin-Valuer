
import math
from typing import Optional

MM_TO_CM = 0.1               # 1 mm = 0.1 cm
MM3_TO_CM3 = 1e-3            # 1 mm^3 = 0.001 cm^3

def ring_volume_cm3(outer_d_mm: float, thickness_mm: float, inner_d_mm: float = 0.0) -> float:
    """Volume of a cylindrical coin/ring in cm^3 (supports holed coins)."""
    R = (outer_d_mm / 2.0) * MM_TO_CM
    r = (max(0.0, inner_d_mm) / 2.0) * MM_TO_CM
    h = thickness_mm * MM_TO_CM
    return math.pi * (R*R - r*r) * h

def density_gcm3(mass_g: float, *args, **kwargs) -> float:
    """Return density in g/cm³.

    Backwards-compatible overloads supported:
      1) density_gcm3(mass_g, volume_mm3)
      2) density_gcm3(mass_g, outer_d_mm, thickness_mm, inner_d_mm=0.0)
      3) density_gcm3(mass_g, volume_mm3=<mm^3>)  # via kwarg
    """
    try:
        if mass_g is None or float(mass_g) <= 0:
            return float('nan')

        # Pattern 1 or 3: explicit volume in mm^3
        volume_mm3: Optional[float] = None
        if len(args) == 1 and isinstance(args[0], (int, float)) and not kwargs:
            # Called like density_gcm3(mass_g, volume_mm3)
            volume_mm3 = float(args[0])
        elif 'volume_mm3' in kwargs:
            volume_mm3 = float(kwargs['volume_mm3'])

        if volume_mm3 is not None:
            vol_cm3 = volume_mm3 * MM3_TO_CM3
            return float(mass_g) / vol_cm3 if vol_cm3 > 0 else float('nan')

        # Pattern 2: geometry provided
        if len(args) >= 2:
            outer_d_mm = float(args[0])
            thickness_mm = float(args[1])
            inner_d_mm = float(args[2]) if len(args) >= 3 else float(kwargs.get('inner_d_mm', 0.0))
            vol_cm3 = ring_volume_cm3(outer_d_mm, thickness_mm, inner_d_mm)
            return float(mass_g) / vol_cm3 if vol_cm3 > 0 else float('nan')

        # If we get here, the signature didn't match—return NaN rather than raising
        return float('nan')
    except Exception:
        return float('nan')

def mass_from_density_g(density_gcm3: float, volume_mm3: float) -> float:
    """Mass (g) from density (g/cm^3) and volume (mm^3)."""
    try:
        return float(density_gcm3) * (float(volume_mm3) * MM3_TO_CM3)
    except Exception:
        return float('nan')

def volume_mm3(outer_d_mm: float, thickness_mm: float, inner_mm: float = 0.0, **_ignore) -> float:
    """Volume of a flat coin/ring in mm^3. Compatible with prior gem utils API."""
    R = outer_d_mm / 2.0
    r = max(0.0, inner_mm) / 2.0
    h = thickness_mm
    return math.pi * (R*R - r*r) * h

