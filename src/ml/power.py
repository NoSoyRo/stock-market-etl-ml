"""Poder estadístico para el IC medio semanal."""
from __future__ import annotations

import math

from scipy import stats


def ic_se(sigma_ic: float, n: int) -> float:
    """Error estándar del IC medio: σ/√n."""
    return sigma_ic / math.sqrt(n)


def required_weeks(ic: float, sigma_ic: float, alpha: float = 0.05,
                   power: float = 0.8, two_sided: bool = False) -> int:
    """Semanas para detectar un IC medio `ic`: ((z_α + z_poder)·σ/IC)², redondeado arriba."""
    z_a = stats.norm.ppf(1 - (alpha / 2 if two_sided else alpha))
    z_p = stats.norm.ppf(power)
    return int(math.ceil(((z_a + z_p) * sigma_ic / ic) ** 2))
