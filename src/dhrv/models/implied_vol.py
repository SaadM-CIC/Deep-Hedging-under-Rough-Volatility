import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_price_from_vol(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price: float, S: float, K: float, r: float, T: float) -> float:
    """Inversion de Black-Scholes par bisection (brentq).

    Si le prix (typiquement Monte Carlo) tombe legerement sous l'intrinseque
    par bruit d'echantillonnage (frequent en deep ITM/OTM a maturite courte,
    ou l'extrinseque est minuscule), on le clippe a l'intrinseque + epsilon
    plutot que de retourner 0.0 -- 0.0 est faux et casse visuellement le smile.
    """
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    price = max(price, intrinsic + 1e-8)
    try:
        return brentq(lambda s: bs_price_from_vol(S, K, r, s, T) - price, 1e-6, 5.0)
    except ValueError:
        return np.nan