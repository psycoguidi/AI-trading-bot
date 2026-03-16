"""utils/pip_utils.py – helpers per pip size e pip value."""


def pip_size(symbol: str, price: float) -> float:
    """Dimensione di 1 pip per il simbolo dato."""
    if price < 100:          # Forex (EURUSD, GBPUSD …)
        return 0.0001
    elif price < 5_000:      # Oro (XAUUSD)
        return 0.1
    else:                    # Indici (US30, NAS100)
        return 1.0


def to_pips(distance: float, symbol: str, price: float) -> float:
    """Converte una distanza in prezzi in pips."""
    ps = pip_size(symbol, price)
    return distance / ps if ps else 0.0


def pip_value_per_lot(symbol: str, price: float) -> float:
    """USD per pip per lotto standard (approssimazione)."""
    if price < 100:
        return 10.0     # Forex
    elif price < 5_000:
        return 1.0      # Oro
    else:
        return 1.0      # Indici
