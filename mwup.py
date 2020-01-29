import numpy as np

def mwup(eta):
    pass

def update_rule(eta, w, m):
    """
    INPUT:
    eta     -> float (in interval [0,0.5])
    w       -> np.array (weights)
    m       -> np.array (same dimension as w, costs or gains)
    OUTPU:
    w       -> np.array (updated weights)
    p       -> np.array (new distribution)
    """
    w = (1 - eta * m) * w
    phi = np.sum(w)
    p = w / phi
    return w, p
