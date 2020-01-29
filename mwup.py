import numpy as np

def mwup(eta):
    pass


def random_decision(p):
    """
    INPUT:
    p       -> np.array (distribution)
    OUTPUT:
    i       -> int (randomly chosen decision)
    """
    n = p.size
    i = np.random.choice(n, 1, p=p)[0]
    return i


def update_rule(eta, w, m):
    """
    INPUT:
    eta     -> float (in interval [0,0.5])
    w       -> np.array (weights)
    m       -> np.array (same dimension as w, costs or gains)
    OUTPUT:
    w       -> np.array (updated weights)
    p       -> np.array (new distribution)
    """
    w = (1 - eta * m) * w
    phi = w.sum()
    p = w / phi
    return w, p
