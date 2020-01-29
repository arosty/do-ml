import numpy as np

def winnow(epsilon, A, l):
    """
    INPUT:
    epsilon -> float (margin)
    A       -> np.ndarray (2-dimensional, each row on feature vector)
    l       -> np.ndarray (1-dimensional, labels)
    OUTPUT:
    x       -> np.ndarray (1-dimensional, solution vector, distribution)
    """
    rho = abs(A).max()
    M = np.diag(l) * A / rho
    eta = epsilon / 2 / rho
    x = mwup(eta, M, winnow=True, A=A)
    return x


def mwup(eta, M, winnow=False, A=np.empty(0)):
    """
    INPUT:
    eta     -> float (in interval [0,0.5])
    M       -> np.ndarray (2-dimensional, costs or gains for all rounds,
               each row one round)
    OUTPUT:
    p       -> np.ndarray (1-dimensional, distribution)
    """
    T, n = M.shape
    w = np.ones(n)
    p = w / n
    for m in M:
        if winnow:
            current = np.matmul(A, p)
            if (current >= 0).all(): break
        i = random_decision(p)      # for demonstrating purposes
        w, p = update_rule(eta, w, m)
    return p


def random_decision(p):
    """
    INPUT:
    p       -> np.ndarray (1-dimensional, distribution)
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
    w       -> np.ndarray (1-dimensional, weights)
    m       -> np.ndarray (same dimension as w, costs or gains)
    OUTPUT:
    w       -> np.ndarray (1-dimensional, updated weights)
    p       -> np.ndarray (same dimension as w, new distribution)
    """
    w = (1 - eta * m) * w
    phi = w.sum()
    p = w / phi
    return w, p
