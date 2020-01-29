import numpy as np
import sys

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


def winnow(epsilon, A, l):
    """
    INPUT:
    epsilon -> float (margin)
    A       -> np.ndarray (2-dimensional, each row on feature vector)
    l       -> np.ndarray (1-dimensional, labels, all values 1 or -1)
    OUTPUT:
    x       -> np.ndarray (1-dimensional, solution vector, distribution)
    """
    rho = abs(A).max()
    M = np.matmul(np.diag(l), A) / rho
    eta = epsilon / 2 / rho
    x = mwup(eta, M, winnow=True, A=A)
    return x


def read_data():
    input_data = sys.stdin
    for line in input_data:
        if line[0] == '#':
            values = line.split()[1:4]
            n, k = [np.int(i) for i in values[:2]]
            epsilon = np.float64(values[2])
            A = np.empty([k,n], dtype=np.float64)
            j = 0
        else:
            values = line.split()
            A[j] = [np.float64(i) for i in values]
            j += 1
    return A, epsilon


# example:
epsilon = .1
A = np.array([[1, 2, 3], [3, 4, 5], [1, 0, 1], [2, 2, 9]])
l = np.array([-1, -1, 1, -1])

# x = winnow(epsilon, A, l)
# print("SOLUTION:")
# print(x)

