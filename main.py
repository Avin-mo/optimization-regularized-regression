import numpy as np

def make_synthetic_data(m=120, n=300, k=15, noise_std=0.01, seed=0):

    # m = number of samples (rows)
    # n = number of features (cols)
    # k = sparsity level (# of nonzeros in x_true)

    rng = np.random.default_rng(seed)

    # Data matrix A
    A = rng.normal(size=(m, n))

    # True vector x
    x_true = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    x_true[idx] = rng.normal(loc=0.0, scale=1.0, size=k)

    # Observations
    noise = noise_std * rng.normal(size=m)
    b = A @ x_true + noise  # b = Ax* + noise

    return A, b, x_true

if __name__ == "__main__":
    A, b, x_true = make_synthetic_data()
    print("A shape:", A.shape)
    print("b shape:", b.shape)
    print("x_true nonzeros:", np.count_nonzero(x_true))


def lasso_objective(A, b, x, lam):
    r = A @ x - b
    return 0.5 * (r @ r) + lam * np.linalg.norm(x, 1)

def grad_least_squares(A, b, x):
    # gradient of 0.5||Ax-b||^2
    return A.T @ (A @ x - b)

def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lipschitz_step(A):
    # L = ||A||_2^2, so step = 1/L
    smax = np.linalg.norm(A, 2)
    L = smax * smax
    return 1.0 / L

def ista(A, b, lam, max_iter=500, tol=1e-6):
    n = A.shape[1]
    x = np.zeros(n)
    step = lipschitz_step(A)

    history = {"obj": [], "nnz": []}

    for it in range(max_iter):
        # gradient step on smooth part
        g = grad_least_squares(A, b, x)
        x_next = soft_threshold(x - step * g, step * lam)

        obj = lasso_objective(A, b, x_next, lam)
        history["obj"].append(obj)
        history["nnz"].append(int(np.count_nonzero(x_next)))

        # stopping condition
        if np.linalg.norm(x_next - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_next
            break

        x = x_next

    return x, history

if __name__ == "__main__":
    A, b, x_true = make_synthetic_data(m=120, n=300, k=15, noise_std=0.01, seed=0)

    lam = 0.05  # you can tweak this
    x_hat, hist = ista(A, b, lam, max_iter=1000)

    print("Recovered nonzeros:", np.count_nonzero(x_hat))
    print("Final objective:", hist["obj"][-1])
    print("L2 error ||x_hat - x_true||:", np.linalg.norm(x_hat - x_true))
