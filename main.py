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
