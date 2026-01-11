import numpy as np
import matplotlib.pyplot as plt


# generate synthetic data
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


# lasso objective function
def lasso_objective(A, b, x, lam):
    r = A @ x - b
    return 0.5 * (r @ r) + lam * np.linalg.norm(x, 1)

# gradient of least squares
def grad_least_squares(A, b, x):
    # gradient of 0.5||Ax-b||^2
    return A.T @ (A @ x - b)

# soft thresholding
def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


# lipschitz step
def lipschitz_step(A):
    # L = ||A||_2^2, so step = 1/L
    smax = np.linalg.norm(A, 2)
    L = smax * smax
    return 1.0 / L

# iterative soft thresholding algorithm
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

# main function
if __name__ == "__main__":
    A, b, x_true = make_synthetic_data(m=120, n=300, k=15, noise_std=0.01, seed=0)

    lam = 0.05
    x_hat, hist = ista(A, b, lam, max_iter=1000)

    print("Recovered nonzeros:", np.count_nonzero(x_hat))
    print("Final objective:", hist["obj"][-1])
    print("L2 error ||x_hat - x_true||:", np.linalg.norm(x_hat - x_true))

    # ---- Plot 1: Objective value ----
    plt.figure()
    plt.plot(hist["obj"])
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title(f"ISTA convergence (λ = {lam}) - Objective vs. Iterations")
    plt.grid(True)
    plt.show()


    # ---- Plot 2: Nonzero count ----
    plt.figure()
    plt.plot(hist["nnz"])
    plt.xlabel("Iteration")
    plt.ylabel("Number of nonzeros")
    plt.title(f"Sparsity over iterations (λ = {lam})")
    plt.grid(True)
    plt.show()

    # ---- Plot 3: Support comparison (x_true vs x_hat) ----
    plt.figure()
    plt.stem(x_true, linefmt='C0-', markerfmt='C0o', basefmt=" ", label='x_true')
    plt.stem(x_hat,  linefmt='C1-', markerfmt='C1x', basefmt=" ", label='x_hat')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"True vs recovered coefficients (λ = {lam})")
    plt.legend()
    plt.grid(True)
    plt.show()


    # ---- Plot 4: lambda sweep ----
    lams = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    final_nnz = []
    final_err = []

    for lam_i in lams:
        x_hat_i, hist_i = ista(A, b, lam_i, max_iter=2000)
        final_nnz.append(np.count_nonzero(x_hat_i))
        final_err.append(np.linalg.norm(x_hat_i - x_true))

    plt.figure()
    plt.plot(lams, final_nnz, marker='o')
    plt.xscale("log")
    plt.xlabel("λ")
    plt.ylabel("Final number of nonzeros")
    plt.title("Sparsity vs λ")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(lams, final_err, marker='o')
    plt.xscale("log")
    plt.xlabel("λ")
    plt.ylabel("||x_hat - x_true||_2")
    plt.title("Recovery error vs λ")
    plt.grid(True)
    plt.show()
