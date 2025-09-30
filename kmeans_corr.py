import numpy as np

def _center_unitnorm_rows(A):
    """Row-wise mean-center then scale to unit L2 norm. Constant rows -> zeros."""
    A = np.asarray(A, dtype=float)
    mu = A.mean(axis=1, keepdims=True)
    Z = A - mu
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return Z / norms

def _corr_dist_matrix(ZX, C):
    """Correlation distance matrix: D = 1 - ZX @ C.T; ZX, C are centered & unit-norm."""
    return 1.0 - ZX @ C.T

def _kmeanspp_init(ZX, K, rng):
    """
    k-means++ initialization in correlation space.
    ZX: (N,P) centered+unit-norm data.
    Returns (K,P) initial centroids (in correlation space).
    """
    N = ZX.shape[0]
    # choose first center uniformly
    first = rng.integers(0, N)
    centers_idx = [first]
    centers = [ZX[first].copy()]

    # squared correlation distance for seeding
    D2 = _corr_dist_matrix(ZX, np.vstack(centers))[:, 0] ** 2

    for _ in range(1, K):
        probs = D2 / D2.sum()
        next_idx = rng.choice(N, p=probs)
        centers_idx.append(next_idx)
        centers.append(ZX[next_idx].copy())
        # update D2 to nearest existing center
        curD2 = _corr_dist_matrix(ZX, np.vstack(centers)) ** 2
        D2 = np.min(curD2, axis=1)
    return np.vstack(centers)

# ---------- main ----------
def kmeans_correlation_matlab(X, K, num_replicates=10, max_iter=100,
                              random_state=None, init=None, return_D=False):
    """
    MATLAB-like kmeans(X, K, 'distance','correlation','replicates',R)
    with optional init (like 'Start' option).
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    N, P = X.shape

    # Precompute correlation-space data
    ZX = _center_unitnorm_rows(X)

    best_obj = np.inf
    best_idx0, best_C_corr, best_sumd, best_D = None, None, None, None

    for rep in range(num_replicates):
        # initialization
        if init is not None:
            if init.shape != (K, P):
                raise ValueError(f"init must have shape ({K}, {P})")
            C = _center_unitnorm_rows(init)
        else:
            C = _kmeanspp_init(ZX, K, rng)

        # Lloyd iterations
        prev_idx0 = None
        for _it in range(max_iter):
            D = _corr_dist_matrix(ZX, C)
            idx0 = np.argmin(D, axis=1)
            if prev_idx0 is not None and np.array_equal(idx0, prev_idx0):
                break
            prev_idx0 = idx0
            for k in range(K):
                members = (idx0 == k)
                if not np.any(members):
                    C[k] = ZX[rng.integers(0, N)]
                else:
                    m = ZX[members].mean(axis=0, keepdims=True)
                    m = m - m.mean(axis=1, keepdims=True)
                    n = np.linalg.norm(m, axis=1, keepdims=True)
                    C[k] = (m / (n if n.any() else 1.0)).ravel()

        # evaluate objective
        D = _corr_dist_matrix(ZX, C)
        rows = np.arange(N)
        obj = np.sum(D[rows, idx0])
        if obj < best_obj:
            best_obj = obj
            best_idx0, best_C_corr = idx0.copy(), C.copy()
            sumd = np.zeros(K)
            for k in range(K):
                sumd[k] = D[best_idx0 == k, k].sum()
            best_sumd = sumd
            if return_D:
                best_D = D.copy()

    # Raw centroids (mean of original X)
    C_raw = np.zeros((K, P))
    for k in range(K):
        members = (best_idx0 == k)
        if np.any(members):
            C_raw[k] = X[members].mean(axis=0)

    idx = (best_idx0 + 1).astype(int)
    if return_D:
        return idx, best_C_corr, best_sumd, C_raw, best_D
    else:
        return idx, best_C_corr, best_sumd, C_raw