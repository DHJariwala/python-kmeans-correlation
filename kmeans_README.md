
# k-means with Correlation Distance

A lightweight, NumPy-only implementation of **k-means clustering** using the **correlation distance** (`1 - correlation`) to closely mimic MATLAB's:

```
kmeans(X, K, 'Distance', 'correlation', 'Replicates', R)
```

This repository provides a single function, `kmeans_correlation_matlab`, plus a few small helpers. It is designed for reproducibility, clarity, and drop‑in use inside scientific Python projects where the correlation metric is preferred over Euclidean distance.

---

## Why correlation distance?
Correlation distance is defined here as:

> **D(x, y) = 1 − corr(x, y) = 1 − ⟨z(x), z(y)⟩**,  
> where **z(v) = (v − mean(v)) / ||v − mean(v)||₂** (row wise mean center then unit norm).

Using this distance makes clusters invariant to **linear offsets and uniform scaling** of features within a row/sample—handy for time series, gene expression, neuroimaging, and other high‑dimensional signals where shape matters more than absolute level.

---

## Features
- **MATLAB like behavior** (replicates, correlation distance, k‑means++ start)
- **Deterministic seeding** via `random_state` (NumPy `Generator`)
- **k‑means++** initialization in correlation space
- **Empty cluster handling** by reseeding from data
- **Fast & light dependency**: only NumPy

---

## Installation
This is a single file utility. Drop the function into your project or import it directly.

```bash
pip install numpy  # if you don't already have it
```

---

## Usage

```python
idx, C_corr, sumd, C_raw = kmeans_correlation_matlab(
    X, K, num_replicates=10, max_iter=100, random_state=None, init=None, return_D=False
)
```

**Parameters**

- **X** *(ndarray, shape (N, P))*: data matrix, rows are samples.
- **K** *(int)*: number of clusters.
- **num_replicates** *(int, default 10)*: number of random restarts (like MATLAB’s `Replicates`).
- **max_iter** *(int, default 100)*: Lloyd iterations per replicate.
- **random_state** *(int | np.random.Generator | None)*: seeding for reproducibility.
- **init** *(ndarray or None)*: optional initial **centroids in raw space** `(K, P)`. If provided, they are centered+unit‑normalized internally to correlation space.
- **return_D** *(bool, default False)*: if `True`, also return the final full distance matrix `D`.

**Returns**

- **idx** *(ndarray, shape (N,), 1‑based)*: cluster assignments (1..K), MATLAB style.
- **C_corr** *(ndarray, shape (K, P))*: centroids **in correlation space** (already centered & unit‑normed).
- **sumd** *(ndarray, shape (K,))*: within‑cluster correlation‑distance sums (`sum of 1 − corr` for each cluster).
- **C_raw** *(ndarray, shape (K, P))*: raw‑space centroids (plain means of original `X` rows per cluster).
- **D** *(ndarray, shape (N, K), optional)*: full correlation distance matrix for the best replicate (only when `return_D=True`).

> **Note:** Objective minimized is `sum_i D[i, idx[i]]` with `D = 1 − ZX @ C.T`, where `ZX` and `C` are **row centered, unit norm** versions of `X` and centroids.

---

## Quickstart

```python
import numpy as np

# Fake data: three clusters with shape-based similarity
rng = np.random.default_rng(42)
N, P, K = 300, 50, 3

base = np.vstack([
    np.sin(np.linspace(0, 4*np.pi, P)),
    np.cos(np.linspace(0, 4*np.pi, P)),
    np.sign(np.sin(np.linspace(0, 2*np.pi, P)))
])

X = np.vstack([
    base[0] + 0.2*rng.standard_normal((100, P)),
    base[1] + 0.2*rng.standard_normal((100, P)),
    base[2] + 0.2*rng.standard_normal((100, P)),
])

idx, C_corr, sumd, C_raw = kmeans_correlation_matlab(
    X, K=K, num_replicates=5, random_state=0
)

print("Assignments (first 10):", idx[:10])
print("Within-cluster sums of distances:", sumd)
```

---

## Using custom initialization
If you have a meaningful starting guess for centroids (in **raw** space), supply it via `init`:

```python
init = X[np.random.default_rng(0).choice(len(X), size=K, replace=False)]
idx, C_corr, sumd, C_raw = kmeans_correlation_matlab(X, K, init=init, random_state=0)
```

> Internally, `init` is transformed to correlation space with the same row centering & unit norm procedure used for the data.

---

## Relationship to MATLAB `kmeans`
This implementation aims to feel familiar if you use MATLAB’s `kmeans` with `'Distance','correlation'`:

- **Replicates**: controlled by `num_replicates`.
- **Start**: uses k‑means++ in correlation space by default; can pass explicit `init` like MATLAB’s `'Start'`.
- **Labels**: returned as **1‑based** integers to match MATLAB’s conventions.
- **Objective**: sum of correlation distances to a sample’s assigned centroid.
- **Centroids**: both correlation space (normalized) and raw space (plain means) are exposed for convenience.

Small differences may arise from random number generators, tie breaking, or empty cluster handling, but behavior should be comparable in practice.

---

## Implementation notes

- **Centering & scaling**: The helper `_center_unitnorm_rows(A)` mean centers each row of `A` and scales to unit L2 norm. Constant rows become all zeros to avoid division by zero.
- **Distance**: `_corr_dist_matrix(ZX, C)` returns `1 − ZX @ C.T`. Because `ZX` and `C` are already centered and unit norm by row, this equals `1 − correlation`.
- **Initialization**: `_kmeanspp_init(ZX, K, rng)` performs k‑means++ in correlation space using squared correlation distance for seeding probabilities.
- **Empty clusters**: If a centroid loses all members, it is reseeded from a random data row (in correlation space).
- **Stopping**: Lloyd iterations stop early if assignments do not change.

---

## Tips & gotchas

- **Reproducibility**: Set `random_state` to a fixed integer (or a pre‑built `np.random.Generator`) to make runs deterministic.
- **Scale/offset invariance**: Since we center & unit norm each row, absolute scales/offsets within a row don’t affect clustering only the **shape** of patterns matters.
- **Constant rows**: A row with zero variance is mapped to zeros during normalization. Such rows will have a correlation of `0` to any other (distance `1`). Consider removing or perturbing them if this is undesirable.
- **Performance**: For large `(N, P)`, the dominant cost is the matrix multiply to compute distances (`O(N·K·P)`). Use fewer replicates or a smaller `max_iter` if needed.

---

## Minimal example (unit test style)

```python
def test_shapes():
    X = np.arange(12).reshape(4, 3).astype(float)
    idx, C_corr, sumd, C_raw = kmeans_correlation_matlab(X, K=2, num_replicates=2, random_state=0)
    assert idx.shape == (4,)
    assert C_corr.shape == (2, 3)
    assert C_raw.shape == (2, 3)
    assert sumd.shape == (2,)
```

---

## License
MIT

---
