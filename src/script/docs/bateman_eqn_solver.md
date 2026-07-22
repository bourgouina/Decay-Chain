# `bateman_eqn_solver.py` — Implementation Notes

## The Bateman Equation

For a linear chain of $n$ nuclides with decay constants $\lambda_1, \lambda_2, \ldots, \lambda_n$, the number of atoms of the $n$-th nuclide at time $t$, starting from $N_0$ atoms of the root nuclide, is:

$$N_n(t) = N_0 \cdot \underbrace{\prod_{i=1}^{n-1} \lambda_{2,i}}_{\texttt{kk}} \cdot \sum_{i=1}^{n} \left[ \underbrace{\frac{1}{\prod_{j \neq i}(\lambda_j - \lambda_i)}}_{\texttt{coeffs[i]}} \cdot \exp\!\left(-\underbrace{\lambda_i}_{\texttt{lambdas[i]}} \cdot t\right) \right]$$

where $\lambda_{2,i} = \lambda_i \cdot \frac{p_i}{100}$ is the decay constant of nuclide $i$ scaled by the branching probability $p_i$ of the transition leading to nuclide $i+1$.

Activity is then simply:

$$A_n(t) = \lambda_n \cdot N_n(t)$$

---

## Two-Cache BFS Design

- **`_path_cache`** — one `BatemanState` per distinct root-to-node path, extended from the parent path's cached state rather than recomputed from the root.
- **`_nuclide_cache`** — one `BatemanCalcData` per distinct nuclide.

### BFS And The Pre-Cache Guarantee

The key property of BFS is that it processes nodes level by level — depth $d$ is fully processed before any node at depth $d+1$ is visited. This means:

> When computing the `BatemanState` for a path of length $n$, the `BatemanState` for the parent path of length $n-1$ is guaranteed to already be in the cache.

Therefore `parent_state = self._path_cache[path]` always exists by the time it is required, which is what allows each state to be built incrementally from its parent.

### Why `_nuclide_cache` Is Required, Not Just An Optimization

Under MC perturbation, `read_nuclide_data` draws a fresh sample per call. A physical nuclide can appear multiple times in one traversal (intermediate on one path, daughter on another, shared ancestor of several downstream chains). Without the cache, each appearance would independently resample, giving the same physical nuclide many different decay constants/branching ratios values within one trial. The cache forces exactly one sample per nuclide per solve.

---

## Root Seeding

`kk` for the root is an empty product (no edges traversed yet), which by convention is `1.0`. 

`coeffs`/`lambdas` are both length 1:

```python
self._path_cache[(self._root,)] = BatemanState(
    kk      = 1.0,                      # Empty product
    coeffs  = np.array([1.0]),          # Single term, coefficient = 1
    lambdas = np.array([root_lambda])
)
```

---

## Incremental `BatemanState` Calculation

When extending a path from depth $n$ to $n+1$ by adding a new nuclide with decay constant $\lambda_{new}$:

### `coeffs`

- The **existing coefficients** each gain one new factor in their denominator $\frac{1}{\lambda_{new} - \lambda_i}$. Therefore:

  $$c_i^{new} = \frac{c_i^{prev}}{\lambda_{new} - \lambda_i}$$

  ```python
  diffs           = daughter_lambda - parent_state.lambdas   # (λ_new - λᵢ) for all i
  new_coeffs[:-1] = parent_state.coeffs / diffs              # calculated over all existing i
  ```

- The **new nuclide's** own coefficient is:

  $$c_{new} = \frac{1}{\prod_{i=1}^{n}(\lambda_{new} - \lambda_i)} = \frac{1}{\prod_{i=1}^{n}(-(\lambda_i - \lambda_{new}))} = \frac{1}{\prod(-\texttt{diffs})}$$

  ```python
  new_coeffs[-1] = 1.0 / np.prod(-diffs)
  ```

### `kk`

Obtained by multiplying by the edge-weighted $\lambda_{current}$ to the `kk` value of its parent:

$$kk_{new} = kk_{parent} \cdot \lambda_{2, current} = kk_{parent} \cdot \lambda_{current} \cdot \frac{p}{100}$$

```python
new_kk = parent_state.kk * (current_data.decay_const * prob / 100.0)
```

### `lambdas`

Obtained by appending the new decay constant to its parent's decay constants list:

```python
new_lambdas = np.append(parent_state.lambdas, daughter_lambda)
```

---

## Batch Array Construction

$P$ = number of distinct cached paths, $V$ = number of distinct terminal nuclides, $d_{max}$ = depth of the longest cached path, $T$ = number of timestamps in `self._t`.

Paths of different depth are packed into matrices by right-padding, producing `all_kk` (shape `(P,)`), `all_coeffs` and `all_lambdas` (shape `(P, d_max)`):

- `coeffs` padded with `0.0` — a padded slot must contribute nothing to the sum regardless of its paired `exp_term`.
- `lambdas` padded with `1.0` — the value is irrelevant since its coefficient is `0.0`.

### Grouping Matrix

`grouping_matrix` has shape $(V, P)$:

$$G_{v,p} = \begin{cases} 1 & \text{if path } p \text{ terminates at nuclide } v \\ 0 & \text{otherwise} \end{cases}$$

This encodes which paths contribute to which nuclide, so the final summation across paths is a single matrix multiplication (`N_paths @ grouping_matrix.T`) rather than a Python loop accumulating per-path results into per-nuclide buckets.

---

## `_build_batch_arrays` — $N_n(t)$ Calculation Breakdown

```python
exp_terms  = np.exp(-all_lambdas[np.newaxis, :, :] * self._t[:, np.newaxis, np.newaxis])
N_paths    = self._N0 * all_kk[np.newaxis, :] * \
    (exp_terms * all_coeffs[np.newaxis, :, :]).sum(axis=-1)
N_nuclides = N_paths @ grouping_matrix.T
```

### Step 1 — `exp_terms`

`all_lambdas[np.newaxis, :, :]`, shape $(1, P, d_{max})$, broadcast-multiplied against `t[:, np.newaxis, np.newaxis]`, shape $(T, 1, 1)$, gives shape $(T, P, d_{max})$:

$$\texttt{exp\_terms}[t, p, i] = e^{-\lambda_{p,i} \cdot t}$$

One exponential per timestamp, per path, per depth position.

### Step 2 — `N_paths`

`exp_terms * all_coeffs[np.newaxis, :, :]` broadcasts `all_coeffs` from $(P, d_{max})$ to $(T, P, d_{max})$; `.sum(axis=-1)` collapses the $d_{max}$ axis:

$$\texttt{N\_paths}[t, p] = \sum_{i=1}^{d_{max}} c_{p,i} \cdot e^{-\lambda_{p,i} \cdot t} = \sum_{i=1}^{d_p} c_{p,i} \cdot e^{-\lambda_{p,i} \cdot t}$$

(padded terms vanish since $c_{p,i}=0$ for $i>d_p$). Scaling by `N0 * all_kk[np.newaxis, :]` broadcasts `all_kk` from $(P,)$ to $(T,P)$.

### Step 3 — `N_nuclides`

The matmul $(T,P)\cdot(P,V)\to(T,V)$ sums contributions across all paths terminating at each nuclide:

$$\texttt{N\_nuclides}[t, v] = \sum_{p=1}^{P} \texttt{N\_paths}[t, p] \cdot G_{v,p} = \sum_{\text{paths ending at } v} N(t \mid \text{path})$$

---

## Time Complexity — $O(T \cdot P \cdot V)$

Since the DAG has no cycles, a path of length $d_{\max}$ visits $d_{\max}$ distinct nuclides, all among the $V$ nuclides reachable from `root` — so $d_{\max} \le V$.

- Computing `BatemanState`s (**`_compute_bateman_states`**) — $O(P \cdot d_{\max}) \le O(P \cdot V)$:
    - Calculations inside the BFS loop — $O(d_{\max})$
    - The BFS loop runs for $P$ iterations since it enumerates every unique path.

- Calculating $N(t)$ for all nuclides — $O(T \cdot P \cdot V)$:
    - Calculating `exp_terms` and summing over `d_max` — $O(T \cdot P \cdot d_{\max}) \le O(T \cdot P \cdot V)$
    - Grouping matrix multiplication — $O(T \cdot P \cdot V)$

- Calculating $A(t)$ for all nuclides — $O(T \cdot V)$

## Space Complexity — $O(T \cdot P \cdot V)$

- **`_path_cache`** — $O(P \cdot d_{\max}) \le O(P \cdot V)$
- **`_nuclide_cache`** — $O(V)$
- **`exp_terms`** — $O(T \cdot P \cdot d_{\max}) \le O(T \cdot P \cdot V)$
- **`_N_nuclides` / `_A_nuclides`** — $O(T \cdot V)$ each