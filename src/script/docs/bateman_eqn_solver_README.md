# Bateman Equation Solver — Implementation Notes

## The Bateman Equation

For a linear chain of $n$ nuclides with decay constants $\lambda_1, \lambda_2, \ldots, \lambda_n$, the number of atoms of the $n$-th nuclide at time $t$, starting from $N_0$ atoms of the root nuclide, is:

$$N_n(t) = N_0 \cdot \underbrace{\prod_{i=1}^{n-1} \lambda_{2,i}}_{\texttt{kk}} \cdot \sum_{i=1}^{n} \left[ \underbrace{\frac{1}{\prod_{j \neq i}(\lambda_j - \lambda_i)}}_{\texttt{coeffs[i]}} \cdot \exp\!\left(-\underbrace{\lambda_i}_{\texttt{lambdas[i]}} \cdot t\right) \right]$$

where $\lambda_{2,i} = \lambda_i \cdot \frac{p_i}{100}$ is the decay constant of nuclide $i$ scaled by the branching probability $p_i$ of the transition leading to nuclide $i+1$.

Activity is then simply:

$$A_n(t) = \lambda_n \cdot N_n(t)$$

Since $A(t) = \lambda \cdot N(t)$ is a trivial single multiply, the solver only computes $N(t)$ — the caller applies $\lambda$ themselves.

---

## `BatemanState`

```python
@dataclass
class BatemanState:
    kk:      float        # Product of edge-weighted lambdas
    coeffs:  np.ndarray   # Partial fraction coefficients, one per nuclide in path
    lambdas: np.ndarray   # Decay constants, one per nuclide in path
```

For a path of length $n$, `BatemanState` stores everything needed to evaluate $N(t)$ at any timestamp without recomputation:

- `kk` $= \prod_{i=1}^{n-1} \lambda_{2,i}$ — time-independent scalar prefix
- `lambdas` $= [\lambda_1, \lambda_2, \ldots, \lambda_n]$ — appears in the exponents
- `coeffs` $= [c_1, c_2, \ldots, c_n]$ where $c_i = \frac{1}{\prod_{j \neq i}(\lambda_j - \lambda_i)}$ — partial fraction coefficients, time-independent

These are computed once and cached. Evaluation at any $t$ is then a single dot product (see `evaluate`).

---

## BFS and the Pre-cache Guarantee

The DAG is a directed acyclic graph where each node is a nuclide and each edge is a decay transition. A path from root to any node uniquely determines the Bateman coefficients for that node — the same nuclide reached via a different path has different coefficients.

`_compute_bateman_states` uses BFS to traverse every root-to-node path. The key property of BFS is that **it processes nodes level by level** — depth $d$ is fully processed before any node at depth $d+1$ is visited. This means:

> When computing the `BatemanState` for a path of length $n$, the `BatemanState` for the parent path of length $n-1$ is **guaranteed to already be in the cache**.

This allows each state to be built **incrementally** from its parent — only one new nuclide is added per step, and the parent's cached arrays are extended by one element rather than recomputed from scratch.

---

## Path Enumeration

BFS enumerates **every unique root-to-node path**, not just root-to-leaf. This matters because the same nuclide can appear at different depths via different branching paths, and each such occurrence has its own distinct `BatemanState`.

The queue carries `(current_nuclide, path_so_far)`:

```python
queue: deque[tuple[NuclideID, tuple[NuclideID, ...]]]
```

Each time a daughter is visited, a new path tuple is formed by appending the daughter to the current path:

```python
new_path = path + (daughter,)
```

A branching node with $k$ daughters spawns $k$ independent path extensions, each carrying its own path tuple forward into the queue.

---

## Incremental Coefficient Extension

When extending a path from depth $n$ to $n+1$ by adding a new nuclide with decay constant $\lambda_{new}$:

**Existing coefficients** each gain one new factor in their denominator:

$$c_i^{new} = \frac{c_i^{prev}}{\lambda_{new} - \lambda_i}$$

```python
diffs           = daughter_lambda - parent_state.lambdas   # (λ_new - λᵢ) for all i
new_coeffs[:-1] = parent_state.coeffs / diffs              # vectorized over all existing i
```

**New nuclide's own coefficient:**

$$c_{new} = \frac{1}{\prod_{i=1}^{n}(\lambda_{new} - \lambda_i)} = \frac{1}{\prod_{i=1}^{n}(-(\lambda_i - \lambda_{new}))} = \frac{1}{\prod(-\texttt{diffs})}$$

```python
new_coeffs[-1] = 1.0 / np.prod(-diffs)
```

`diffs` is already computed for the existing coefficient update, so the new coefficient costs one `np.prod` with no extra work.

**`kk` extension** — multiply by the edge-weighted lambda of the current node:

$$kk_{new} = kk_{prev} \cdot \lambda_{2, current} = kk_{prev} \cdot \lambda_{current} \cdot \frac{p}{100}$$

```python
new_kk = parent_state.kk * (current_data.decay_const * prob / 100.0)
```

**`lambdas` extension** — append the new decay constant:

```python
new_lambdas = np.append(parent_state.lambdas, daughter_lambda)
```

---

## Root Seeding

For the root nuclide ($n=1$), the Bateman equation degenerates to simple exponential decay:

$$N_1(t) = N_0 \cdot e^{-\lambda_1 t}$$

The `kk` prefix is an empty product (no edges yet), which by convention equals 1. `coeffs` and `lambdas` are both length-1 arrays:

```python
cache[(self._root,)] = BatemanState(
    kk      = 1.0,                      # Empty product
    coeffs  = np.array([1.0]),          # Single term, coefficient = 1
    lambdas = np.array([root_lambda])
)
```

This seeds the BFS correctly — the first daughter extension will pick up `parent_state.kk = 1.0` and multiply by $\lambda_{root} \cdot p/100$ to get the correct `kk` for depth 2.

---

## `_build_batch_arrays`

After BFS completes, all `BatemanState`s are packed into padded 2D arrays once at init:

- `all_kk` of shape $(P,)$ — one scalar value per path
- `all_coeffs` of shape $(P, d_{max})$ — rows are `coeffs` arrays, shorter paths zero-padded
- `all_lambdas` of shape $(P, d_{max})$ — rows are `lambdas` arrays, shorter paths padded with $1$

Zero-padding `coeffs` does not break calculations as padded terms contribute $0 \cdot e^{-\lambda t} = 0$ to the Bateman sum. The padding value for `lambdas` is arbitrary since its corresponding `coeffs` entry is 0.

### Grouping matrix

`grouping_matrix` shape $(V, P)$ where $V$ is the number of unique terminal nuclides:

$$G_{v,p} = \begin{cases} 1 & \text{if path } p \text{ terminates at nuclide } v \\ 0 & \text{otherwise} \end{cases}$$

This encodes which paths contribute to which nuclide, enabling the final summation across paths to be expressed as a single matrix multiplication rather than a Python loop.

---

## `evaluate_all`

```python
exp_terms  = np.exp(-self._all_lambdas[np.newaxis, :, :] * t[:, np.newaxis, np.newaxis])

N_paths    = N0 * self._all_kk[np.newaxis, :] * \
    (exp_terms * self._all_coeffs[np.newaxis, :, :]).sum(axis=-1)

N_nuclides = N_paths @ self._grouping_matrix.T

```

### **Step 1 — `exp_terms`**

`self._all_lambdas[np.newaxis, :, :]` is a matrix of shape $(1, P, d_{max})$. Multiplying `t[:, np.newaxis, np.newaxis]`  of shape $(T, 1, 1)$ to it broadcasts to $(T, P, d_{max})$:

$$\texttt{exp\_terms}[t, p, i] = e^{-\lambda_{p,i} \cdot t}$$

One exponential per timestamp, per path, per depth position — computed in a single C-level operation (much faster than pure Python).

Its pure Python equivalent is:
```python
for timestamp in T:
    for path in P:
        for lamda in lamdas[path]:  # All lamdas[path] have d_max vals due to padding
            exp_terms[timestamp, path, lamda] = math.exp(-lamda * timestamp)
```

### **Step 2 — `N_paths`**

`exp_terms * all_coeffs[np.newaxis, :, :]` broadcasts `all_coeffs` from $(P, d_{max})$ to $(T, P, d_{max})$, then `.sum(axis=-1)` collapses the $d_{max}$ dimension by summing over them for each `[timestamp, path]` pair:

$$\texttt{N\_paths}[t, p] = \sum_{i=1}^{d_{max}} c_{p,i} \cdot e^{-\lambda_{p,i} \cdot t} = \sum_{i=1}^{d_p} c_{p,i} \cdot e^{-\lambda_{p,i} \cdot t}$$

(padded terms vanish since $c_{p,i} = 0$ for $i > d_p$)

Scaling the result by `N0 * all_kk[np.newaxis, :]` broadcasts `all_kk` from $(P,)$ to $(T, P)$, giving $N(t)$ for every path simultaneously. The shape of the result array remains $(T, P)$.

Its pure Python equivalent is:
```python
# exp_terms * all_coeffs[np.newaxis, :, :]
for timestamp in T:
    for path in P:
        for i < d_max:
            result_arr[timestamp, path, i] = exp_terms[timestamp, path, i] * coeffs[path, i]

# .sum(axis=-1)
for timestamp in T:
    for path in P:
        summation[timestamp, path] = 0

        for result in result_arr[timestamp, path]:
            summation[timestamp, path] += result

# Scaling by N0 * all_kk[np.newaxis, :]
for timestamp in T:
    for path in P:
        calc_result[timestamp, path] = N0 * all_kk[path] * summation[timestamp, path]
```

### **Step 3 — `N_nuclides`**

The matrix multiplication of $(T, P) \cdot (P, V) \to (T, V)$ sums contributions across all paths terminating at each nuclide:

$$\texttt{N\_nuclides}[t, v] = \sum_{p=1}^{P} \texttt{N\_paths}[t, p] \cdot G_{v,p} = \sum_{\text{paths ending at } v} N(t \mid \text{path})$$

Its pure Python equivalent is:
```python
for timestamp in T:
    for v in terminal_nuclides:
        collated_result[timestamp, v] = 0

        for path in P:
            if path.end() == v:
                collated_result[timestamp, v] += calc_result[timestamp, path]
```

### Return Value

The final dict comprehension maps column index $v$ back to its `NuclideID`:

```python
return {nuclide: N_nuclides[:, v] for v, nuclide in enumerate(self._nuclides)}
```

---

## Time Complexity

Let:
- $P$ = total number of unique root-to-node paths in the DAG
- $d$ = $d_{max}$ = maximum chain depth (longest root-to-leaf path)
- $T$ = number of timestamps in the evaluation array
- $V$ = number of unique terminal nuclides
### `_compute_bateman_states` — $O(P \cdot d)$
 
Each BFS step extends one path by one nuclide. Per step:
- `np.append(lambdas)` — $O(d)$, copies the array
- `diffs = lambdas - daughter_lambda` — $O(d)$
- `new_coeffs[:-1] = coeffs / diffs` — $O(d)$
- `new_coeffs[-1] = 1 / np.prod(-diffs)` — $O(d)$

There are $P$ such steps total, giving $O(P \cdot d)$. All operations are numpy — no pure Python loops over nuclides.
 
### `_build_batch_arrays` — $O(P \cdot d)$
 
One pass over all $P$ cached states to fill `all_kk`, `all_coeffs`, `all_lambdas`, and `grouping_matrix`. Each fill is $O(d)$ per path (copying `coeffs` and `lambdas` slices). Total: $O(P \cdot d)$.
 
### `evaluate_all` — $O(T \cdot P \cdot d)$
 
Three numpy operations, each dominant in one dimension:
 
| Operation | Shape | Cost |
|---|---|---|
| `np.exp(...)` | $(T, P, d_{max})$ | $O(T \cdot P \cdot d)$ |
| `(exp_terms * coeffs).sum(axis=-1)` | $(T, P, d_{max}) \to (T, P)$ | $O(T \cdot P \cdot d)$ |
| `N_paths @ grouping_matrix.T` | $(T, P) \cdot (P, V) \to (T, V)$ | $O(T \cdot P \cdot V)$ |

$V$ and $d$ are independent — $V$ can exceed $d$ in heavily branching chains where many distinct nuclides appear as leaves. The total for `evaluate_all` is therefore $O(T \cdot P \cdot (d + V))$. Zero Python loops — all operations execute at C level.
 
### Combined — $O(T \cdot P \cdot (d + V))$
 
Precompute steps $O(P \cdot d)$ are dominated by `evaluate_all` since $T \gg 1$. The precompute is paid once at init; `evaluate_all` is the per-call cost.

### Multiple root isotopes

Each `BatemanEqnSolver` instance maintains its own private cache — Bateman states are not shared across solvers. A shared cache is not helpful because `BatemanState` coefficients depend on the full path from the root, so paths from different root isotopes are never equivalent even if they pass through the same nuclides.

The total precompute cost across $R$ root isotopes is therefore $O(R \cdot P \cdot d)$ — each solver runs its own independent BFS.

`evaluate_all` cost per call is $O(T \cdot P_{root} \cdot (d + V))$ per solver.
 
---

## Memory Consumption
 
### Per `BatemanState`
 
For a path of depth $d$, one `BatemanState` stores:
- `kk` — 1 float64 = 8 bytes
- `coeffs` — $d$ float64s = $8d$ bytes
- `lambdas` — $d$ float64s = $8d$ bytes

Total per state: $8(2d + 1) \approx 16d$ bytes. For $d = 40$ (theoretical worst case): ~640 bytes per state.
 
### Cache and batch arrays
 
| Structure | Shape | Size |
|---|---|---|
| `_cache` | $P$ states of depth $\leq d$ | $\approx 16 \cdot P \cdot d$ bytes |
| `all_kk` | $(P,)$ | $8P$ bytes |
| `all_coeffs` | $(P, d_{max})$ | $8 \cdot P \cdot d$ bytes |
| `all_lambdas` | $(P, d_{max})$ | $8 \cdot P \cdot d$ bytes |
| `grouping_matrix` | $(V, P)$ | $8 \cdot V \cdot P$ bytes |
 
Total precomputed: $\approx 48 \cdot P \cdot d$ bytes. For $P = 1000$, $d = 40$: ~1.9 MB.

### Multiple root isotopes

Each solver holds its own private cache and batch arrays. Total memory across $R$ root isotopes scales as $O(R \cdot P \cdot d)$ for cache and $O(R \cdot P \cdot d)$ for batch arrays — no sharing.

Note that the **DAG is shared** across all solvers — nuclide data (decay constants, transitions) is stored once regardless of how many solvers reference it. Only the Bateman computation state is per-solver.
 
### `evaluate_all` working memory
 
The intermediate `exp_terms` array of shape $(T, P, d_{max})$ is the peak allocation per call:
 
$$8 \cdot T \cdot P \cdot d_{max} \text{ bytes}$$
 
For $T = 1001$, $P = 1000$, $d = 40$: ~320 MB. For realistic chains ($P = 10$, $d = 10$): ~800 KB. 
 
### Path count bounds
 
The number of paths $P$ is bounded by:
 
$$P \leq \sum_{k=1}^{d} B^k = \frac{B^{d+1} - 1}{B - 1}$$
 
where $B$ is the average branching factor. Physically unreachable worst case ($B = 5$, $d = 40$): $P \leq 5^{40} \approx 10^{28}$. Realistic bounds:
 
| Chain | Typical $P$ | Batch array size | `exp_terms` per call |
|---|---|---|---|
| Ac-225 | ~10 | < 50 KB | < 1 MB |
| Th-232 (complex) | ~50-100 | < 500 KB | < 40 MB |
| Theoretical worst case actinide | ~1000 | < 2 MB | ~320 MB |