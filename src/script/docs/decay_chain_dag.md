# `decay_chain_dag.py` — Implementation Notes

## Design Assumptions

This design depends on the assumption that real-world radioactive decay data will be used to build the graph, as it guarantees some real-world physics invariants which are required for its correctness.

Real-world data ensures 2 things:
- The graph remains acyclic
- The decay constants sampled are almost never zero

### Why The Graph is Acyclic for Real-World Data

- Spontaneous decay always releases energy and moves a nuclide to a lower-energy state.
- A cycle implies that some nuclide decays into an ancestor of itself (i.e. it somehow manages to spontaneously gain energy which it had released). This is not possible without an external energy source.

### Why Decay Constants Sampled Are Almost Never Zero

- Reported uncertainties on decay constants are generally magnitudes smaller than their central values [1].
- A sampled decay constant being 0 would imply that its uncertainty range was as big as the central value, if not more. This case is extremely rare for real-world reported uncertainties for decay constants.

### Why These Assumptions Are Important For Correctness

- The acyclic assumption allows the graph to skip the computation required to re-verify its acyclic nature every time a new nuclide (along with its data) is added, and also allows traversal of the graph without having to keep track of visited nodes.
- The assumption of the sampled decay constants almost always being non-zero helps reduce the probability of a division-by-zero errors during Bateman equation calculations to practically 0%.

---

## The Sampling Model

### Decay Constant

```python
def _perturb_decay_const(rng, value, unc) -> float:
    ...
    sample = rng.normal(value, unc)
    return max(0.0, sample)
```

$$\tilde\lambda = \max\!\big(0,\ \lambda + \varepsilon\big), \qquad \varepsilon \sim \mathcal N(0,\sigma_\lambda^2)$$

- The Normal draw is floored at zero as decay constants for real-world nuclides are always non-negative.
- This is a shortcut, not a perfect fix. A "proper" version would redraw until the sample lands within range. This is not done in this implementation as it would result in the run-time depending on luck instead of being bounded.
- In theory, flooring means nuclides with a poorly-constrained decay constants could see samples pile up at exactly $\tilde\lambda = 0$. In practice this almost never happens (see [Assumptions](#why-decay-constants-sampled-are-almost-never-0) above).

### Branching Ratio (Per Transition)

```python
def _perturb_branching_ratio(rng, value, unc) -> float:
    ...
    sample = rng.normal(value, unc)
    return np.clip(sample, 0.0, _BRANCHING_TOTAL_PCT)
```

$$\tilde b_i = \mathrm{clip}\big(b_i + \varepsilon_i,\ 0,\ 100\big), \qquad \varepsilon_i \sim \mathcal N(0,\sigma_i^2), \quad i = 1,\dots,n$$

The value is clipped if it goes out of the $[0, 100]$ range as it is a sample of a percentage.

#### Why Branching Ratio Perturbation Requires a Correction Step

- `_perturb_branching_ratio` samples each transition's branching ratio independently.
- In the majority of cases this breaks a physical invariant that a nuclide's branching ratios must sum to exactly 100% — as independent Normal sampling has no way to account for the other branches.

---

## Branching Ratio Correction

### Why Corrections Are Weighted By Uncertainty Value

#### Intuition

- Each branch is sampled from a range of possible values: a narrow range for a well-measured branch (small $\sigma_i$), a wide range for a poorly-measured one (large $\sigma_i$).
- When correcting the sampled values so that they to add back to 100, it is natural that the branching ratio sample with a wider uncertainty range is adjusted more than the branching ratio with a narrower one.

#### Formal Notation

- The raw sample $\tilde b_i$ can be treated as the true (but unknown) valid value $x_i$ plus a random offset drawn from that branch's own uncertainty range: $\tilde b_i = x_i + \eta_i$, where $\eta_i \sim \mathcal N(0,\sigma_i^2)$, independent for each branch.
- Given that model, "the most likely correct value of $x$" is a well-defined statistical question. The answer is a known result [2]: find the $x$ that minimizes the total squared offset, with each branch's offset divided by its own $\sigma_i^2$, subject to $x$ actually being valid.

$$\min_x \sum_i \frac{(x_i-\tilde b_i)^2}{\sigma_i^2} \quad\text{s.t.}\quad \sum_i x_i = 100,\ \ 0\le x_i\le 100$$

### Equality Constraint (Closed Form)

Let $w_i = 1/\sigma_i^2$. To handle the constraint $\sum_i x_i=100$ alone (ignoring the constraint $x_i\in[0,100]$), form the Lagrangian by subtracting $\mu$ times the constraint (rewritten as "equals zero") from the objective:

$$\mathcal{L} = \sum_i w_i(x_i - \tilde b_i)^2 \;-\; \mu\Big(\sum_i x_i - 100\Big)$$

Using the Lagrange multiplier method (set the derivative to zero to find the minimum) on $\mathcal{L}$ with respect to each $x_i$ gives, for multiplier $\mu$:

$$2w_i(x_i-\tilde b_i) - \mu = 0 \ \Longrightarrow\ x_i = \tilde b_i + \frac{\mu}{2}\sigma_i^2$$

Substituting this back into the constraint $\sum_i x_i = 100$:

$$\sum_i \tilde b_i + \frac{\mu}{2}\sum_i \sigma_i^2 = 100$$

and solving for $\mu$, with $C = 100-\sum_i \tilde b_i$ (the code's `correction` variable):

$$\frac{\mu}{2} = \frac{C}{\sum_j \sigma_j^2} \quad\Longrightarrow\quad x_i = \tilde b_i + C\cdot\frac{\sigma_i^2}{\sum_j\sigma_j^2}$$

Which is exactly:

```python
weights      = uncs[active] ** 2
total_weight = weights.sum()
proposed     = ratios[active] + correction * (weights / total_weight)
```

Therefore, a branch with a larger uncertainty range absorbs a larger share of $C$ (since the cheapest way to satisfy the sum-to-100 constraint is to push the correction onto whichever branches the data constrains least).

### $x_i\in[0,100]$ Constraint

The closed form above ignores the constraint of $x_i\in[0,100]$. The full function:

```python
def _redistribute_to_total(ratios: np.ndarray, uncs: np.ndarray, nuclide: NuclideID) -> np.ndarray:
    n      = len(ratios)
    ratios = ratios.copy()
    active = np.ones(n, dtype=bool)

    for _ in range(n):
        correction = _BRANCHING_TOTAL_PCT - ratios.sum()

        if abs(correction) < _CONVERGENCE_TOL_PCT:
            break
        if not active.any():
            break

        weights      = uncs[active] ** 2
        total_weight = weights.sum()

        if total_weight <= 0.0:
            break

        proposed = ratios[active] + correction * (weights / total_weight)
        violates = (proposed < 0.0) | (proposed > _BRANCHING_TOTAL_PCT)

        active_idx = np.flatnonzero(active)

        if not violates.any():
            ratios[active_idx] = proposed
            break

        ratios[active_idx[violates]]  = np.clip(proposed[violates], 0.0, _BRANCHING_TOTAL_PCT)
        ratios[active_idx[~violates]] = proposed[~violates]
        active[active_idx[violates]]  = False

    if abs(ratios.sum() - _BRANCHING_TOTAL_PCT) >= _CONVERGENCE_TOL_PCT:
        raise ValueError(f"The uncertainty values of the branching ratios of {nuclide} are "
                         "too constrained to correct the sampled values back to summing to 100%.")

    return ratios
```

#### What It Does

1. Recompute the closed form over the currently active branches.
2. Clamp and permanently freeze any branch whose proposal violates a bound.
3. Repeat over the shrinking active set until the residual correction is within a threshold convergence value (`_CONVERGENCE_TOL_PCT`).

**This is a known problem:** minimize a weighted sum of squared differences under one sum constraint and per-branch upper/lower bounds. It is called the *"continuous quadratic knapsack problem"*, and this clamp-and-freeze approach is the standard way to solve it exactly [3]-[5].

Freezing (rather than clamping and leaving a value active) is what guarantees convergence:
- A frozen value is excluded from all future weight sums.
- It cannot re-absorb further correction beyond what it already took at its boundary.
- The remaining correction is always redistributed only among branches that still have room to move.

### Why The Objective Is Strictly Convex

The objective is $f(x) = \sum_i w_i(x_i - \tilde b_i)^2$ with $w_i = 1/\sigma_i^2 > 0$. Since each term only involves its own $x_i$ (no cross terms between different branches), the Hessian is diagonal:

$$\frac{\partial^2 f}{\partial x_i^2} = 2w_i, \qquad \frac{\partial^2 f}{\partial x_i\, \partial x_j} = 0 \ \ (i \neq j)$$

Every diagonal entry $2w_i$ is strictly positive whenever $\sigma_i > 0$, so the Hessian is positive-definite everywhere. A positive-definite Hessian is exactly what "strictly convex" means: the objective curves upward in every direction, at every point, so there is exactly one lowest point, never a tie.

### Convergence To The Exact Global Optimum

- Reference [3]-[5] proves that this clamp-and-freeze approach, for this exact kind of problem, always finishes in at most $n$ passes and lands on the single best answer, not just a valid one.
- That guarantee holds here because the objective is strictly convex (shown directly above, so there is exactly one lowest point, not several tied for best), and every constraint is a straight line or flat boundary (also directly checkable, shown above).
- Under those conditions, the standard optimality check from [6] (find the point where the derivative of the objective, combined with the constraints, is exactly balanced) is both necessary and sufficient for a solution to be the true best one.

### Edge Cases

| Case | Code behavior | Reason |
|---|---|---|
| Every active $\sigma_i = 0$, correction still outstanding | `total_weight <= 0` breaks out of the loop, then the post-loop check raises `ValueError` | Every branch still free to move is reported as having zero uncertainty, so none of them can legitimately absorb the remaining correction. It is a data problem, not an algorithmic one |
| One branch's $\sigma_i = 0$, others nonzero | Its weight share is exactly 0 | Matches the infinitely confident ($w_i\to\infty$), should not move convention |
| Active set exhausted, correction remains | `not active.any()` breaks out of the loop, then the post-loop check raises `ValueError` | Underlying data is infeasible (central values too far from summing to 100%, uncertainties too tight to bridge the gap). It is a data problem, not an algorithmic one |
| Iteration cap ($n$ passes) reached without converging | Loop ends naturally, then the post-loop check raises `ValueError` | Covers the theoretical case where every branch gets frozen one at a time, right up to the last allowed pass, without the sum ever landing within tolerance. Same underlying cause as the row above: infeasible input data |

---

## Time Complexity

$k$ = out-degree of the nuclide / no. of decay transitions of nuclide

**In practice, $k$ is small:** real nuclides rarely have more than a couple of decay branches, empirically $k \le 3$ for essentially all measured nuclides [1].

### `_redistribute_to_total` — $O(1)$

The worst-case runtime is not $O(k)$ as in the worse-case, the outer-loop runs for $k$ iterations and the calculations inside the loop always require $O(k)$ time.

| Case | Iterations | Total cost | Condition |
|---|---|---|---|
| Typical | $1$ | $O(k)$ | Initial sample already sums close to 100% (within the error margin), or the first correction lands every branch in-bounds. |
| Worst | $k$ | $O(k^2)$ | Each pass freezes exactly one branch. Requires a pathological uncertainty value configurations which are not expected in real measured data. |

But since $k\le 3$ in most cases, $O(k)=O(1)$

### `read_nuclide_data` — $O(1)$

```python
def read_nuclide_data(self, nuclide, rng) -> BatemanCalcData:
    node = self.nuclides.get(nuclide)
    if node is None:
        raise RuntimeError(...)
    if rng is None:
        return BatemanCalcData(node.nuclide, node.decay_const,
                                list(node.decay_transitions))
    return self._perturbed_nuclide_data(node, rng)
```

| Case | Cost | Notes |
|---|---|---|
| `rng=None` | $O(k)$ | Copies the nuclide's $k$ transitions into a new list. No per-transition computation |
| `rng` provided | $O(k)$ average, $O(k^2)$ worst-case | Delegates to `_sample_transitions` which in-turn calls `_redistribute_to_total` |

But since $k\le 3$ in most cases, $O(k)=O(1)$

### Initialization — $O(N)$

$N$ = number of nuclides, $E$ = total transitions across the graph.

Covers the one-time costs incurred while building and preparing the DAG, before any solving begins.

#### Building The DAG

- Each nuclide needs to be inserted one by one — $O(N)$ total.
- For each nuclide, each decay transition needs to be inserted one by one. This adds up to $E$ transitions in total so the total runtime for it is $O(E)$.
- But since $k$ is capped at 3 in most cases, $E\le 3N$. Therefore, $O(E)=O(N)$.

#### `fill_missing_data`

- Each nuclide is visited once — $O(N)$ total.
- For each nuclide, finding the minimum branching ratio and filling in missing uncertainties both walk that nuclide's own transitions once. Summed across all nuclides this is $O(E)$.
- Same as above, $E\le 3N$ in practice since $k\le 3$, so $O(E)=O(N)$.

#### `get_missing_data`

- Each nuclide is visited once, and its transitions are scanned once, for the same $O(N + E) = O(N)$ reasoning as above.
- Building the two returned sets from the collected lists is an additional $O(N)$ pass, which doesn't change the overall order.

`fill_missing_data` and `get_missing_data` are used exclusively — only one of the two is ever called in a given run. Their costs don't stack; the one-time DAG-lifecycle overhead is $O(N)$ from whichever of the two runs, not $O(N)$ from each.

---

## Space Complexity

### DAG Storage — $O(N + E)$ (Fixed at Construction)

$N$ = number of nuclides, $E$ = total transitions across the graph.

#### Breakdown
- One dict entry per nuclide — $O(N)$ total.
- One `DecayTransition` per edge, distributed across each parent's `decay_transitions` list. Summing every out-degree gives exactly $E$ — $O(E)$ total.

### `read_nuclide_data` — $O(1)$

- Every call to `read_nuclide_data`, perturbed or not, allocates and returns a fresh `BatemanCalcData` and retains nothing beyond the call.
- This ensures that the memory consumed by the DAG does not scale with the no. of calls to `read_nuclide_data`.

---

## References

[1] International Atomic Energy Agency, Nuclear Data Section, "Live Chart of Nuclides," IAEA Nuclear Data Services. [Online]. Available: https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html

[2] A. C. Aitken, "On least squares and linear combinations of observations," *Proc. R. Soc. Edinburgh*, vol. 55, pp. 42–48, 1935.

[3] M. Held, P. Wolfe, and H. P. Crowder, "Validation of subgradient optimization," *Math. Program.*, vol. 6, pp. 62–88, 1974.

[4] P. Brucker, "An O(n) algorithm for quadratic knapsack problems," *Oper. Res. Lett.*, vol. 3, pp. 163–166, 1984.

[5] P. M. Pardalos and N. Kovoor, "An algorithm for a singly constrained class of quadratic programs subject to upper and lower bounds," *Math. Program.*, vol. 46, pp. 321–328, 1990.

[6] S. Boyd and L. Vandenberghe, *Convex Optimization*. Cambridge, U.K.: Cambridge Univ. Press, 2004, ch. 5.