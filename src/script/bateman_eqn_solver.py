from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass

from decay_chain_dag import DecayChainDAG, NuclideID


# ----- Custom Types --------------------
Path = tuple[NuclideID, ...]             # Ordered list of NuclideIDs defining decay path


# ----- Data Classes --------------------
@dataclass
class BatemanState:
    """
    Stores relevant sub-expression values of the Bateman eqn
    """

    kk:      float        # Product of edge-weighted lambdas along path
    coeffs:  np.ndarray   # Partial fraction coefficients, one per nuclide in path
    lambdas: np.ndarray   # Decay constants, one per nuclide in path


# ----- Bateman Solver --------------------
class BatemanEqnSolver:
    def __init__(self, dag: DecayChainDAG, root: NuclideID, cache: dict[Path, BatemanState]):
        """
        Solves the Bateman equation. Expects completed DAG as input.
        Caches coefficients of different Bateman states and creates batch arrays for fast 
        calculations on initialization.
        """

        self._dag   = dag
        self._root  = root
        self._cache = cache
        
        # Computes and caches Bateman sub-expressions for different states
        self._compute_bateman_states()

        # Build padded batch arrays and grouping matrix from cache for vectorized evaluation
        self._all_kk, self._all_coeffs, self._all_lambdas, self._nuclides, \
            self._grouping_matrix = self._build_batch_arrays()


    # ----- Private Methods --------------------
    def _compute_bateman_states(self):
        """
        Computes and caches the values of the sub-expressions of the Bateman eqn for the terminal 
        nuclide of each path from the root.
        """

        # Queue also stores path taken along with next nuclide in order to use it as a key to 
        # quickly access previously calculated sub-expressions
        queue: deque[tuple[NuclideID, tuple[NuclideID, ...]]] = deque()
        queue.append((self._root, (self._root,)))

        root_lambda = self._dag.read_nuclide_data(self._root).decay_const

        # Cache root's trivial sub-expressions (n=1) if it does not already exist
        if not (self._root,) in self._cache:
            self._cache[(self._root,)] = BatemanState(
                kk      = 1.0,
                coeffs  = np.array([1.0]),
                lambdas = np.array([root_lambda])
            )

        while queue:
            current, path = queue.popleft()
            current_data = self._dag.read_nuclide_data(current)
            parent_state  = self._cache[path]   # Extract parent paths cached sub-expressions

            # For each new possible path calculate and cache their sub-expressions
            for daughter, prob in current_data.decay_transitions:
                new_path = path + (daughter,)

                # Skip calculation if values already exist in cache
                if new_path in self._cache:
                    continue

                daughter_lambda = self._dag.read_nuclide_data(daughter).decay_const

                new_kk      = parent_state.kk * \
                    (current_data.decay_const * prob / 100.0)
                new_lambdas = np.append(parent_state.lambdas, daughter_lambda)

                diffs            = parent_state.lambdas - daughter_lambda
                new_coeffs       = np.empty(len(new_lambdas))
                new_coeffs[:-1]  = parent_state.coeffs / diffs
                new_coeffs[-1]   = 1.0 / np.prod(-diffs)

                self._cache[new_path] = BatemanState(
                    kk      = new_kk,
                    coeffs  = new_coeffs,
                    lambdas = new_lambdas
                )

                queue.append((daughter, new_path))


    def _build_batch_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[NuclideID], np.ndarray]:
        """
        Packs cached `BatemanStates` into padded 2D arrays and builds a grouping matrix, all 
        precomputed once at init.

        Shorter paths are padded to `d_max`:
        - `coeffs`  padded with 0.0 — padded terms contribute nothing to the dot product
        - `lambdas` padded with 1.0 — arbitrary non-zero, irrelevant since corresponding `coeffs` 
          are 0

        grouping_matrix shape (V, P):
        - `grouping_matrix[v, p]` = 1 if path `p` terminates at nuclide `v`, else 0
        - Allows grouping via a single matrix multiplication: `N_paths @ grouping_matrix`.
        """

        paths    = [path for path in self._cache.keys() if path[0] == self._root]
        P        = len(paths)
        d_max    = max(len(p) for p in paths)

        # Stores unique terminal nuclides in order of first appearance with iteration through 
        # paths
        seen = {}

        for path in paths:
            t = path[-1]

            if t not in seen:
                seen[t] = len(seen)
        
        nuclides:list[NuclideID]    = list(seen.keys())
        V                           = len(nuclides)

        # Create batch arrays
        all_kk           = np.empty(P)
        all_coeffs       = np.zeros((P, d_max))
        all_lambdas      = np.ones((P, d_max))
        grouping_matrix  = np.zeros((V, P))

        for i, path in enumerate(paths):
            state = self._cache[path]
            d     = len(state.lambdas)

            all_kk[i]                           = state.kk
            all_coeffs[i, :d]                   = state.coeffs
            all_lambdas[i, :d]                  = state.lambdas
            grouping_matrix[seen[path[-1]], i]  = 1.0

        return all_kk, all_coeffs, all_lambdas, nuclides, grouping_matrix


    # ----- Public Methods --------------------
    def evaluate_all(self, N0: float, t: np.ndarray) -> dict[NuclideID, np.ndarray]:
        """
        Returns `N(t)` for every nuclide in the chain, summed across all paths that terminate at 
        that nuclide.
 
        Parameters
        ----------
        `N0`: Initial atom count of root nuclide
        `t`:  1D numpy array of time points (in seconds)

        Returns
        -------
        dict mapping `NuclideID` -> 1D atom count array of shape `(len(t),)`
        """

        # Calculate exp terms
        exp_terms = np.exp(-self._all_lambdas[np.newaxis, :, :] * t[:, np.newaxis, np.newaxis])

        # Sum over d_max
        N_paths = N0 * self._all_kk[np.newaxis, :] * \
            (exp_terms * self._all_coeffs[np.newaxis, :, :]).sum(axis=-1)

        # Group by terminal nuclide via matrix multiplication
        N_nuclides = N_paths @ self._grouping_matrix.T

        return {nuclide: N_nuclides[:, v] for v, nuclide in enumerate(self._nuclides)}