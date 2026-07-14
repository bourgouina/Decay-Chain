from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass

from decay_chain_dag import DecayChainDAG, NuclideID, BatemanCalcData


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
    def __init__(self, dag: DecayChainDAG, root: NuclideID, rng: np.random.Generator | None):
        """
        Solves the Bateman equation for all nuclides reachable from `root` in the DAG.
 
        On initialization:
        1. Runs BFS from `root` to compute and cache `BatemanState` for every root-to-node path
        2. Packs cached states into padded batch arrays and a grouping matrix for vectorized
           evaluation
 
        Parameters
        ----------
        - `dag`:   Fully built decay chain DAG, shared across all solver instances
        - `root`:  Root nuclide to solve from
        """

        self._dag   = dag
        self._root  = root
        self._rng   = rng

        # Value caches
        self._path_cache: dict[Path, BatemanState]              = {}
        self._nuclide_cache: dict[NuclideID, BatemanCalcData]   = {}

        # Stores all paths which start at the root and terminate at a stable nuclide
        self._final_paths: list[Path]           = []
        
        # Computes and caches Bateman sub-expressions for different states
        self._compute_bateman_states()

        # Build padded batch arrays and grouping matrix from cache for vectorized evaluation
        self._all_kk, self._all_coeffs, self._all_lambdas, self._nuclides, \
            self._grouping_matrix = self._build_batch_arrays()


    # ----- Private Methods --------------------
    def _compute_bateman_states(self):
        """
        Performs BFS from `root` over the completed DAG. Computes and caches `BatemanState` for 
        every root-to-node path incrementally — each state is extended from its parent path's 
        cached state by one nuclide, avoiding full recomputation.
 
        BFS guarantees that when extending a path to depth n, the parent path state at depth n-1
        is already cached.
        """

        # Queue also stores path taken along with next nuclide in order to use it as a key to 
        # quickly access previously calculated sub-expressions
        queue: deque[tuple[NuclideID, tuple[NuclideID, ...]]] = deque()
        queue.append((self._root, (self._root,)))

        self._nuclide_cache[self._root] = self._dag.read_nuclide_data(self._root, self._rng)
        root_lambda = self._nuclide_cache[self._root].decay_const

        # Cache root's trivial sub-expressions (n=1)
        self._path_cache[(self._root,)] = BatemanState(
            kk      = 1.0,
            coeffs  = np.array([1.0]),
            lambdas = np.array([root_lambda])
        )

        while queue:
            current, path = queue.popleft()

            # If nuclide data already in cache, use it, else fetch data from DAG and cache it
            if current in self._nuclide_cache:
                current_data = self._nuclide_cache[current]
            else:
                current_data = self._dag.read_nuclide_data(current, self._rng)
                self._nuclide_cache[current] = current_data
            
            parent_state = self._path_cache[path]   # Extract parent paths cached sub-expressions

            # If current nuclide does not have any decay transitions from it, then it is stable
            # Log it and move on to next nuclide in queue
            if not current_data.decay_transitions:
                self._final_paths.append(path)
                continue

            # For each new possible path calculate and cache their sub-expressions
            for daughter, prob in current_data.decay_transitions:
                new_path = path + (daughter,)

                # If nuclide data is already in cache, use it, else fetch data from DAG and cache it
                if daughter in self._nuclide_cache:
                    daughter_data = self._nuclide_cache[daughter]
                else:
                    daughter_data = self._dag.read_nuclide_data(daughter, self._rng)
                    self._nuclide_cache[daughter] = daughter_data
                
                daughter_lambda = daughter_data.decay_const

                new_kk      = parent_state.kk * \
                    (current_data.decay_const * prob / 100.0)
                new_lambdas = np.append(parent_state.lambdas, daughter_lambda)

                diffs            = daughter_lambda - parent_state.lambdas
                new_coeffs       = np.empty(len(new_lambdas))
                new_coeffs[:-1]  = parent_state.coeffs / diffs
                new_coeffs[-1]   = 1.0 / np.prod(-diffs)

                self._path_cache[new_path] = BatemanState(
                    kk      = new_kk,
                    coeffs  = new_coeffs,
                    lambdas = new_lambdas
                )

                queue.append((daughter, new_path))


    def _build_batch_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[NuclideID], np.ndarray]:
        """
        Packs cached `BatemanState`s for this root into padded 2D arrays and a grouping matrix,
        precomputed once at init.
 
        Shorter paths are padded to `d_max`:
        - `coeffs`  padded with 0.0 — padded terms contribute nothing to the Bateman sum
        - `lambdas` padded with 1.0 — arbitrary non-zero, irrelevant since corresponding 
          `coeffs` are 0
 
        `grouping_matrix`:
        - `grouping_matrix[v, p]` = 1 if path `p` terminates at nuclide `v`, else 0
        - Allows for grouping of `N(t)` results by terminal nuclide through a single matrix 
          multiplication operation
 
        Returns
        -------
        - `all_kk`:          shape (P,)        — kk scalar per path
        - `all_coeffs`:      shape (P, d_max)  — padded coefficients per path
        - `all_lambdas`:     shape (P, d_max)  — padded decay constants per path
        - `nuclides`:        list of V unique terminal nuclides, ordered by first appearance
        - `grouping_matrix`: shape (V, P)      — binary path-to-nuclide assignment matrix
        """

        paths    = list(self._path_cache.keys())
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
            state = self._path_cache[path]
            d     = len(state.lambdas)

            all_kk[i]                           = state.kk
            all_coeffs[i, :d]                   = state.coeffs
            all_lambdas[i, :d]                  = state.lambdas
            grouping_matrix[seen[path[-1]], i]  = 1.0

        return all_kk, all_coeffs, all_lambdas, nuclides, grouping_matrix


    # ----- Public Methods --------------------
    def evaluate_all(self, N0: float, t: np.ndarray) -> dict[NuclideID, np.ndarray]:
        """
        Returns `N(t)` for every nuclide reachable from `root`, summed across all paths that
        terminate at that nuclide.
 
        Evaluation steps:
        1. `exp_terms (T, P, d_max)` — one exponential per time point, path, and depth position
        2. `N_paths (T, P)`          — Bateman sum contracted over d_max, scaled by kk and N0
        3. `N_nuclides (T, V)`       — contributions grouped by terminal nuclide via matrix multiplication
 
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
    

    def get_final_paths(self) -> list[Path]:
        """Returns the list of paths which start from the root and end at a stable nuclide."""

        return self._final_paths