from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass

from decay_chain_dag import DecayChainDAG, NuclideID


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
    def __init__(self, dag: DecayChainDAG):
        """
        Solves the Bateman equation. Expects completed DAG as input.
        """

        self._dag   = dag
        self._cache = self._compute_bateman_states()


    # ----- Private Methods --------------------
    def _compute_bateman_states(self) -> dict[tuple[NuclideID, ...], BatemanState]:
        """
        Computes and caches the values of the sub-expressions of the Bateman eqn for the terminal 
        nuclide of each path from the root.
        """

        # For each path from the root, the values of the sub-expressions of the Bateman eqn for the 
        # terminal nuclide of the path are cached for easier calculation subsequent sub-expressions
        cache: dict[tuple[NuclideID, ...], BatemanState] = {}

        # Queue also stores path taken along with next nuclide in-order to use it as a key to 
        # quickly access previously calculated sub-expressions
        queue: deque[tuple[NuclideID, tuple[NuclideID, ...]]] = deque() # Queue for BFS
        queue.append((self._dag.root, (self._dag.root,)))               # Add root to queue

        root_lambda = self._dag.nuclides[self._dag.root].decay_const

        # Cache root's trivial sub-expressions (n=1)
        cache[(self._dag.root,)] = BatemanState(
            kk      = 1.0,
            coeffs  = np.array([1.0]),
            lambdas = np.array([root_lambda])
        )

        while queue:
            current, path = queue.popleft()
            parent_state  = cache[path]         # Extract parent paths cached sub-expressions

            # For each new possible path calculate and cache their sub-expressions
            for daughter, prob in self._dag.nuclides[current].decay_transitions:
                daughter_lambda = self._dag.nuclides[daughter].decay_const
                new_path        = path + (daughter,)

                new_kk      = parent_state.kk * (self._dag.nuclides[current].decay_const * prob / 100.0)
                new_lambdas = np.append(parent_state.lambdas, daughter_lambda)

                diffs            = parent_state.lambdas - daughter_lambda
                new_coeffs       = np.empty(len(new_lambdas))
                new_coeffs[:-1]  = parent_state.coeffs / diffs
                new_coeffs[-1]   = 1.0 / np.prod(-diffs)

                cache[new_path] = BatemanState(
                    kk      = new_kk,
                    coeffs  = new_coeffs,
                    lambdas = new_lambdas
                )

                queue.append((daughter, new_path))

        return cache


    def _evaluate(self, path: tuple[NuclideID, ...], N0: float, t: np.ndarray) -> np.ndarray:
        """
        Evaluates `N(t)` for the terminal nuclide of a path.
            `N(t) = N0 * kk * dot(coeffs, exp(-lambdas * t[:, None]))`

        Parameters
        ----------
        `path`: Tuple of `NuclideID`s from root to terminal nuclide
        `N0`:   Initial atom count of root nuclide
        `t`:    1D numpy array of time points (in seconds)

        Returns
        --------
        1D numpy array of atom counts, same shape as `t`
        """

        state     = self._cache[path]
        exp_terms = np.exp(-state.lambdas * t[:, np.newaxis])

        return N0 * state.kk * (exp_terms @ state.coeffs)
 
 
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
        result: dict[NuclideID, np.ndarray] = {}
 
        for path in self._cache:
            terminal = path[-1]
            N        = self._evaluate(path, N0, t)

            if terminal in result:
                result[terminal] += N
            else:
                result[terminal]  = N
 
        return result