import numpy as np
import numpy.typing as npt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from decay_chain_dag import DecayChainDAG, NuclideID, BatemanCalcData
from bateman_eqn_solver import BatemanEqnSolver


# ----- Custom Types --------------------
Map = dict[NuclideID, npt.NDArray[np.float64]]


# ----- Data Classes --------------------
@dataclass
class TrialResult:
    """
    Stores the results of a single Bateman equation solve (one trial) for one root nuclide.

    `atom_count_vals` and `activity_vals` are raw `(T, V)` matrices. 

    Column `v` of either matrix corresponds to `nuclides[v]`.

    Use `BatemanEqnSolver.create_nuclide_map()` to convert a matrix to a `{nuclide: array}` dict if 
    needed.

    Attributes
    ----------
    - `atom_count_vals`: `N(t)` atom count numpy matrix, shape `(T, V)`
    - `activity_vals`:   `A(t)` activity numpy matrix, shape `(T, V)`
    - `nuclides`:        List of `V` reachable nuclides, ordered to match both matrices' columns
    """

    atom_count_vals:    np.ndarray
    activity_vals:      np.ndarray
    nuclides:           list[NuclideID]


@dataclass
class RootResult:
    """
    Stores the aggregated result of all MC trials for a single root nuclide and `N0`.

    Attributes
    ----------
    - `perturbed_activity_std`: Maps each reachable nuclide to its per-timestamp standard
                                deviation of activity across trials.
    - `non_perturbed_activity`: Maps each reachable nuclide to its activity array from a single
                                deterministic solve.
    """

    perturbed_activity_std: Map
    non_perturbed_activity: Map


# ----- Constants --------------------
_ROOT_WORKER_COUNT = 5
_TRIAL_WORKER_COUNT = 50
THREAD_COUNT = 20


# ----- Helper Methods --------------------
def _calc_activity_std(results: list[TrialResult]) -> np.ndarray:
    """
    Computes standard deviation of activity per nuclide per timestamp across trials.

    Raises `RuntimeError` if every trial's `nuclides` list does not match positionally as it means 
    that the ordering of the rows is not uniform and the calculated standard deviation value will 
    be incorrect.

    Parameters
    ----------
    - `results`: List of `TrialResult`s (one per trial)

    Returns
    -------
    Numpy array of shape `(T, V)` — standard deviation of activity per timestamp, per nuclide; 
    `ddof=1` since trials are a finite sample estimating the true MC variance, not a full population.
    """

    nuclides = results[0].nuclides
    if not all(r.nuclides == nuclides for r in results):
        raise RuntimeError("Nuclide ordering diverged across trials — unsafe to stack columns directly")

    stacked = np.stack([r.activity_vals for r in results], axis=0)   # (trials, T, V)

    return stacked.std(axis=0, ddof=1)


class ConcurrentComputationHandler:
    def __init__(self, dag: DecayChainDAG, trials: int = 1, seed: int | None = None):
        """
        Coordinates concurrent Bateman equation solves across multiple root nuclides, optionally
        with Monte Carlo (MC) perturbation.

        Parameters
        ----------
        - `dag`:    Fully built decay chain DAG, shared across all root computations

        - `trials`: Number of MC trials to run per root. Must be greater than 1 as standard deviation 
                    is undefined for a single sample.

        - `seed`:   Test-mode determinism switch. 
                    When set, every root constructs its `SeedSequence` from the same seed, and 
                    trial `i`'s `Generator` is always spawned from child index `i` of that sequence 
                    (i.e. trial `i` draws the same perturbations regardless of root). This is
                    intended for reproducible testing, not statistically independent MC sampling.
                    `None` (default) gives each root's `SeedSequence` independent OS-entropy
                    seeding, so spawned trial generators are independent across both roots and
                    trials — appropriate for production MC runs.
        """

        # Guard against trial counts that can't produce a meaningful MC standard deviation
        if trials <= 1:
            raise RuntimeError("Trial count needs to be greater than 1.")
        
        self._dag       = dag
        self._trials    = trials
        self._seed      = seed
        self._compute_vals: dict[tuple[NuclideID, int], RootResult] = {}
    

    # ----- Private Methods --------------------
    def _compute_one_root(self, root: NuclideID, N0: int, timestamps: npt.NDArray[np.float64]):
        """
        Runs all trials for a single root nuclide, aggregates them, and stores the result.

        Workflow
        --------
        - Spawn `trials` child `SeedSequence`s for this root.
        - For each trial (run concurrently): construct that trial's own `Generator` from its
          child sequence, build a fresh `BatemanEqnSolver`, and store its atom-count/activity
          matrices in this trial's pre-assigned slot.
        - Once all trials complete: compute the per-nuclide, per-timestamp standard deviation of
          activity across trials, and run one additional deterministic (`rng=None`) solve to get
          a non-perturbed activity baseline. Store both as a `RootResult`.

        Parameters
        ----------
        - `root`:        Root nuclide to solve from
        - `N0`:          Initial atom count of `root`
        - `timestamps`:  1D array of time points (in seconds) to evaluate at
        """

        # Each trial has a pre-assigned slot in the list which depends on its trial number
        trial_results: list[TrialResult] = [None] * self._trials

        # Spawn one independent child SeedSequence per trial
        seed_seq    = np.random.SeedSequence() if self._seed is None \
            else np.random.SeedSequence(self._seed)
        child_seqs  = seed_seq.spawn(self._trials)

        # Calculates the Bateman equation for each trial in parallel
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(self._compute_one_trial, i, trial_results,
                                       np.random.default_rng(child_seqs[i]),
                                       root, N0, timestamps)
                       for i in range(self._trials)]

            for future in futures:
                future.result()
        
        # Calculate the standard deviation of activity across trials, per nuclide per timestamp
        activity_std = _calc_activity_std(trial_results)

        # Deterministic (unperturbed) Bateman equation solve
        non_perturbed_bateman  = BatemanEqnSolver(self._dag, root, N0, timestamps, None)
        non_perturbed_activity = non_perturbed_bateman.get_activity_matrix()
        non_perturbed_nuclides = non_perturbed_bateman.get_nuclides_list()

        self._compute_vals[(root, N0)] = RootResult(
            perturbed_activity_std  = BatemanEqnSolver.create_nuclide_map(activity_std, trial_results[0].nuclides),
            non_perturbed_activity  = BatemanEqnSolver.create_nuclide_map(non_perturbed_activity, non_perturbed_nuclides)
        )
    

    def _compute_one_trial(self, i: int, results: list[TrialResult], rng: np.random.Generator | None, 
                           root: NuclideID, N0: int, timestamps: npt.NDArray[np.float64]):
        """
        Runs a single Bateman equation solve (one trial) for one root nuclide and stores
        the result in this trial's pre-assigned slot.

        Parameters
        ----------
        - `i`:           Trial index; determines this trial's slot in `results`
        - `results`:     Shared pre-sized list this trial writes its result into, at index `i`.
                         Safe under concurrent trial threads since each writes a distinct index.
        - `rng`:         This trial's own `Generator` for MC perturbation, or `None` for
                         a deterministic (unperturbed) solve
        - `root`:        Root nuclide to solve from
        - `N0`:          Initial atom count of `root`
        - `timestamps`:  1D array of time points (in seconds) to evaluate at
        """

        bateman_solver  = BatemanEqnSolver(self._dag, root, N0, timestamps, rng)
                    
        results[i] = TrialResult(
            atom_count_vals = bateman_solver.get_atom_count_matrix(),
            activity_vals   = bateman_solver.get_activity_matrix(),
            nuclides        = bateman_solver.get_nuclides_list()
        )
    

    # ----- Public Methods --------------------
    def compute_all(self, roots: list[tuple[NuclideID, int]], timestamps: npt.NDArray[np.float64]):
        """
        Runs `_compute_one_root` concurrently across all requested roots using a thread pool.

        Parameters
        ----------
        - `roots`:      List of `(root nuclide, initial atom count)` pairs to compute
        - `timestamps`: 1D array of timestamps (in seconds) to evaluate at, shared across
                        all roots

        Returns
        -------
        Dict mapping `(root nuclide, N0)` to a `RootResult`.
        """

        # Does Monte-Carlo simulations for each root in parallel
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(self._compute_one_root, root, N0, timestamps)
                       for root, N0 in roots]

            for future in futures:
                future.result()
        
        return self._compute_vals