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
    Stores the aggregated result of all MC trials for a single root nuclide.

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
_MC_WORKER_COUNT = 20


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


class MCTrialManager:
    def __init__(self, dag: DecayChainDAG, trials: int, seed: int | None = None):
        """
        Coordinates concurrent Bateman equation solves across MC trials for a single root
        nuclide, initial atom count, and set of timestamps.

        Parameters
        ----------
        - `dag`:    Fully built decay chain DAG

        - `trials`: Number of MC trials to run. Must be greater than 1 since standard deviation is 
                    undefined for a single sample.

        - `seed`:   Test-mode determinism switch.
                    When set, trial `i`'s `Generator` is always spawned from child index `i`
                    of a `SeedSequence` built from this seed. This is intended for reproducible
                    testing, not statistically independent MC sampling. 
                    `None` (default) gives independent OS-entropy seeding, so spawned trial 
                    generators are independent across trials — appropriate for production MC runs.
        """

        # Guard against trial counts that can't produce a meaningful MC standard deviation
        if trials <= 1:
            raise RuntimeError("Trial count needs to be greater than 1.")

        self._dag       = dag
        self._trials    = trials
        self._seed      = seed


    # ----- Private Methods --------------------
    def _compute_one_trial(self, i: int, results: list[TrialResult], rng: np.random.Generator,
                           root: NuclideID, N0: int, timestamps: npt.NDArray[np.float64]):
        """
        Runs a single perturbed Bateman equation solve (one trial) and stores the result in this 
        trial's pre-assigned slot.

        Parameters
        ----------
        - `i`:           Trial index; determines this trial's slot in `results`
        - `results`:     Shared pre-sized list this trial writes its result into, at index `i`
        - `rng`:         This trial's own `Generator` for MC perturbation
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
    def compute(self, root: NuclideID, N0: int, timestamps: npt.NDArray[np.float64]) -> RootResult:
        """
        Runs all MC trials for a given `root`, `N0`, `timestamps`, aggregates them, and returns the 
        result.

        Workflow
        --------
        - Spawn `trials` child `SeedSequence`s.
        - For each trial (run concurrently): construct that trial's own `Generator` from its
          child sequence, build a fresh `BatemanEqnSolver`, and store its atom-count/activity
          matrices in this trial's pre-assigned slot.
        - Once all trials complete: compute the per-nuclide, per-timestamp standard deviation of
          activity across trials, and run one additional deterministic solve to get a non-perturbed 
          activity baseline.

        Parameters
        ----------
        - `root`:        Root nuclide to solve from
        - `N0`:          Initial atom count of `root`
        - `timestamps`:  1D array of time points (in seconds) to evaluate at

        Returns
        -------
        `RootResult` for the given `root`, `N0`, `timestamps`.
        """

        # Each trial has a pre-assigned slot in the list which depends on its trial number
        trial_results: list[TrialResult] = [None] * self._trials

        # Spawn one independent child SeedSequence per trial
        seed_seq    = np.random.SeedSequence() if self._seed is None \
            else np.random.SeedSequence(self._seed)
        child_seqs  = seed_seq.spawn(self._trials)

        # Calculates the Bateman equation for each trial in parallel
        with ThreadPoolExecutor(max_workers=_MC_WORKER_COUNT) as executor:
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

        return RootResult(
            perturbed_activity_std  = BatemanEqnSolver.create_nuclide_map(activity_std, trial_results[0].nuclides),
            non_perturbed_activity  = BatemanEqnSolver.create_nuclide_map(non_perturbed_activity, non_perturbed_nuclides)
        )