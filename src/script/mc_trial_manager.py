import os
import numpy as np
import numpy.typing as npt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from decay_chain_dag import DecayChainDAG, NuclideID, BatemanCalcData
from bateman_eqn_solver import BatemanEqnSolver


# ----- Custom Types --------------------
Map = dict[NuclideID, npt.NDArray[np.float64]]


# ----- Constants --------------------
_MC_WORKER_COUNT = os.cpu_count() - 2


# ----- Data Classes --------------------
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


@dataclass
class _WelfordAccumulator:
    """
    Running `(count, mean, M2)` for Welford's online variance algorithm, tracked as `(T, V)` arrays
    so every update is one vectorized step across all nuclides/timestamps at once.
    """

    count: int
    mean:  np.ndarray   # (T, V)
    M2:    np.ndarray   # (T, V), sum of squared deviations from the running mean


# ----- Helper Methods --------------------
def _welford_update(acc: _WelfordAccumulator, activity: np.ndarray) -> None:
    """
    Folds one new `(T, V)` activity sample into `acc`, in place, using Welford's online formula.

    Parameters
    ----------
    - `acc`:        Accumulator to update in place
    - `activity`:   This trial's `(T, V)` activity matrix
    """

    acc.count  += 1
    delta       = activity - acc.mean
    acc.mean   += delta / acc.count
    delta2      = activity - acc.mean
    acc.M2     += delta * delta2


def _welford_merge(a: _WelfordAccumulator, b: _WelfordAccumulator) -> _WelfordAccumulator:
    """
    Combines two independent Welford accumulators (from two different worker threads, each
    having accumulated a disjoint batch of trials) into one, via Chan et al.'s parallel variance
    combination formula. 
    
    Neither input is mutated.

    Parameters
    ----------
    - `a`, `b`: Independent accumulators over disjoint trial batches

    Returns
    -------
    A new `_WelfordAccumulator` equivalent to having accumulated both batches' trials in sequence.
    """

    count = a.count + b.count
    delta = b.mean - a.mean
    mean  = a.mean + delta * (b.count / count)
    M2    = a.M2 + b.M2 + delta**2 * (a.count * b.count / count)

    return _WelfordAccumulator(count=count, mean=mean, M2=M2)


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
    def _compute_trial_batch(self, trial_indices: npt.NDArray[np.int_],
                             child_seqs: list[np.random.SeedSequence], root: NuclideID, N0: int,
                             timestamps: npt.NDArray[np.float64],
                             expected_nuclides: list[NuclideID]) -> _WelfordAccumulator:
        """
        Runs a batch of MC trials sequentially within one worker thread, folding each trial's
        activity matrix into one local `_WelfordAccumulator` as it's produced.

        Raises `RuntimeError` if any trial's nuclide list is not the same as `expected_nuclides`.

        Parameters
        ----------
        - `trial_indices`:              Trial indices assigned to this worker, drawn from 
                                        `np.array_split` over the full trial range
        - `child_seqs`:                 Full per-trial `SeedSequence` list; only entries at 
                                        `trial_indices` are used by this call
        - `root`, `N0`, `timestamps`:   Parameters for each trial's `BatemanEqnSolver`
        - `expected_nuclides`:          Canonical nuclide ordering every trial is validated against

        Returns
        -------
        `_WelfordAccumulator` over this batch's trials.
        """

        acc = None      # Trial data accumulator

        for i in trial_indices:
            rng    = np.random.default_rng(child_seqs[i])
            solver = BatemanEqnSolver(self._dag, root, N0, timestamps, rng)

            if solver.get_nuclides_list() != expected_nuclides:
                raise RuntimeError("Nuclide ordering diverged across trials - unsafe to accumulate.")

            activity = solver.get_activity_matrix()

            # Create accumulator data structure if it does not exist
            if acc is None:
                acc = _WelfordAccumulator(
                    count = 0,
                    mean  = np.zeros_like(activity),
                    M2    = np.zeros_like(activity)
                )

            _welford_update(acc, activity)      # Update accumulator with latest trial activity values

        return acc


    # ----- Public Methods --------------------
    def compute(self, root: NuclideID, N0: int, timestamps: npt.NDArray[np.float64]) -> RootResult:
        """
        Runs all MC trials for a given `root`, `N0`, `timestamps`, aggregates them, and returns the 
        result.

        Workflow
        --------
        - Run one deterministic (unperturbed) solve to get the non-perturbed activity baseline;
          its nuclide ordering also becomes the canonical order every trial is validated against.
        - Spawn `trials` child `SeedSequence`s, then split the trial range into up to
          `_MC_WORKER_COUNT` contiguous batches.
        - For each batch (run concurrently, one worker thread per batch): sequentially solve
          every trial in the batch and fold its activity matrix into one local Welford
          accumulator.
        - Once all batches complete: merge each worker's accumulator into one global accumulator
          and derive the per-nuclide, per-timestamp standard deviation from it.

        Parameters
        ----------
        - `root`:        Root nuclide to solve from
        - `N0`:          Initial atom count of `root`
        - `timestamps`:  1D array of time points (in seconds) to evaluate at

        Returns
        -------
        `RootResult` for the given `root`, `N0`, `timestamps`.
        """

        # Deterministic (unperturbed) Bateman equation solve
        # Also establishes the canonical nuclide ordering that every trial is validated against 
        # before being accumulated
        non_perturbed_bateman  = BatemanEqnSolver(self._dag, root, N0, timestamps, None)
        non_perturbed_activity = non_perturbed_bateman.get_activity_matrix()
        non_perturbed_nuclides = non_perturbed_bateman.get_nuclides_list()

        # Spawn one independent child SeedSequence per trial
        seed_seq    = np.random.SeedSequence() if self._seed is None \
            else np.random.SeedSequence(self._seed)
        child_seqs  = seed_seq.spawn(self._trials)

        # Split the trial range into contiguous batches, one per worker. Capped at self._trials
        # so no worker ever gets an empty batch.
        num_workers = min(_MC_WORKER_COUNT, self._trials)
        batches     = np.array_split(np.arange(self._trials), num_workers)

        # Runs each batch of trials in its own thread, each returning one local accumulator
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._compute_trial_batch, batch, child_seqs,
                                       root, N0, timestamps, non_perturbed_nuclides)
                       for batch in batches]

            batch_accs = [future.result() for future in futures]

        # Merge every worker's local accumulator into one global accumulator over all trials
        total_acc = batch_accs[0]
        for acc in batch_accs[1:]:
            total_acc = _welford_merge(total_acc, acc)

        # Calculate standard deviation of activity, per timestamp, per nuclide
        activity_std = np.sqrt(total_acc.M2 / (total_acc.count - 1))

        return RootResult(
            perturbed_activity_std  = BatemanEqnSolver.create_nuclide_map(activity_std, non_perturbed_nuclides),
            non_perturbed_activity  = BatemanEqnSolver.create_nuclide_map(non_perturbed_activity, non_perturbed_nuclides)
        )