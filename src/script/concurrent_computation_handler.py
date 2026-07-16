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

    Attributes
    ----------
    - `atom_count_vals`:    Maps each reachable nuclide to its `N(t)` atom count array over the
                            requested timestamps
    - `activity_vals`:      Maps each reachable nuclide to its activity array (`decay_const * N(t)`)
                            over the requested timestamps
    - `trial_data`:         Maps each reachable nuclide to its data (maybe perturbed) used in Bateman 
                            equation calculations.
    """

    atom_count_vals:    Map
    activity_vals:      Map
    trial_data:         dict[NuclideID, BatemanCalcData]    # Currently always None as data storage not in scope


# ----- Constants --------------------
_ROOT_WORKER_COUNT = 5
_TRIAL_WORKER_COUNT = 50
THREAD_COUNT = 20


class ConcurrentComputationHandler:
    def __init__(self, dag: DecayChainDAG, trials: int = 1, seed: int | None = None):
        """
        Coordinates concurrent Bateman equation solves across multiple root nuclides, optionally
        with Monte Carlo (MC) perturbation.

        Parameters
        ----------
        - `dag`:    Fully built decay chain DAG, shared across all root computations

        - `trials`: Number of MC trials to run per root. `1` runs a single deterministic
                    (unperturbed) solve per root, and no `Generator` is constructed.

        - `seed`:   Test-mode determinism switch. 
                    When set, every root constructs its `SeedSequence` from the same seed, and 
                    trial `i`'s `Generator` is always spawned from child index `i` of that sequence 
                    (i.e. trial `i` draws the same perturbations regardless of root). This is
                    intended for reproducible testing, not statistically independent MC sampling.
                    `None` (default) gives each root's `SeedSequence` independent OS-entropy
                    seeding, so spawned trial generators are independent across both roots and
                    trials — appropriate for production MC runs.
        """

        # Gaurd against negative trial count
        if trials < 1:
            raise RuntimeError("Trial count needs to be at least 1.")
        
        self._dag       = dag
        self._trials    = trials
        self._seed      = seed
        self._compute_vals: dict[tuple[NuclideID, int], list[TrialResult]] = {}
    

    # ----- Private Methods --------------------
    def _compute_one_root(self, root: NuclideID, N0: int, timestamps: npt.NDArray[np.float64]):
        """
        Runs all trials for a single root nuclide and stores the results.

        Builds one `SeedSequence` for this root (or skips it, if `trials == 1`) and spawns
        `trials` independent child sequences from it up front, before any trial thread is
        submitted. Each trial thread then constructs its own `Generator` from its child
        sequence, so trials never share a mutable `Generator` object — each trial's draws
        are isolated from every other concurrently running trial for this root.

        Workflow
        --------
        - Spawn `trials` child `SeedSequence`s for this root (skipped if `trials == 1`).
        - For each trial: construct that trial's own `Generator` from its child sequence
          (or `None` for a deterministic solve), build a fresh `BatemanEqnSolver`, evaluate
          atom counts, derive activity values from the solver's own cached trial data, and
          store the result in this trial's pre-assigned slot.

        Parameters
        ----------
        - `root`:        Root nuclide to solve from
        - `N0`:          Initial atom count of `root`
        - `timestamps`:  1D array of time points (in seconds) to evaluate at
        """

        # Each trial has a pre-assigned slot in the list which depends on its trial number
        self._compute_vals[(root, N0)] = [None] * self._trials

        # If no. of trials is greater than 1, spawn one independent child SeedSequence per trial, 
        # up front, so each trial's Generator is constructed from its own child rather than a shared 
        # object
        child_seqs = None

        if self._trials > 1:
            seed_seq    = np.random.SeedSequence() if self._seed is None \
                else np.random.SeedSequence(self._seed)
            child_seqs  = seed_seq.spawn(self._trials)

        # Calculates the Bateman equation for each trial in parallel
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(self._compute_one_trial, i, 
                                       np.random.default_rng(child_seqs[i]) if self._trials > 1 else None, 
                                       root, N0, timestamps)
                       for i in range(self._trials)]

            for future in futures:
                future.result()
    

    def _compute_one_trial(self, i: int, rng: np.random.Generator | None, root: NuclideID, N0: int, 
                           timestamps: npt.NDArray[np.float64]):
        """
        Runs a single Bateman equation solve (one trial) for one root nuclide and stores
        the result in this trial's pre-assigned slot.

        Parameters
        ----------
        - `i`:           Trial index; determines this trial's slot in the results list
        - `rng`:         This trial's own `Generator` for MC perturbation, or `None` for
                         a deterministic (unperturbed) solve
        - `root`:        Root nuclide to solve from
        - `N0`:          Initial atom count of `root`
        - `timestamps`:  1D array of time points (in seconds) to evaluate at
        """

        bateman_solver  = BatemanEqnSolver(self._dag, root, rng)
        atom_count_vals = bateman_solver.evaluate_all(N0, timestamps)

        activity_vals: Map = {
            nuclide: bateman_solver.read_nuclide_data(nuclide).decay_const * values
            for nuclide, values in atom_count_vals.items()
        }
                    
        self._compute_vals[(root, N0)][i] = TrialResult(
            atom_count_vals = atom_count_vals,
            activity_vals   = activity_vals,
            trial_data      = None
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
        Dict mapping `(root nuclide, N0)` to a list of `TrialResult`, one entry per trial.
        """

        # Does Monte-Carlo simulations for each root in parallel
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(self._compute_one_root, root, N0, timestamps)
                       for root, N0 in roots]

            for future in futures:
                future.result()
        
        return self._compute_vals