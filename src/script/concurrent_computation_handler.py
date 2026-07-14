import numpy as np
import numpy.typing as npt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from decay_chain_dag import DecayChainDAG, NuclideID
from bateman_eqn_solver import BatemanEqnSolver


# ----- Custom Types --------------------
Map = dict[NuclideID, npt.NDArray[np.float64]]


# ----- Data Classes --------------------
@dataclass
class ComputeData:
    """
    """

    atom_count_vals:    Map
    activity_vals:      Map


# ----- Constants --------------------
THREAD_COUNT = 20


class ConcurrentComputationHandler:
    def __init__(self, dag: DecayChainDAG, trials: int = 1, seed: int | None = None):
        """
        """

        # Gaurd against negative trial count
        if trials < 1:
            raise RuntimeError("Trial count needs to be at least 1.")
        
        self._dag       = dag
        self._trials    = trials
        self._seed      = seed
        self._compute_vals: dict[tuple[NuclideID, int], list[ComputeData]] = {}
    

    def _compute_one(self, root: NuclideID, N0: int, timestamps: npt.NDArray[np.float64]):
        """
        """

        rng = None
        self._compute_vals[(root, N0)] = []

        # Set random value generator if no. of trials is greater than 1
        if self._trials > 1:
            rng = np.random.default_rng() if self._seed is None else np.random.default_rng(seed=self._seed)

        for _ in range(self._trials):
            bateman_solver  = BatemanEqnSolver(self._dag, root, rng)
            atom_count_vals = bateman_solver.evaluate_all(N0, timestamps)

            activity_vals: Map = {
                nuclide: bateman_solver.read_nuclide_data(nuclide).decay_const * values
                for nuclide, values in atom_count_vals.items()
            }
                       
            self._compute_vals[(root, N0)].append(ComputeData(
                atom_count_vals = atom_count_vals,
                activity_vals   = activity_vals
            ))
    

    def compute_all(self, roots: list[tuple[NuclideID, int]], timestamps: npt.NDArray[np.float64]):
        """
        """

        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(self._compute_one, root, N0, timestamps)
                       for root, N0 in roots]

            for future in futures:
                future.result()
        
        return self._compute_vals
