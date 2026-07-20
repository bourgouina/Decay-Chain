from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ----- Custom Types --------------------
NuclideID       = tuple[str, str, int]      # (symbol, meta, mass no.)
TransitionEdge  = tuple[NuclideID, float]   # (nuclide, branching ratio)


# ----- Constants --------------------
_BRANCHING_TOTAL_PCT  = 100.0
_CONVERGENCE_TOL_PCT  = 1e-9     # Correction below this threshold is treated as "already summed to 100%"


# ----- Data Classes --------------------
@dataclass
class BatemanCalcData:
    """Snapshot of a nuclide's data used for Bateman equation calculations."""

    nuclide:            NuclideID
    decay_const:        float
    decay_transitions:  list[TransitionEdge]


@dataclass
class Nuclide:
    """Stored graph node for a nuclide in the decay chain DAG."""

    nuclide:            NuclideID
    decay_const:        float
    decay_unc:          float | None
    decay_transitions:  list[DecayTransition]


@ dataclass
class DecayTransition:
    """Stored graph edge representing one decay transition from a parent nuclide."""

    nuclide:            NuclideID
    decay_type:         str
    branching_ratio:    float
    branching_unc:      float | None


# ----- Private Helpers --------------------
def _perturb_decay_const(rng: np.random.Generator, value: float, unc: float) -> float:
    """
    Returns a perturbed decay constant sampled from `N(value, unc)`, clipped to be non-negative.
    Raises `RuntimeError` if `unc` is `None`.

    Parameters
    ----------
    - `rng`:    Caller-owned `np.random.Generator` instance
    - `value`:  Central value to sample around
    - `unc`:    Standard deviation of the sample
    """

    if unc is None:
        raise RuntimeError("All uncertainty values need to be set before sampling.")

    sample = value + rng.normal(0.0, unc)
    return max(0.0, sample)


def _perturb_branching_ratio(rng: np.random.Generator, value: float, unc: float) -> float:
    """
    Returns a perturbed branching ratio sampled from `N(value, unc)`, clamped to [0, 100].
    Raises `RuntimeError` if `unc` is `None`.

    Parameters
    ----------
    - `rng`:    Caller-owned `np.random.Generator` instance
    - `value`:  Central value to sample around
    - `unc`:    Standard deviation of the sample
    """

    if unc is None:
        raise RuntimeError("All uncertainty values need to be set before sampling.")

    sample = value + rng.normal(0.0, unc)
    return min(100.0, max(0.0, sample))


def _redistribute_to_total(ratios: np.ndarray, uncs: np.ndarray, nuclide: NuclideID) -> np.ndarray:
    """
    Returns `ratios` adjusted so they sum to 100% while each stays within [0, 100], weighted by
    uncertainty. 
    Raises `ValueError` if the branching ratios of `nuclide` cannot be made to sum to 100%.

    Parameters
    ----------
    - `ratios`:   Sampled values, already individually clipped to `[0, 100]`
    - `uncs`:     Uncertainty (std dev) of each value, same order/length as `ratios`
    - `nuclide`:  Nuclide the ratios belong to, used only for the error message
    """

    n      = len(ratios)
    ratios = ratios.copy()
    active = np.ones(n, dtype=bool)

    for _ in range(n):
        correction = _BRANCHING_TOTAL_PCT - ratios.sum()

        # If correction value is less than threshold, it can be treated as 0
        # So no more redistribution required
        if abs(correction) < _CONVERGENCE_TOL_PCT:
            break

        # If every value is frozen at a boundary, then there is no room left for redistribution
        if not active.any():
            break

        weights      = uncs[active] ** 2
        total_weight = weights.sum()

        # If all active weights are zero, then the sampled branching ratios cannot sum up to 100%
        if total_weight <= 0.0:
            break

        proposed: np.ndarray    = ratios[active] + correction * (weights / total_weight)
        violates                = (proposed < 0.0) | (proposed > _BRANCHING_TOTAL_PCT)

        active_idx = np.flatnonzero(active)

        # If all of the proposed branching ratios lie within the 0-100 range, then no 
        # redistribution needed
        if not violates.any():
            ratios[active_idx] = proposed
            break

        # Clip values which are outside the inclusive 0-100 range and freeze them
        # Provisionally apply the rest of the altered values
        ratios[active_idx[violates]]  = np.clip(proposed[violates], 0.0, _BRANCHING_TOTAL_PCT)
        ratios[active_idx[~violates]] = proposed[~violates]
        active[active_idx[violates]]  = False
    
    # Check if it was possible to correct branching ratios so that they followed the constraints
    if abs(ratios.sum() - _BRANCHING_TOTAL_PCT) >= _CONVERGENCE_TOL_PCT:
        raise ValueError(f"The uncertainty values of the branching ratios of {nuclide} are "
                         "too constrained to correct the sampled values back to summing to 100%.")

    return ratios


# ----- Custom DAG Datastructure --------------------
class DecayChainDAG:
    def __init__(self):
        """DAG of nuclides and decay transitions, shared read-only across concurrent solvers."""

        self.nuclides: dict[NuclideID, Nuclide] = {}
    

    # ----- Private Methods --------------------
    def _perturbed_nuclide_data(self, nuclide: NuclideID, rng: np.random.Generator) -> BatemanCalcData:
        """
        Returns `BatemanCalcData` for `nuclide` with a freshly sampled `decay_const` and 
        `decay_transitions`.

        Returns `0` as decay constant without sampling if decay constant central value is 0.

        Parameters
        ----------
        - `nuclide`:    Nuclide identifier of the form `(symbol, meta, mass_number)`
        - `rng`:        Caller-owned `np.random.Generator` instance
        """

        node = self.nuclides[nuclide]

        decay_const = 0.0 if node.decay_const == 0.0 \
            else _perturb_decay_const(rng, node.decay_const, node.decay_unc)
        transitions = self._sample_transitions(nuclide, node.decay_transitions, rng)

        return BatemanCalcData(
            nuclide             = node.nuclide,
            decay_const         = decay_const,
            decay_transitions   = transitions
        )


    def _sample_transitions(self, nuclide: NuclideID, decay_transitions: list[DecayTransition], 
                            rng: np.random.Generator) -> list[TransitionEdge]:
        """
        Returns sampled `(daughter nuclide, branching ratio)` pairs for `decay_transitions`,
        adjusted so the branching ratios sum to 100%.

        Parameters
        ----------
        - `nuclide`:            Parent nuclide the transitions belong to
        - `decay_transitions`:  Transitions belonging to that parent nuclide
        - `rng`:                Caller-owned `np.random.Generator` instance
        """

        n = len(decay_transitions)

        # If there are no decay transitions, then theres nothing to sample
        if n == 0:
            return []

        # Generate a sample for each branching ratio
        ratios = np.fromiter(
            (_perturb_branching_ratio(rng, t.branching_ratio, t.branching_unc) 
             for t in decay_transitions),
            dtype=np.float64, count=n
        )

        uncs = np.fromiter((t.branching_unc for t in decay_transitions), 
                           dtype=np.float64, count=n)

        # Correct branching ratio sample values
        ratios = _redistribute_to_total(ratios, uncs, nuclide)

        return [(t.nuclide, ratios[i]) for i, t in enumerate(decay_transitions)]


    # ----- Public Methods --------------------
    def add_nuclide(self, nuclide: NuclideID, decay_const: float, decay_unc: float):
        """
        Adds a nuclide node to the DAG. No-op if `nuclide` already exists.

        Parameters
        ----------
        - `nuclide`:        Nuclide identifier of the form `(symbol, meta, mass_number)`
        - `decay_const`:    Decay constant lambda (in 1/s)
        - `decay_unc`:      Decay constant uncertainty (in 1/s)
        """

        # Skip addition if nuclide node already exists in DAG 
        # (physics garuntees that the values would have been the same)
        if nuclide in self.nuclides:
            return

        self.nuclides[nuclide] = Nuclide(
            nuclide             = nuclide,
            decay_const         = decay_const,
            decay_unc           = decay_unc,
            decay_transitions   = []
        )
    

    def add_transition(self, parent: NuclideID, daughter: NuclideID, decay_type: str, 
                       branching_ratio: float, branching_unc: float):
        """
        Adds a decay transition to `parent`. No-op if it already exists.

        Parameters
        ----------
        - `parent`:             Parent nuclide identifier of the form `(symbol, meta, mass_number)`
        - `daughter`:           Daughter nuclide identifier of the form `(symbol, meta, mass_number)`
        - `decay_type`:         Decay type
        - `branching_ratio`:    Branching ratio % (0-100 range)
        - `branching_unc`:      Branching ratio uncertainty
        """

        # Skip addition if decay transition already exists for nuclide 
        # (physics garuntees that the values would have been the same)
        existing_transitions = self.nuclides[parent].decay_transitions

        if any(t.nuclide == daughter for t in existing_transitions):
            return

        self.nuclides[parent].decay_transitions.append(DecayTransition(
            nuclide         = daughter,
            decay_type      = decay_type,
            branching_ratio = branching_ratio,
            branching_unc   = branching_unc
        ))


    def read_nuclide_data(self, nuclide: NuclideID, rng: np.random.Generator | None) -> BatemanCalcData:
        """
        Returns a copy of `nuclide`'s data as `BatemanCalcData`; perturbed if `rng` is given,
        unperturbed otherwise. 
        Raises `RuntimeError` if `nuclide` is not in the DAG.

        Parameters
        ----------
        - `nuclide`:  Nuclide identifier of the form `(symbol, meta, mass_number)`
        - `rng`:      Caller-owned `np.random.Generator` instance, or `None` for unperturbed data
        """

        if nuclide not in self.nuclides:
            raise RuntimeError(f"Nuclide {nuclide} is not in decay chain DAG.")

        if rng is not None:
            return self._perturbed_nuclide_data(nuclide, rng)
        

        transitions = [(t.nuclide, t.branching_ratio) 
                       for t in self.nuclides[nuclide].decay_transitions]

        return BatemanCalcData(
            nuclide             = self.nuclides[nuclide].nuclide,
            decay_const         = self.nuclides[nuclide].decay_const,
            decay_transitions   = transitions
        )


    def fill_missing_data(self):
        """Fills in missing decay constant and branching ratio uncertainties in place."""

        for nuclide in self.nuclides.values():
            # Set decay constant uncertainty if not set
            if nuclide.decay_unc is None:
                nuclide.decay_unc = 0.0
            

            # Set branching ratio uncertainty if not set
            if not nuclide.decay_transitions:
                continue

            min_uncertainty = min([t.branching_ratio for t in nuclide.decay_transitions])
            
            for transition in nuclide.decay_transitions:
                if transition.branching_unc is None:
                    transition.branching_unc = min_uncertainty
    

    def get_missing_data(self) -> tuple[set[NuclideID], set[NuclideID]]:
        """Returns `(nuclides missing decay uncertainty, nuclides missing branching ratio uncertainty)`."""

        decay_uncs  = []
        br_uncs     = []

        for nuclide, data in self.nuclides.items():
            if data.decay_unc is None:
                decay_uncs.append(nuclide)
            
            for transition in data.decay_transitions:
                if transition.branching_unc is None:
                    br_uncs.append(nuclide)
        
        return (set(decay_uncs), set(br_uncs))