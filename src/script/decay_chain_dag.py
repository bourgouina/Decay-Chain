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
    """
    Stores relevant data for Bateman equation calculation.
 
    Attributes
    ----------
    - `nuclide`:           Nuclide identifier of the torm `(symbol, meta, mass_number)`
    - `decay_const`:       Decay constant of nuclide (in 1/s)
    - `decay_transitions`: List of tuples of the form `(daughter nuclide, branching ratio)`
    """

    nuclide:            NuclideID
    decay_const:        float
    decay_transitions:  list[TransitionEdge]


@dataclass
class Nuclide:
    """
    Stores relevant nuclide data.

    Attributes
    ----------
    - `nuclide`:            Nuclide identifier of the form `(symbol, meta, mass_number)`
    - `decay_const`:        Decay constant value (in 1/s)
    - `decay_unc`:          Decay constant uncertainty (in 1/s)
    - `decay_transitions`:  List of `DecayTransition` objects containing transition data for nuclide
    """

    nuclide:            NuclideID
    decay_const:        float
    decay_unc:          float | None
    decay_transitions:  list[DecayTransition]


@ dataclass
class DecayTransition:
    """
    Stores relevant decay transition data.

    Attributes
    ----------
    - `nuclide`:            Daughter nuclide identifier of the form `(symbol, meta, mass_number)`
    - `decay_type`:         Decay type (alpha, beta+, ...)
    - `branching_ratio`:    Branching ratio in % (0-100 range)
    - `branching_unc`:      Branching ratio uncertainty
    """

    nuclide:            NuclideID
    decay_type:         str
    branching_ratio:    float
    branching_unc:      float | None


# ----- Private Helpers --------------------
def _perturb_decay_const(rng: np.random.Generator, value: float, unc: float) -> float:
    """
    Returns a perturbed decay constant value from within its uncertainty range.

    Sampled using Normal distribution.
    If sample is less than 0 it is clipped to 0.

    Parameters
    ----------
    - `rng`:    Caller-owned `np.random.Generator` instance
    - `value`:  Central value to sample around
    - `unc`:    Standard deviation of the sample

    Returns
    -------
    Sampled value.
    """

    if unc is None:
        raise RuntimeError("All uncertainty values need to be set before sampling.")

    sample = value + rng.normal(0.0, unc)
    return max(0.0, sample)


def _perturb_branching_ratio(rng: np.random.Generator, value: float, unc: float) -> float:
    """
    Returns a perturbed branching ratio value from within its uncertainty range.

    Sampled using Normal distribution.
    Sample value is clamped between 0-100 range (inclusive).

    Parameters
    ----------
    - `rng`:    Caller-owned `np.random.Generator` instance
    - `value`:  Central value to sample around
    - `unc`:    Standard deviation of the sample

    Returns
    -------
    Sampled value.
    """

    if unc is None:
        raise RuntimeError("All uncertainty values need to be set before sampling.")

    sample = value + rng.normal(0.0, unc)
    return min(100.0, max(0.0, sample))


def _redistribute_to_total(ratios: np.ndarray, uncs: np.ndarray, nuclide: NuclideID) -> np.ndarray:
    """
    Adjusts perturbed branching ratio values so that they add up to 100% while each of them still 
    lying in the 0-100 range.

    Adjustments are weighted using the uncertainty variance of each branching ratio.

    Raises `ValueError` if there is no way for sampled branching ratios to add up to 100%.

    Workflow
    --------
    - Compute the correction needed for the active (not yet frozen) values so that the sum of all 
      the values (including frozen) sum to 100%.
    - If all computed values lie in the 0-100 range (inclusive) or if all values are frozen, then 
      all values are valid and redistribution is complete.
    - If some computer values are not in the inclusive 0-100 range, clamp them within that range and 
      freeze them (i.e. they are not altered in future iterations).
    - Repeat till all values are valid.

    Note
    ----
    The redistribution terminates withing `n` iterations where `n` is the no. of branching ratios 
    as each pass freezes at least one value.

    Parameters
    ----------
    - `ratios`: Sampled values, already individually clipped to `[0, 100]`
    - `uncs`:   Uncertainty (std dev) of each value, same order/length as `ratios`

    Returns
    -------
    Numpy array of adjusted branching ratios.
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
            raise ValueError(f"The uncertainty values of the branching ratios of {nuclide} are "
                             "too constrained to correct the sampled values back to summing to 100%.")

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

    return ratios


# ----- Custom DAG Datastructure --------------------
class DecayChainDAG:
    def __init__(self):
        """
        DAG representation of a radioactive decay chain.
 
        Nodes are nuclides identified by `NuclideID`. Directed edges represent decay transitions
        from parent to daughter nuclide, weighted by branching probability (%).
 
        The graph is shared across all `BatemanEqnSolver` instances — one solver per root nuclide
        reads from the same DAG. Thread-safe for concurrent reads.
        """

        self.nuclides: dict[NuclideID, Nuclide] = {}
    

    # ----- Private Methods --------------------
    def _perturbed_nuclide_data(self, nuclide: NuclideID, rng: np.random.Generator) -> BatemanCalcData:
        """
        Returns perturbed data of the requested nuclide, where the perturbed values are within their 
        uncertainty ranges.

        Perturbed value sampled using Normal distribution on the allowed uncertainty range.

        Parameters
        ----------
        - `nuclide`:    Nuclide identifier of the form `(symbol, meta, mass_number)`
        - `rng`:        Caller-owned `np.random.Generator` instance

        Returns
        -------
        `BatemanCalcData` instance with sampled `decay_const` and `decay_transitions`.
        """

        node = self.nuclides[nuclide]

        decay_const = _perturb_decay_const(rng, node.decay_const, node.decay_unc)
        transitions = self._sample_transitions(nuclide, node.decay_transitions, rng)

        return BatemanCalcData(
            nuclide             = node.nuclide,
            decay_const         = decay_const,
            decay_transitions   = transitions
        )


    def _sample_transitions(self, nuclide: NuclideID, decay_transitions: list[DecayTransition], 
                            rng: np.random.Generator) -> list[TransitionEdge]:
        """
        Returns a list sampled branching ratios (1 sample for each decay transition) where each 
        sample lies inside the uncertainty range of their respective branching ratio.

        Values sampled using Normal distribution.
        Ensures that the sum of all the branching ratios add up to 100% while each branching ratio 
        remains within the 0-100 range (inclusive).
        
        Workflow
        --------
        - Sample a branching ratio value for each decay transition.
        - Adjust sample values such that their sum adds up to 100% while each of them lie within 
          the 0-100 range.

        Parameters
        ----------
        - `decay_transitions`:  Transitions belonging to a single parent nuclide
        - `rng`:                Caller-owned `np.random.Generator` instance

        Returns
        -------
        List of `(daughter nuclide, sampled branching ratio)` tuples.
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

        ratios = _redistribute_to_total(ratios, uncs, nuclide)

        return [(t.nuclide, ratios[i]) for i, t in enumerate(decay_transitions)]


    # ----- Public Methods --------------------
    def add_nuclide(self, nuclide: NuclideID, decay_const: float, decay_unc: float):
        """
        Adds a nuclide node to the decay chain.
 
        Skips silently if the nuclide already exists — physics guarantees the data would be
        identical.
 
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
        Adds a decay transition to the specifc `parent` nuclide.

        Skips silently if the decay transition already exists — physics guarantees the data would 
        be identical.

        Parameters
        ----------
        - `parent`:             Nuclide identifier of parent nuclide of the form `(symbol, meta, mass_number)`
        - `daughter`:           Nuclide identifier of daughter nuclide of the form `(symbol, meta, mass_number)`
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
        Returns a copy of the `Nuclide` data for the requested nuclide required for Bateman 
        equation calculations.
        Raises `RuntimeError` if nuclide is not in decay chain DAG.

        If `rng` is `None`, returns a copy of the nuclide's stored (unperturbed) data. 
        If `rng` is provided, returns freshly perturbed data — computed fresh on every call.
 
        Returns a copy rather than a direct reference to protect graph integrity when the DAG
        is shared across multiple solvers and threads.
 
        Parameters
        ----------
        - `nuclide`: Nuclide identifier as `(symbol, meta, mass_number)`
        - `rng`:     Caller-owned `np.random.Generator` instance, or `None` for unperturbed data
 
        Returns
        -------
        `BatemanCalData` instance.
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
        """
        Fills out uncertainty values for decay constants and branching ratios when they are 
        undefined.

        Convention
        ----------
        - Decay constant uncertainty values are set to 0.
        - Branching ratio uncertainty values are set to the minimum branching ratio value for that 
          nuclide.
        """

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
        """
        Returns nuclides in DAG which are missing uncertainty values.

        Returns tuple of the form:
        `(set of nuclides missing decay uncertainty, set of nuclides missing branching ratio uncertainty)`
        """

        decay_uncs  = []
        br_uncs     = []

        for nuclide, data in self.nuclides.items():
            if data.decay_unc is None:
                decay_uncs.append(nuclide)
            
            for transition in data.decay_transitions:
                if transition.branching_unc is None:
                    br_uncs.append(nuclide)
        
        return (set(decay_uncs), set(br_uncs))