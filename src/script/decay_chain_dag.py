from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ----- Custom Types --------------------
NuclideID       = tuple[str, str, int]      # (symbol, meta, mass no.)
TransitionEdge  = tuple[NuclideID, float]   # (nuclide, branching ratio)


# ----- Constants --------------------
_BRANCHING_TOTAL_PCT = 100.0    # Branching ratios for a nuclide's transitions must sum to this


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
def _sample_clipped_gaussian(rng: np.random.Generator, value: float, unc: float) -> float:
    """
    Samples a value from within its uncertainty range.

    Sampled using Normal distribution.
    Samples less than 0 clipped to 0.

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
    return sample if sample > 0.0 else 0.0


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

        decay_const = _sample_clipped_gaussian(rng, node.decay_const, node.decay_unc)
        transitions = self._sample_transitions(node.decay_transitions, rng)

        return BatemanCalcData(
            nuclide             = node.nuclide,
            decay_const         = decay_const,
            decay_transitions   = transitions
        )


    def _sample_transitions(self, decay_transitions: list[DecayTransition], 
                            rng: np.random.Generator) -> list[TransitionEdge]:
        """
        Returns a list sampled branching ratios (1 sample for each decay transition) where each 
        sample lies inside the uncertainty range of their respective branching ratio.

        Values sampled using Normal distribution.
        Ensures that the sum of all the branching ratios add up to 100%.
        
        Workflow
        --------
        - Sample a branching ratio value for each decay transition.
        - Calculate the correction factor of how much the sum of the branching ratios need to be 
          changed to add upto 100%.
        - Adjust each branching ratio such that their sum adds up to 100% where adjustment factor 
          depends on the uncertainty variance of the branching ratio.

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
            (_sample_clipped_gaussian(rng, t.branching_ratio, t.branching_unc) 
             for t in decay_transitions),
            dtype=np.float64, count=n
        )

        # Calculate adjustment required for branching ratio samples sum is off from 100% 
        # (can be positive or negative)
        correction = _BRANCHING_TOTAL_PCT - ratios.sum()

        # Skip adjustment calculations if no adjustment is required (almost never the case)
        if correction != 0.0:
            # Calculate the uncertainty variance of each branching ratio
            uncs         = np.fromiter((t.branching_unc for t in decay_transitions), 
                                       dtype=np.float64, count=n)
            weights      = uncs ** 2
            total_weight = weights.sum()

            # If the variance of all weights are 0, then split adjustments evenly across all 
            # branching ratio samples
            if total_weight <= 0.0:
                weights, total_weight = np.ones(n), float(n)

            # Adjust sample values based on their variance
            ratios += correction * (weights / total_weight)

        np.clip(ratios, 0.0, 100.0, out=ratios)

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


    def read_nuclide_data(self, nuclide: NuclideID) -> BatemanCalcData:
        """
        Returns a copy of the `Nuclide` data for the requested nuclide required for Bateman 
        equation calculations.
 
        Returns a copy rather than a direct reference to protect graph integrity when the DAG
        is shared across multiple solvers and threads.
 
        Parameters
        ----------
        - `nuclide`: Unique identifier as `(symbol, meta, mass_number)`
 
        Returns
        -------
        `BatemanCalData` instance.
        """

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