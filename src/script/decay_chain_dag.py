from __future__ import annotations

from dataclasses import dataclass


# ----- Custom Types --------------------
NuclideID       = tuple[str, str, int]      # (symbol, meta, mass no.)
TransitionEdge  = tuple[NuclideID, float]   # (nuclide, branching ratio)


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