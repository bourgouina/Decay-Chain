from __future__ import annotations

from dataclasses import dataclass


# ----- Custom Types --------------------
NuclideID       = tuple[str, str, int]              # (symbol, meta, mass no.)
TransitionEdge  = tuple[NuclideID, float]           # (nuclide, % chance to take that transition)


# ----- Constants --------------------
"""
Decay Transition Types:
1.  Alpha decay
2.  Beta-minus decay
3.  Beta-plus decay
4.  Electron capture
5.  Gamma decay
6.  Internal conversion
7.  Spontaneous fission
8.  Cluster decay
9.  Proton emission
10. Neutron emission
11. Double beta decay
12. Double proton emission
"""
MAX_TRANSITIONS = 12    # Max no. of different decay transitions types theoretically possible


# ----- Data Classes --------------------
@dataclass
class Nuclide:
    """
    Stores relevant nuclide data.
 
    Attributes
    ----------
    - `nuclide`:           Unique identifier for the nuclide as `(symbol, meta, mass_number)`
    - `decay_const`:       Decay constant lambda (s^-1), where lambda = ln(2) / half_life
    - `decay_transitions`: Weighted edges to daughter nuclides, weights stored as branching 
                           probability in % (0-100 range)
    """

    nuclide:            NuclideID
    decay_const:        float
    decay_transitions:  list[TransitionEdge]


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
    

    def add_nuclide(self, nuclide: NuclideID, decay_const: float, 
                    decay_transitions: list[TransitionEdge]):
        """
        Adds a nuclide node to the decay chain.
 
        Skips silently if the nuclide already exists — physics guarantees the data would be
        identical regardless of which decay chain first added it.
 
        Parameters
        ----------
        -c`nuclide`:            Unique identifier as `(symbol, meta, mass_number)`
        - `decay_const`:        Decay constant lambda (s^-1)
        - `decay_transitions`:  List of `(daughter_NuclideID, branching_prob)` edges.
                                Branching probabilities in % (0-100). Must not exceed 
                                `MAX_TRANSITIONS` entries.
        """

        # Skip addition if nuclide node already exists in DAG 
        # (physics garuntees that the values would have been the same)
        if nuclide in self.nuclides:
            return

        assert len(decay_transitions) <= MAX_TRANSITIONS, (
            f"An element can have a maximum of {MAX_TRANSITIONS} decay transitions. " 
            f"decay_transitions param list has {len(decay_transitions)} decay transitions."
        )

        self.nuclides[nuclide] = Nuclide(
            nuclide             = nuclide,
            decay_const         = decay_const,
            decay_transitions   = decay_transitions
        )

    def read_nuclide_data(self, nuclide: NuclideID) -> Nuclide:
        """
        Returns a copy of the `Nuclide` data for the requested nuclide.
 
        Returns a copy rather than a direct reference to protect graph integrity when the DAG
        is shared across multiple solvers and threads.
 
        Parameters
        ----------
        - `nuclide`: Unique identifier as `(symbol, meta, mass_number)`
 
        Returns
        -------
        `Nuclide` instance copy
        """

        return Nuclide(
            nuclide             = self.nuclides[nuclide].nuclide,
            decay_const         = self.nuclides[nuclide].decay_const,
            decay_transitions   = self.nuclides[nuclide].decay_transitions.copy()
        )