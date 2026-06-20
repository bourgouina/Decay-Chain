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
    Stores relevant nuclide data:
    - Identification (symbol, meta, mass no.)
    - Decay constant value (lambda)
    - Weighted decay transitions to other elements (weights stored as % in 0-100 range)
    """

    nuclide:            NuclideID
    decay_const:        float
    decay_transitions:  list[TransitionEdge]


# ----- Custom DAG Datastructure --------------------
class DecayChainDAG:
    def __init__(self):
        """
        DAG model to represent a decay chain where the edge weight between nuclides A -> B 
        represents the % chance of that decay transition occuring.
        """

        self.nuclides: dict[NuclideID, Nuclide] = {}
    

    def add_nuclide(self, nuclide: NuclideID, decay_const: float, 
                    decay_transitions: list[TransitionEdge]):
        """
        Adds a nucleide to the decay chain along with its decay constant (lambda) and weighted 
        decay transitions (in %).
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
        """Returns a `Nuclide` instance copy of the requested nuclide."""

        return Nuclide(
            nuclide             = self.nuclides[nuclide].nuclide,
            decay_const         = self.nuclides[nuclide].decay_const,
            decay_transitions   = self.nuclides[nuclide].decay_transitions
        )