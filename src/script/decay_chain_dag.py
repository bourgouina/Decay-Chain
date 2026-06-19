from __future__ import annotations

from dataclasses import dataclass


# ----- Custom Types --------------------
NuclideID       = tuple[str, str, int]              # (symbol, meta, mass no.)
TransitionEdge  = tuple[NuclideID, float]           # (nuclide, % chance to take that transition)


# ----- Constants --------------------
MAX_TRANSITIONS = 10    # Max no. of different decay transitions theoretically possible


# ----- Data Classes --------------------
@dataclass
class Nuclide:
    nuclide:            NuclideID
    decay_transitions:  list[TransitionEdge]


# ----- Custom DAG Datastructure --------------------
class DecayChainDAG:
    def __init__(self):
        """
        """

        self.nuclides: dict[NuclideID, Nuclide] = {}
    

    def add_nuclide(self, nuclide: NuclideID, decay_transitions: list[TransitionEdge]):
        """
        """

        if nuclide in self.nuclides:
            return

        assert len(decay_transitions) <= MAX_TRANSITIONS, (
            f"An element can have a maximum of {MAX_TRANSITIONS} decay transitions." 
            f"decay_transitions param list has {len(decay_transitions)} decay transitions."
        )

        self.nuclides[nuclide] = Nuclide(
            nuclide=nuclide,
            decay_transitions=decay_transitions
        )