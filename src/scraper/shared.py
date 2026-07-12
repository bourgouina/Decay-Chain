from __future__ import annotations

from dataclasses import dataclass


# ----- Data Classes --------------------
@dataclass
class NuclideData:
    """
    Stores relevant nuclide data.

    Attributes
    ----------
    - `symbol`:         Nuclide element symbol
    - `meta`:           Meta identifier ("m" if meta, else "")
    - `mass_num`:       Mass no. of nuclide
    - `stable`:         Is nuclide stable (True/False)
    - `half_life`:      Half-life of nuclide in seconds, `None` if stable
    - `decay_const`:    Decay constant in s^-1, `None` if stable
    - `decay_unc`:      Decay constant uncertainty in s^-1, `None` if data not provided
    - `transitions`:    List of decay transitions
    """

    symbol:         str
    meta:           str
    mass_num:       int
    stable:         bool
    half_life:      float | None
    decay_const:    float | None
    decay_unc:      float | None
    transitions:    list[TransitionData]


    @staticmethod
    def order(nuclides: list[NuclideData]):
        """Arranges `NuclideData` objects in ascending order inside list."""

        nuclides.sort(key=lambda n: (n.mass_num, n.symbol, n.meta))


@dataclass
class TransitionData:
    """
    Stores relevant decay transition data.

    Attributes
    ----------
    - `symbol`:         Nuclude element symbol of daughter nuclide
    - `meta`:           Meta identifier of daughter nuclide
    - `mass_num`:       Mass no. of daughter nuclide
    - `decay_type`:     Decay type (alpha, beta+, beta-, ...)
    - `branch_pct`:     % chance of that branch being taken, [0-100] range
    - `branch_unc`:     Uncertainty in branching percentage, [0-100] range, `None` if no data provided
    """

    symbol:     str
    meta:       str
    mass_num:   int
    decay_type: str
    branch_pct: float
    branch_unc: float | None


    @staticmethod
    def order(transitions: list[TransitionData]):
        """Arranges `TransitionData` objects in ascending order inside list."""

        transitions.sort(key=lambda t: (t.mass_num, t.symbol, t.meta))


# ----- Exceptions --------------------
class InformationFetchError(Exception):
    """Raised when fetching data from LARAWeb fails."""

class ParseError(Exception):
    """Raised when parsing fetched data fails."""