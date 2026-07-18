import re
import csv
import sys
import numpy as np
from pathlib import Path

from shared_data import decay_chain_graph, build_decay_chain_graph
from decay_chain_dag import NuclideID
from mc_trial_manager import MCTrialManager, RootResult


# ----- Constants --------------------
# Regexes
_TRIAL_COUNT_RE     = re.compile(r"--trials=(\d+)")
_ROOT_RE            = re.compile(r"--root=([A-Z][a-z]?)(\d+)(m)?")
_N0_RE              = re.compile(r"--N0=(\d+)")
_LOWERBOUND_RE      = re.compile(r"--lower=(\d+(\.\d+)?([eE][+-]\d+)?)")
_UPPERBOUND_RE      = re.compile(r"--upper=(\d+(\.\d+)?([eE][+-]\d+)?)")
_TIMESTAMP_COUNT_RE = re.compile(r"--timestamps=(\d+)")

# NuclideID tuple indexes
NUCID_SYMBOL    = 0
NUCID_META      = 1
NUCID_MASSNUM   = 2

# Result directory path
RESULT_DIRPATH = Path(__file__).parent.parent.parent / "results"

# Output CSV headers
CSV_HEADERS = ["timestamp_in_s", 
               "nuclide_symbol", "nuclide_meta", "nuclide_mass_num",
               "activity_val", "activity_std"]


# ----- Helper Methods --------------------
def parse_params(args: list[str]) -> tuple[NuclideID, int, int, np.ndarray]:
    """
    Parses CLI arguments to program parameters.
    Raises `RuntimeError` if arguments contain a duplicate parameter, unknown parameter or is missing 
    required parameters.

    Parameter
    ---------
    - `args`: List of CLI arguments

    Returns
    -------
    A tuple  of the form `(root nuclide, N0, no. of trials, timestamps array)`
    """

    root = N0 = trial_count = lowerbound = upperbound = timestamp_count = None

    # Extract parameters from args
    for i in range(len(args)):
        # Root parameter parsing
        if _ROOT_RE.fullmatch(args[i]):
            if root is not None:
                raise RuntimeError(f'Duplicate "--root" parameter: {args[i]!r}')
            
            m = _ROOT_RE.fullmatch(args[i])

            symbol      = m.group(1)
            mass_num    = int(m.group(2))
            meta        = m.group(3) or ""

            root: NuclideID = (symbol, meta, mass_num)
            continue

        # N0 parameter parsing
        if _N0_RE.fullmatch(args[i]):
            if N0 is not None:
                raise RuntimeError(f'Duplicate "--N0" parameter: {args[i]!r}')
            
            N0 = int(_N0_RE.fullmatch(args[i]).group(1))
            continue

        # Trial count parameter parsing
        if _TRIAL_COUNT_RE.fullmatch(args[i]):
            if trial_count is not None:
                raise RuntimeError(f'Duplicate "--trials" parameter: {args[i]!r}')
            
            trial_count = int(_TRIAL_COUNT_RE.fullmatch(args[i]).group(1))
            continue
        
        # Timestamps lowerbound parameter parsing
        if _LOWERBOUND_RE.fullmatch(args[i]):
            if lowerbound is not None:
                raise RuntimeError(f'Duplicate "--lower" parameter: {args[i]!r}')
            
            lowerbound = float(_LOWERBOUND_RE.fullmatch(args[i]).group(1))
            continue
        
        # Timestamps upperbound parameter parsing
        if _UPPERBOUND_RE.fullmatch(args[i]):
            if upperbound is not None:
                raise RuntimeError(f'Duplicate "--upper" parameter: {args[i]!r}')
            
            upperbound = float(_UPPERBOUND_RE.fullmatch(args[i]).group(1))
            continue
        
        # Timestamps count parameter parsing
        if _TIMESTAMP_COUNT_RE.fullmatch(args[i]):
            if timestamp_count is not None:
                raise RuntimeError(f'Duplicate "--timestamps" parameter: {args[i]!r}')
            
            timestamp_count = int(_TIMESTAMP_COUNT_RE.fullmatch(args[i]).group(1))
            continue

        # If this section is reached, then there is an unrecognized parameter
        raise RuntimeError(f"Unrecognized parameter: {args[i]!r}")
    
    # Raise RuntimeError if any required argument is missing
    if (root is None) or (N0 is None) or (trial_count is None) or (lowerbound is None) or \
        (upperbound is None) or (timestamp_count is None):
        raise RuntimeError("Missing required parameters.\n"
                           "USEAGE: python -m main --root=<ROOT_NUCLIDE> --N0=<INT> --trials=<INT> "
                           "--lower=<FLOAT> --upper=<FLOAT> --timestamps=<INT>\n"
                           "(parameters can be listed in a different order)")
    
    return (root, N0, trial_count, np.linspace(lowerbound, upperbound, timestamp_count))


def output_writer(root: NuclideID, results: RootResult, timestamps: np.ndarray):
    """
    Writes MC calculation results to CSV file.

    Parameters
    ----------
    - `root`:       Root nuclide for calculations
    - `results`:    Activity results for each nuclide at each timestamp
    - `timestamps`: Numpy array of timestamps at which activity was evaluated
    """

    # Create result folder if it does not exist
    RESULT_DIRPATH.mkdir(exist_ok=True)

    filename = (f"{root[NUCID_SYMBOL]}{root[NUCID_MASSNUM]}{"m" if root[NUCID_META] else ""}.csv")
    filepath = RESULT_DIRPATH / filename

    # Write output data to CSV file
    with open(filepath, "w", encoding="ascii", newline="") as f:
        csv_writer = csv.writer(f)

        # Write CSV header
        csv_writer.writerow(CSV_HEADERS)

        # Write data
        for i in range(len(timestamps)):
            for nuclide in results.non_perturbed_activity.keys():
                csv_writer.writerow([timestamps[i],
                                     nuclide[NUCID_SYMBOL],
                                     nuclide[NUCID_META],
                                     nuclide[NUCID_MASSNUM],
                                     results.non_perturbed_activity[nuclide][i],
                                     results.perturbed_activity_std[nuclide][i]])


# ----- Entrypoint --------------------
def main():
    """Entry point to Monte Carlo activity value calculator."""

    root, N0, trial_count, timestamps = parse_params(sys.argv[1:])
    build_decay_chain_graph()

    mc_manager = MCTrialManager(decay_chain_graph, trial_count)
    results = mc_manager.compute(root, N0, timestamps)

    output_writer(root, results, timestamps)


if __name__ == "__main__":
    main()