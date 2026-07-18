import csv
from pathlib import Path

from decay_chain_dag import DecayChainDAG, NuclideID


# ----- Shared Data Structures/Caches --------------------
decay_chain_graph                           = DecayChainDAG()
symbol_to_atomic_num_map : dict[str, int]   = {}

#  ----- Constants --------------------
DATA_DIR_PATH               = Path(__file__).parent.parent.parent / "data"
ELEMENTS_CSV_FILEPATH       = DATA_DIR_PATH  / "elements.csv"
NUCLIDES_CSV_FILEPATH       = DATA_DIR_PATH / "nuclides.csv"
DECAY_BRANCHES_CSV_FILEPATH = DATA_DIR_PATH / "decay_branches.csv"


# ----- Data Loaders & Data Structure Builders --------------------
def load_symbol_to_atomic_num_map(filepath: Path = ELEMENTS_CSV_FILEPATH):
    """Loads element symbols and their respective atomic numbers from CSV."""

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            symbol_to_atomic_num_map[row["symbol"]] = int(row["atomic_num"])


def build_decay_chain_graph(nuclide_filepath: Path = NUCLIDES_CSV_FILEPATH,
                            decay_filepath: Path = DECAY_BRANCHES_CSV_FILEPATH,
                            fill_missing_data: bool = False):
    """
    Builds decay chain DAG by loading data from CSVs.
    Raises `RuntimeError` if `fill_missing_data` is `False` and DAG is missing uncertainty data.
    """

    # Load data from Nuclides CSV
    with open(nuclide_filepath, "r", encoding="ascii") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract values from CSV
            nuclide_id: NuclideID   = (row["symbol"], row["meta"], int(row["mass_num"]))
            decay_const             = float(row["decay_const"])
            decay_unc               = float(row["decay_unc"]) if row["decay_unc"] else None

            decay_chain_graph.add_nuclide(nuclide_id, decay_const, decay_unc)
    
    # Load data from Decay Branches CSV
    with open(decay_filepath, "r", encoding="ascii") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract values from CSV
            parent_id: NuclideID    = (row["parent_symbol"], row["parent_meta"], int(row["parent_mass_num"]))
            daughter_id: NuclideID  = (row["daughter_symbol"], row["daughter_meta"], int(row["daughter_mass_num"]))
            decay_type              = row["decay_type"]
            branching_ratio         = float(row["branching_ratio"])
            branching_unc           = float(row["branching_unc"]) if row["branching_unc"] else None

            # Add decay transition of parent
            decay_chain_graph.add_transition(parent_id, daughter_id, decay_type, branching_ratio, 
                                             branching_unc)
            
            # Add daughter as a stable nuclide if it doesnt already exist in the DAG
            decay_chain_graph.add_nuclide(daughter_id, 0.0, 0.0)
    
    # Fill/verify missing data depending on mode
    if fill_missing_data:
        decay_chain_graph.fill_missing_data()
    else:
        decay_uncs, br_uncs = decay_chain_graph.get_missing_data()

        if decay_uncs or br_uncs:
            raise RuntimeError(f"Nuclides missing decay uncertainty: {decay_uncs}\n"
                               f"Nuclides missing branching ratio uncertainties: {br_uncs}")