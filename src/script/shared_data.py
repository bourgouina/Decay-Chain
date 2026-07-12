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
                            decay_filepath: Path = DECAY_BRANCHES_CSV_FILEPATH):
    """Builds decay chain DAG by loading data from CSVs."""

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

            decay_chain_graph.add_transition(parent_id, daughter_id, decay_type, branching_ratio, 
                                             branching_unc)