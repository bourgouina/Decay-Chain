import csv
from pathlib import Path

from decay_chain_dag import DecayChainDAG


# ----- Shared Data Structures/Caches --------------------
decay_chain_graph                           = DecayChainDAG()
symbol_to_atomic_num_map : dict[str, int]   = {}


# ----- Data Loaders & Data Structure Builders --------------------
def load_symbol_to_atomic_num_map():
    """
    """

    path = Path(__file__).parent.parent / "data" / "elements.csv"

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            symbol_to_atomic_num_map[row["symbol"]] = int(row["atomic_num"])


def build_decay_chain_graph():
    """
    """