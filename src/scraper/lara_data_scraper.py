from __future__ import annotations

import csv
import shutil
from pathlib import Path

from http_client import HTTPClient
from shared import NuclideData
from lara_ensdf_parser import LARA_ENSDFParser
from lara_txt_parser import LARA_TxtParser
from lara_data_fetcher import LARADataFetcher, FETCH_WORKER_COUNT


# ----- Constants --------------------
TMP_DIR_PATH    = Path(__file__).parent / "tmp"
DATA_DIR        = Path(__file__).parent.parent.parent / "data"

NUCLIDES_CSV_HEADER     = ["symbol", "meta", "mass_num", 
                           "is_stable", "half_life_s", "decay_const", "decay_unc"]
TRANSITIONS_CSV_HEADER  = ["parent_symbol", "parent_meta", "parent_mass_num",
                           "daughter_symbol", "daughter_meta", "daughter_mass_num",
                           "decay_type", "branching_ratio", "branching_uncertainty"]


# ----- LARAWeb Data Scraper --------------------
class LARADataScraper:
    # ----- Private Methods --------------------
    @staticmethod
    def _make_tmp_dir():
        """Created a temporary folder to store raw data files before they are processed."""

        # Remove folder if it already exists
        if TMP_DIR_PATH.exists():
            shutil.rmtree(TMP_DIR_PATH)
        
        TMP_DIR_PATH.mkdir()
    

    @staticmethod
    def _parse_file(filepath: Path) -> NuclideData:
        """
        Parses raw data file depending on file type.
        Throws `RuntimeError` if unexpected file type is encountered.

        File Types Handled:
        - `.ensdf`  - For LARA ENSDF files
        - `.txt`    - For LARA text files
        """

        parsed_data: NuclideData = None

        ensdf_parser    = LARA_ENSDFParser()
        txt_parser      = LARA_TxtParser()

        if filepath.suffix == ".ensdf":
            parsed_data = ensdf_parser.parse(filepath)
        elif filepath.suffix == ".txt":
            parsed_data = txt_parser.parse(filepath)
        else:
            raise RuntimeError(f"Unexpected file type: {filepath.name!r}")
        
        return parsed_data


    @staticmethod
    def _write_to_csvs(parsed_data: list[NuclideData]):
        """
        Writes parsed data to CSVs.

        Parameter
        ---------
        - `parsed_data`: List of `NuclideData` objects containing parsed data.
        """

        # CSV file paths
        nuclides_csv_filepath       = DATA_DIR / "nuclides.csv"
        decay_branch_csv_filepath   = DATA_DIR / "decay_branches.csv"

        with open(nuclides_csv_filepath, "w", encoding="utf-8", newline="") as n, \
            open(decay_branch_csv_filepath, "w", encoding="utf-8", newline="") as d:
            nuclide_writer  = csv.writer(n)
            decay_writer    = csv.writer(d)

            # Write CSV headers
            nuclide_writer.writerow(NUCLIDES_CSV_HEADER)
            decay_writer.writerow(TRANSITIONS_CSV_HEADER)

            # Iterate through nuclide data and write them to CSVs
            for nuclide in parsed_data:
                nuclide_writer.writerow([nuclide.symbol,
                                         nuclide.meta,
                                         nuclide.mass_num,
                                         nuclide.stable,
                                         nuclide.half_life if nuclide.half_life else "",
                                         nuclide.decay_const if nuclide.decay_const else "",
                                         nuclide.decay_unc if nuclide.decay_unc else ""])
                
                # Write decay transition data
                for daughter in nuclide.transitions:
                    decay_writer.writerow([nuclide.symbol,
                                           nuclide.meta,
                                           nuclide.mass_num,
                                           daughter.symbol,
                                           daughter.meta,
                                           daughter.mass_num,
                                           daughter.decay_type,
                                           daughter.branch_pct,
                                           daughter.branch_unc if daughter.branch_unc else ""])
    

    @staticmethod
    def _move_raw_data_files():
        """
        Moves folder containing raw data files to data folder.
        """

        renamed_dir = TMP_DIR_PATH.rename("raw_data_files")

        # Remove previous raw data files if they exist
        if (DATA_DIR / renamed_dir).exists():
            shutil.rmtree((DATA_DIR / renamed_dir))

        shutil.move(renamed_dir, DATA_DIR)


    # ----- Public Methods --------------------
    @staticmethod
    def run():
        """Runs data scraping pipeline."""

        LARADataScraper._make_tmp_dir()     # Make temporary folder to store raw data files
        nuclide_count = 0

        # Fetch and store raw data from LARAWeb
        with HTTPClient(pool_maxsize=FETCH_WORKER_COUNT+1, multithreaded=True) as client:
            client.register_thread()
            
            data_fetcher    = LARADataFetcher(client, TMP_DIR_PATH)
            nuclide_count   = data_fetcher.fetch_all_nuclide_data()
        
        # Parse data
        parsed_data: list[NuclideData] = [None] * nuclide_count

        for i, file in enumerate(TMP_DIR_PATH.iterdir()):
            parsed_data[i] = LARADataScraper._parse_file(file)
        
        NuclideData.order(parsed_data)
        print(f"All {nuclide_count} nuclide data files successfuly parsed.")
        
        # Store data
        LARADataScraper._write_to_csvs(parsed_data)
        print("Data successfully written to CSVs")

        # Move raw data files to data folder
        LARADataScraper._move_raw_data_files()


# Entrypoint if run as root
if __name__ == "__main__":
    LARADataScraper.run()