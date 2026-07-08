from __future__ import annotations

import re
import threading
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import HTTPError

from http_client import HTTPClient
from shared import ParseError, InformationFetchError


# ----- Constants --------------------
_LARA_ENSDF_FILE_BASE   = "http://www.lnhb.fr/nuclides"
_LARA_TXT_FILE_BASE     = "http://www.lnhb.fr/Laraweb/Results"
_LARA_NUCLIDE_LIST_URL  = "http://www.lnhb.fr/Laraweb/Choix_Lara.php"

_OPTIONS_NUCLIDE_RE     = re.compile(r"(\d+)([A-Z][a-z]?)(\-M)?(EQUI)?")
FETCH_WORKER_COUNT      = 20


# ----- Helper Functions --------------------
def _lara_ensdf_file_url(nuclide: str) -> str:
    """For a given nuclide identifier, returns URL for its ENSDF file on LARAWeb."""

    return f"{_LARA_ENSDF_FILE_BASE}/{nuclide}.txt"

def _lara_txt_file_url(nuclide: str) -> str:
    """For a given nuclide identifier, returns URL for its LARAWeb text file."""

    return f"{_LARA_TXT_FILE_BASE}/{nuclide}.lara.txt"


# ----- LARAWeb Data Fetcher --------------------
class LARADataFetcher:
    def __init__(self, client: HTTPClient, dir: Path):
        """
        Fetches and stores raw nuclide data from LARAWeb.

        Responsibilities
        ----------------
        - Extract the list of all nuclides for which LARAWeb has data.
        - For each nuclide fetch and store its raw data file in a temporary folder.

        Parameters
        ----------
        - `client`: HTTP client for all web requests
        - `dir`:    Temporary folder in which raw data files are to be stored
        """

        self._client: HTTPClient    = client
        self._dir: Path             = dir


    # ----- Private Methods --------------------
    def _get_nuclides_list(self) -> list[str]:
        """
        Fetches the HTML of the LARAWeb webpage which contains the GUI for looking up nuclide data.
        From there, the list of nuclides which LARAWeb has data on is extracted.

        HTML Format
        -----------
        - The list of nuclides is inside a `<select>` tag containing a list of `<option>` tags.
        - The body of each `<option>` tag contains the identifier of a nuclide.

        Raises `ParseError` if unexpected HTML format/content is detected.
        """

        nuclides: list[str] = []
        soup: BeautifulSoup = None

        # Fetch HTML of the LARA webpage containing the list of nuclides whose data is available
        html = self._client.get_text(_LARA_NUCLIDE_LIST_URL)
        soup = BeautifulSoup(html, "html.parser")
        
        select_tag = soup.find("select", attrs={"name": "Nuclide[]"})

        # If the HTML does not contain a <select> tag, then the HTML received is unexpected
        # To prevent processing incorrect data, raise error and stop the program
        if not select_tag:
            raise ParseError('Expected <select name="Nuclide[]">...</select> tag in HTML.')
        
        # For each nuclide normalize its string and add it to the list of nuclides
        for option_tag in select_tag.find_all("option"):
            text = option_tag.get_text(strip=True)

            # If <option> tag body is empty, then the <option> tag has an unexpected body
            # To prevent processing incorrect data, raise error and stop the program
            if not text:
                raise ParseError("Expected non-empty <option> tag.")
            
            text    = re.sub(r"\s+", r"", text)
            m       = _OPTIONS_NUCLIDE_RE.fullmatch(text)

            # If the regex does not match, then the <option> tag body has some unexpected data
            # To prevent processing incorrect data, raise error and stop the program
            if not m:
                raise ParseError(f"<option> tag body contains unexpected data: {text!r}")
            
            mass_num, symbol, isomer, equi = m.groups()

            # Skip EQUI nuclides
            if equi:
                continue

            nuclide = f"{symbol}-{mass_num}{'m' if isomer else ''}"
            nuclides.append(nuclide)
        
        return nuclides
    

    def _fetch_one(self, nuclide: str, counter: list[int], lock: threading.Lock, total: int):
        """
        Worker for `fetch_all_nuclide_data`. Downloads the raw data file for a given nuclide.

        Workflow
        --------
        - Attempts to download ENSDF file of nuclide.
        - If unsuccessful, attempts to download LARAWeb text file of nuclide.
        - If both are unsuccessful, raises `InformationFetchError`.


        Parameters
        ----------
        - `nuclide`: Nuclide identifier
        - `counter`: Single element list which acts as a shared counter for no. of nuclides fetched
        - `lock`:    Lock protecting `counter`
        - `total`:   Total no. of entries, used for progress display
        """

        self._client.register_thread()

        raw_bytes: bytes    = None
        extension: str      = None

        # If ENSDF file does not exist for nuclide, fetch its LARA text file
        # If even the LARA text file does not exist, data for nuclide does not exist
        # To prevent using incomplete data, raise error and stop the program
        try:
            ensdf_file_url  = _lara_ensdf_file_url(nuclide)
            raw_bytes       = self._client.get_raw_bytes(ensdf_file_url)
            extension       = "ensdf"
        except HTTPError:
            try:
                txt_file_url = _lara_txt_file_url(nuclide)
                raw_bytes    = self._client.get_raw_bytes(txt_file_url)
                extension    = "txt"
            except HTTPError as e:
                raise InformationFetchError(f"Neither ENSDF nor LARA text file exists for {nuclide}") \
                    from e

        
        filename = f"{nuclide}.{extension}"     # Decide file extension depending on source
        filepath = self._dir / filename

        # Write raw data to local file
        filepath.write_bytes(raw_bytes)

        # To provide update on data fetching progress
        with lock:
            counter[0] += 1

            if counter[0] % 10 == 0:
                print(f"Fetched and stored raw data of {counter[0]}/{total} ",
                      f"nuclides ({int(counter[0] / total * 100)}%)")
        

    # ----- Public Methods --------------------
    def fetch_all_nuclide_data(self):
        """
        Fetches and stores raw nuclide data files for all nuclides available on LARAWeb in parallel 
        using a thread pool of size `FETCH_WORKER_COUNT`.
        """

        # Single-threaded section: Fetch list of all nuclides whose data is available on LARAWeb
        nuclides    = self._get_nuclides_list()
        total       = len(nuclides)

        # Multi-threaded section: Fetch and write raw nuclide data to files in local dir
        counter = [0]
        lock    = threading.Lock()

        with ThreadPoolExecutor(max_workers=FETCH_WORKER_COUNT) as executor:
            futures = [executor.submit(self._fetch_one, nuclide, counter, lock, total) 
                        for nuclide in nuclides]

            for future in futures:
                future.result()

        print(f"\nFetched and stored raw data for {total} nuclides.\n")