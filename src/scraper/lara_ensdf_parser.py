from __future__ import annotations

import re
import math
from pathlib import Path

from shared import NuclideData, TransitionData, ParseError


# ----- ENSDF Column Layout Constants --------------------
# ENSDF documentation and comments using 1-based indexing but the code uses 0-based indexing.

_CONT_FLAG_OFFSET   = 5         # Col 6:        Continuation number (blank -> primary record)
_RECTYPE_OFFSET     = 7         # Col 8:        Single-char record type designator (P, N, ...)

# NUCID (Nuclide Identifier) Sub-field Column Mapping
_NUCID_MASS_OFFSET = 0
_NUCID_MASS_LENGTH = 3          # Col 1-3:      Nuclide mass no. field
_NUCID_ELEM_OFFSET = 3
_NUCID_ELEM_LENGTH = 2          # Col 4-5:      Nuclide symbol field

# Identification Record Column Mapping
_ID_NUCID_OFFSET = 0
_ID_NUCID_LENGTH = 5            # Col 1-5:      Daughter nuclide identifier string field
_ID_DSID_OFFSET = 9
_ID_DSID_LENGTH = 30            # Col 10-39:    Decay type field

# P Record Column Mapping
_P_HALFLIFE_OFFSET = 39
_P_HALFLIFE_LENGTH = 10         # Col 40-49:    Parent nuclide half-life field
_P_HALFLIFE_UNC_OFFSET = 49
_P_HALFLIFE_UNC_LENGTH = 6      # Col 50-55:    Parent nuclide half-life uncertainty field

# N Record Column Mapping
_N_NR_OFFSET = 9
_N_NR_LENGTH = 10               # Col 10-19:    Relative-to-absolute intensity normalization field
_N_NT_OFFSET = 21
_N_NT_LENGTH = 8                # Col 22-29:    Relative-to-absolute transition intensity normalization field
_N_BR_OFFSET = 31
_N_BR_LENGTH = 8                # Col 32-39:    Daughter nuclide branching ratio field
_N_BR_UNC_OFFSET = 39
_N_BR_UNC_LENGTH = 2            # Col 40-41:    Daughter nuclide branching ratio uncertainty field

# A/B/D Record Column Mapping
_LEVEL_INTENSITY_OFFSET     = 21
_LEVEL_INTENSITY_LENGTH     = 8     # Col 22-29:    Per-level branch intensity field (IA/IB/IP)
_LEVEL_INTENSITY_UNC_OFFSET = 29
_LEVEL_INTENSITY_UNC_LENGTH = 2     # Col 30-31:    Per-level branch intensity uncertainty field (DIA/DIB/DIP)

# E Record Column Mapping
_E_IE_OFFSET = 31
_E_IE_LENGTH = 8                # Col 32-39:    EC intensity field
_E_IE_UNC_OFFSET = 39
_E_IE_UNC_LENGTH = 2            # Col 40-41:    EC intensity uncertainty field (DIE)
_E_TI_OFFSET = 64
_E_TI_LENGTH = 10               # Col 65-74:    Total EC+beta+ intensity field
_E_TI_UNC_OFFSET = 74
_E_TI_UNC_LENGTH = 2            # Col 75-76:    Total EC+beta+ intensity uncertainty field (DTI)


# ----- Unit Conversion & Physical/Mathematical Constants --------------------
_TIME_UNITS_TO_S = {
    "Y": 365.2425 * 86400,      # Julian year
    "D": 86400.0,               # Day
    "H": 3600.0,                # Hour
    "M": 60.0,                  # Minute
    "S": 1.0,                   # Second
    "MS": 1e-3,                 # Millisecong
    "US": 1e-6,                 # Microsecond
    "NS": 1e-9,                 # Nanosecond
    "PS": 1e-12,                # Picosecond
    "FS": 1e-15,                # Femtosecond
    "AS": 1e-18,                # Attosecond
    "YS": 1e-24                 # Yoctosecond
}

_ENERGY_UNITS_TO_EV  = {"EV": 1.0, "KEV": 1e3, "MEV": 1e6}
_HBAR_EV_S          = 6.582119569e-16                       # Reduced Planck's constant
_LN_2               = math.log(2)


# ----- Uncertainty Qualifiers --------------------
_LIMIT_QUALIFIERS = frozenset({"LT", "GT", "LE", "GE"})     # Uncertainty = 100%
_APPROX_QUALIFIER = "AP"                                    # Uncertainty = 50%
_UNQUANTIFIED_QUALIFIERS = frozenset({"CA", "SY"})          # Uncertainty = undefined


# ----- Regexes --------------------
_FILENAME_PATTERN = re.compile(r'^([A-Z][a-z]?)-(\d+)(m)?$')                    # <SYMBOL>-<MASS>[m]
_DSID_MODE_PATTERN = re.compile(r'\s*(\S+)\s+(\S+)\s+DECAY')                    # <PARENT_ID> <DECAY_MODE> DECAY
_ASYMMETRIC_UNC_PATTERN = re.compile(r'^\+(\d+)-(\d+)$')                        # +<X>-<Y>
_DECIMAL_PRECISION_PATTERN = re.compile(r'^\d+(?:\.(\d+))?(?:[Ee](-?\d+))?$')   # Scientific numeric notation
_HALFLIFE_PATTERN = re.compile(r'^([\d.Ee+\-]+)\s*([A-Z]+)$')                   # <VALUE> <UNIT>


class LARA_ENSDFParser:
    """
    """

    # ----- Private Methods --------------------
    def _read_file(self, filepath: Path) -> str:
        """
        Reads content of an ENSDF file.

        Parameter
        ---------
        - `filepath`: File path of ENSDF file

        Returns
        -------
        Text content of ENSDF file.
        """

        return filepath.read_text(encoding="ascii", errors="replace")


    def _parse_filename(self, filepath: Path) -> tuple[str, int, str]:
        """
        Extracts identifier of nuclide of interest from filename.

        Parameter
        ---------
        - `filepath`: File path of ENSDF file

        Returns
        -------
        Tuple of the form `(symbol, mass no., meta)`
        """

        base    = filepath.stem                     # Extract filename from file path
        m       = _FILENAME_PATTERN.match(base)

        # If file name does not match with expected pattern, identifier cannot be extracted 
        # (invalid file name)
        if not m:
            raise ParseError(f"Filename does not match the <Symbol>-<Mass>[m] pattern: {filepath!r}")
        
        # Extract nuclide identifiers
        symbol      = m.group(1).capitalize()
        mass_num    = int(m.group(2))
        meta        = (m.group(3) or "").lower()

        return (symbol, mass_num, meta)


    def _split_into_datasets(self, text: str) -> list[list[str]]:
        """
        Splits file text into raw dataset blocks.

        Parameters
        ----------
        - `text`: ENSDF file content

        Returns
        -------
        List of raw dataset blocks where each block is an ordered collection of lines.
        """

        # Normalize newline encoding and separate lines
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        blocks: list[list[str]] = []
        cur: list[str]          = []    # Block buffer

        for l in lines:
            # If there is a blank line and the block buffer is not empty, then a dataset block
            # has just finished being extracted
            if l.strip() == "":
                if cur:
                    blocks.append(cur)
                    cur = []
            else:
                cur.append(l)
        
        if cur:
            blocks.append(cur)
        
        return blocks


    def _parse_dataset(self, block: list[str]) -> TransitionData:
        """
        Parses a raw dataset block and extracts its decay transition data.

        Parameter
        ---------
        - `block`: Raw dataset block

        Returns
        -------
        `TransitionData` object containing:
        - Daughter nuclide identifier
        - Decay type
        - Branching ratio + uncertainty
        """

        id_line = block[0]

        # Guard against malformed/truncated identification record
        if len(id_line) < _ID_DSID_OFFSET + _ID_DSID_LENGTH:
            raise ParseError(f"Malformed/truncated Identification record: {id_line!r}")

        # Extract identifier of daughter nuclide of decay transition
        nucid_field = id_line[_ID_NUCID_OFFSET:_ID_NUCID_OFFSET + _ID_NUCID_LENGTH]
        daughter_symbol, daughter_mass = self._parse_nucid(nucid_field)

        # Extract decay type of transition
        dsid    = id_line[_ID_DSID_OFFSET:_ID_DSID_OFFSET + _ID_DSID_LENGTH]
        m       = _DSID_MODE_PATTERN.match(dsid)

        # If pattern does not match, decay type cannot be extracted (malformed data)
        if not m:
            raise ParseError(f"Block does not start with a decay-dataset Identification record: {id_line!r}")
        
        decay_type = m.group(2)

        # Extract branching ratio + uncertainty of transition
        n_record = self._find_n_record(block)
        branch_pct, branch_unc = self._parse_branch_pct(n_record, block)

        return TransitionData(
            symbol      = daughter_symbol,
            meta        = "",
            mass_num    = daughter_mass,
            decay_type  = decay_type,
            branch_pct  = branch_pct,
            branch_unc  = branch_unc
        )


    def _find_p_record(self, block: list[str]) -> str:
        """
        Locates and extracts the P record text from a raw dataset block.
        Raises `ParseError` if no P record is found in the dataset.

        Parameter
        ---------
        - `block`: Raw dataset block

        Returns
        -------
        Raw P record text.
        """

        for line in block:
            # If the lenght of the line is less than the 1-based index of the record type identifier, 
            # then the line cannot have one. So, skip the line
            if len(line) < _RECTYPE_OFFSET + 1:
                continue

            if line[_RECTYPE_OFFSET] == 'P':
                return line
        
        # If this section is reached, then no P record was found in the dataset block
        raise ParseError(f"No P record found in dataset: {block[0]!r}")


    def _find_n_record(self, block: list[str]) -> str:
        """
        Locates and extracts the N record text from a raw dataset block.
        Raises `ParseError` if no N record is found in the dataset.

        Parameter
        ---------
        - `block`: Raw dataset block

        Returns
        -------
        Raw N record text.
        """

        for line in block:
            # If the lenght of the line is less than the 1-based index of the record type identifier, 
            # then the line cannot have one. So, skip the line
            if len(line) < _RECTYPE_OFFSET + 1:
                continue

            if line[_RECTYPE_OFFSET] == 'N':
                return line
        
        # If this section is reached, then no N record was found in the dataset block
        raise ParseError(f"No N record found in dataset: {block[0]!r}")


    def _parse_branch_pct(self, n_record: str, block: list[str]) -> tuple[float | None, float | None]:
        """
        Extracts and parses the branching ratio and its uncertainty for a given raw N record 
        string and raw dataset block. Branching ratio given in [0-100] range.

        Workflow
        --------
        - Extracts the raw strings in the normalization multiplier fields and the branching ratio 
          fields of the N record (NR and NT).
        - If the raw normalization multiplier field strings are empty, it means that the values in 
          the branching ratio fields are just placeholders. So aggregate the branching ratio and its 
          uncertainty using the level intensity entries in the dataset block.
        - Else use the values extracted from the branching ratio fields.

        Parameters
        ----------
        - `n_record`:   Raw N record string
        - `block`:      Raw dataset block

        Returns
        -------
        Tuple of the form `(branching ratio, uncertainty)`
        """
        
        # Extract raw normalization multiplier strings
        nr_str = n_record[_N_NR_OFFSET:_N_NR_OFFSET + _N_NR_LENGTH].strip() \
            if len(n_record) >= _N_NR_OFFSET + _N_NR_LENGTH else ""
        
        nt_str = n_record[_N_NT_OFFSET:_N_NT_OFFSET + _N_NT_LENGTH].strip() \
            if len(n_record) >= _N_NT_OFFSET + _N_NT_LENGTH else ""

        # Extract raw branching ratio and ratio uncertainty strings
        br_str = n_record[_N_BR_OFFSET:_N_BR_OFFSET + _N_BR_LENGTH] \
            if len(n_record) >= _N_BR_OFFSET + _N_BR_LENGTH else ""
        
        bru_str = n_record[_N_BR_UNC_OFFSET:_N_BR_UNC_OFFSET + _N_BR_UNC_LENGTH] \
            if len(n_record) >= _N_BR_UNC_OFFSET + _N_BR_UNC_LENGTH else ""
        
        # Parse branching ratio raw strings into values
        br_val, br_unc = self._parse_val_unc(br_str, bru_str)

        branch_pct = br_val * 100.0 if br_val is not None else None
        branch_unc = br_unc * 100.0 if br_unc is not None else None

        # If both normalization multiplier raw strings are empty, it means that the value in the 
        # branching ratio field is just a placeholder
        # So aggregate the branching ratio from the different level intensity entries in the dataset
        if nr_str == "" and nt_str == "":
            level_sum, level_unc = self._sum_level_intensities(block)

            branch_pct = level_sum
            branch_unc = level_unc

        # If the branching ratio is not defined by this point, its parsing failed
        if branch_pct is None:
            raise ParseError(f"Could not determine branching ratio for dataset: {block[0]!r}")
        
        return (branch_pct, branch_unc)


    def _sum_level_intensities(self, block: list[str]) -> tuple[float | None, float | None]:
        """
        Sums per level branch intensities across raw dataset block's A, B, D and E records to 
        calculate the branching ratio of the dataset.
        Uncertainty is calculated in quadrature.

        Parameter
        ---------
        - `block`: Raw dataset block

        Returns
        -------
        Tuple of the form `(branching ratio, uncertainty)`
        """

        level_sum     = 0.0
        unc_sq_sum    = 0.0
        all_unc_known = True
        found_any     = False

        for line in block:
            # Ignore lines which cannot have a record type identifier or is a continuation of 
            # another record
            if len(line) < _RECTYPE_OFFSET + 1 or line[_CONT_FLAG_OFFSET] != ' ':
                continue

            rec_type = line[_RECTYPE_OFFSET]
            contributions: list[tuple[str, str]]    = []

            # CASE: A/B/D Record
            if rec_type in ('A', 'B', 'D') and len(line) >= _LEVEL_INTENSITY_OFFSET + _LEVEL_INTENSITY_LENGTH:
                val_str = line[_LEVEL_INTENSITY_OFFSET:_LEVEL_INTENSITY_OFFSET + _LEVEL_INTENSITY_LENGTH]
                unc_str = line[_LEVEL_INTENSITY_UNC_OFFSET:_LEVEL_INTENSITY_UNC_OFFSET + _LEVEL_INTENSITY_UNC_LENGTH] \
                    if len(line) >= _LEVEL_INTENSITY_UNC_OFFSET + _LEVEL_INTENSITY_UNC_LENGTH else ""

                contributions.append((val_str, unc_str))

            # CASE: E Record (TI/IE+IB)
            elif rec_type == 'E':
                ti_str = line[_E_TI_OFFSET:_E_TI_OFFSET + _E_TI_LENGTH].strip() \
                    if len(line) >= _E_TI_OFFSET + _E_TI_LENGTH else ""

                # If TI record value exists, then extract its uncertainty
                # Else extract the IE and BI record values along with their uncertainties
                if ti_str:
                    ti_unc_str = line[_E_TI_UNC_OFFSET:_E_TI_UNC_OFFSET + _E_TI_UNC_LENGTH] \
                        if len(line) >= _E_TI_UNC_OFFSET + _E_TI_UNC_LENGTH else ""

                    contributions.append((line[_E_TI_OFFSET:_E_TI_OFFSET + _E_TI_LENGTH], ti_unc_str))
                else:
                    # Extract IE record value + uncertainty
                    ie_str = line[_E_IE_OFFSET:_E_IE_OFFSET + _E_IE_LENGTH] \
                        if len(line) >= _E_IE_OFFSET + _E_IE_LENGTH else ""
                    
                    ie_unc_str = line[_E_IE_UNC_OFFSET:_E_IE_UNC_OFFSET + _E_IE_UNC_LENGTH] \
                        if len(line) >= _E_IE_UNC_OFFSET + _E_IE_UNC_LENGTH else ""

                    contributions.append((ie_str, ie_unc_str))

                    # Extract IB record value + uncertainty
                    ib_str = line[_LEVEL_INTENSITY_OFFSET:_LEVEL_INTENSITY_OFFSET + _LEVEL_INTENSITY_LENGTH] \
                        if len(line) >= _LEVEL_INTENSITY_OFFSET + _LEVEL_INTENSITY_LENGTH else ""
                    
                    ib_unc_str = line[_LEVEL_INTENSITY_UNC_OFFSET:_LEVEL_INTENSITY_UNC_OFFSET + _LEVEL_INTENSITY_UNC_LENGTH] \
                        if len(line) >= _LEVEL_INTENSITY_UNC_OFFSET + _LEVEL_INTENSITY_UNC_LENGTH else ""

                    contributions.append((ib_str, ib_unc_str))

            # Sum up contributions of data in the line to branching ratio
            for val_str, unc_str in contributions:
                if val_str.strip() == "":
                    continue

                value, unc = self._parse_val_unc(val_str, unc_str)

                if value is not None:
                    level_sum += value
                    found_any = True

                    if unc is not None:
                        unc_sq_sum += unc ** 2
                    else:
                        all_unc_known = False

        level_sum = level_sum if found_any else None
        level_unc = math.sqrt(unc_sq_sum) if (found_any and all_unc_known) else None

        return (level_sum, level_unc)


    def _parse_nucid(self, nucid_field: str) -> tuple[str, int]:
        """
        Parses raw daughter nuclide identifier string to extract its element symbol and mass no.

        Parameter
        ---------
        - `nucid_field`: Raw daughter nuclide identifier string

        Returns
        -------
        Tuple of the form `(symbol, mass no.)`
        """

        # Extract daughter nuclide mass no. and symbol
        mass_str    = nucid_field[_NUCID_MASS_OFFSET:_NUCID_MASS_OFFSET + _NUCID_MASS_LENGTH].strip()
        symbol_str  = nucid_field[_NUCID_ELEM_OFFSET:_NUCID_ELEM_OFFSET + _NUCID_ELEM_LENGTH].strip()

        # If the mass string is not a integer and if the symbol string does not exist, then the data 
        # is malformed
        if not mass_str.isdigit() or not symbol_str:
            raise ParseError(f"Malformed NUCID field: {nucid_field!r}")
        
        return (symbol_str.capitalize(), int(mass_str))


    def _parse_val_unc(self, val_str: str, unc_str: str) -> tuple[float | None, float | None]:
        """
        Parses a `(raw value string, raw uncertainty string)` pair to their actual values.

        Parameters
        ----------
        - `val_str`: Raw value string
        - `unc_str`: Raw uncertainty string of value

        Returns
        -------
        Tuple of the form `(value, uncertainty)`
        """

        val_str = val_str.strip()
        unc_str = unc_str.strip()

        # If raw value string is empty, then the value is not defined. Also parsing uncertainty is 
        # redundant as it is also by definition undefined.
        if val_str == "":
            return (None, None)
        
        # Raw value string should represent a floating point number
        try:
            value = float(val_str)
        except ValueError:
            raise ParseError(f"Value field is not blank and not numeric: {val_str!r}")

        # If raw uncertainty string is empty, there is nothing to parse
        if unc_str == "":
            return (value, None)

        asym = _ASYMMETRIC_UNC_PATTERN.match(unc_str)

        # CASE: Uncertainty is in asymmetric form
        # (range averaged to produce a single uncertainty value)
        if asym:
            u1, u2 = int(asym.group(1)), int(asym.group(2))
            unc_digits = (u1 + u2) / 2.0

            return (value, unc_digits * self._unc_scale(val_str))

        # CASE: Uncertainty is a floating point value
        if unc_str.isdigit():
            return (value, float(unc_str) * self._unc_scale(val_str))

        # CASE: Uncertainty is in the form of a qualifier
        qualifier = unc_str.upper()

        if qualifier in _LIMIT_QUALIFIERS:
            return (value, abs(value))
        if qualifier == _APPROX_QUALIFIER:
            return (value, abs(value) * 0.5)
        if qualifier in _UNQUANTIFIED_QUALIFIERS:
            return (value, None)

        # If this point is reached, then parsing the uncertainty value has failed
        raise ParseError(f"Unrecognized uncertainty field: {unc_str!r} (value was {val_str!r})")


    def _unc_scale(self, val_str: str) -> float:
        """
        Calculates scale factor of uncertainty based on last significant digit of value.

        Parameter
        ---------
        - `val_str`: Raw string of value whose uncertainty scale is to be calculated

        Returns
        -------
        Scale of uncertainty value.
        """
        
        m = _DECIMAL_PRECISION_PATTERN.match(val_str)

        # If pattern does not match, uncertainty value scale cannot be determined
        if not m:
            raise ParseError(f"Could not determine decimal precision of value: {val_str!r}")
        
        decimals = len(m.group(1)) if m.group(1) else 0
        exponent = int(m.group(2)) if m.group(2) else 0

        return 10 ** (exponent - decimals)


    def _parse_halflife(self, t_str: str, unc_str: str) -> tuple[float | None, float | None, bool]:
        """
        Parses raw half-life string and half-life uncertainty string and converts them to seconds.

        Parameters
        ----------
        - `t_str`:      Half-life raw string
        - `unc_str`:    Half-life uncertainty raw string

        Returns
        -------
        Tuple of the form `(half-life, uncertainty)`
        """

        t_str = t_str.strip()

        # If half-life raw string is empty, it cannot be parsed
        if t_str == "":
            raise ParseError("Half-life raw string is empty.")
        
        # If nuclide is stable, it does not have a half-life
        if t_str.upper() == "STABLE":
            return (None, None, True)

        m = _HALFLIFE_PATTERN.match(t_str)

        # If pattern does not match, half-life value and its uncertainty cannot be parsed
        if not m:
            raise ParseError(f"Unparseable half-life field: {t_str!r}")
        
        # Extract half-life value and its unit of measurement
        num_str, unit = m.group(1), m.group(2).upper()

        # CASE 1: Unit in terms of energy
        if unit in _ENERGY_UNITS_TO_EV:
            val, unc    = self._parse_val_unc(num_str, unc_str)
            gamma_ev    = val * _ENERGY_UNITS_TO_EV[unit]
            half_life_s = _LN_2 * _HBAR_EV_S / gamma_ev
            unc_s       = half_life_s * (unc / val) if (unc is not None and val) else None

            return (half_life_s, unc_s, False)

        # CASE 2: Unit in terms of time
        # If unit is not in terms of energy or time then the half-life value cannot be parsed
        if unit not in _TIME_UNITS_TO_S:
            raise ParseError(f"Unrecognized half-life unit {unit!r} in {t_str!r}")
        
        # Parse values and convert unit to seconds
        val, unc    = self._parse_val_unc(num_str, unc_str)
        factor      = _TIME_UNITS_TO_S[unit]
        half_life_s = val * factor
        unc_s       = unc * factor if unc is not None else None

        return (half_life_s, unc_s, False)


    def _halflife_to_decay_const(self, half_life_s: float | None, 
                                 unc_s: float | None) -> tuple[float | None, float | None]:
        """
        Calculates decay constant (in s^-1) and its uncertainty from nuclide half-life and its 
        uncertainty.

        Parameters
        ----------
        - `Half_life_s`:    Half-life of nuclide (in s)
        - `unc_s`:          Uncertainty of half-life (in s)

        Returns
        -------
        Tuple of the form `(decay const, uncertainty)`
        """

        if half_life_s is None or half_life_s == 0:
            return None, None
        
        lam     = _LN_2 / half_life_s
        lam_unc = lam * (unc_s / half_life_s) if unc_s is not None else None

        return (lam, lam_unc)


    # ----- Public Methods --------------------
    def parse(self, filepath: Path) -> NuclideData:
        """
        Parses a single ENSDF file obtained from LARAWeb into a `NuclideData` object.

        Parameter
        ---------
        - `filepath`: File path of ENSDF file

        Returns
        -------
        A `NuclideData` object containing:
        - Nuclide identifier
        - Nuclide half-life + uncertainty
        - If the nucldie is stable
        - Its transitions which contain:
            - Daughter nuclide identifier
            - Decay type
            - Branching ratio + uncertainty
        """
        
        symbol, mass_num, meta = self._parse_filename(filepath)
        text = self._read_file(filepath)                        # Raw ENSDF file text

        blocks = self._split_into_datasets(text)

        # If the ENSDF file cannot be separated into datasef blocks, it cannot be parsed 
        # (malformed data)
        if not blocks:
            raise ParseError(f"No decay datasets found in {filepath!r}")

        # Extract if parent nuclide is stable along with its half-life + uncertainty
        p_record = self._find_p_record(blocks[0])
        t_str   = p_record[_P_HALFLIFE_OFFSET:_P_HALFLIFE_OFFSET + _P_HALFLIFE_LENGTH]
        unc_str = p_record[_P_HALFLIFE_UNC_OFFSET:_P_HALFLIFE_UNC_OFFSET + _P_HALFLIFE_UNC_LENGTH]
        half_life_s, hl_unc_s, is_stable = self._parse_halflife(t_str, unc_str)

        # Convert half-life + uncertainty -> decay constant + uncertainty
        decay_const, decay_unc = self._halflife_to_decay_const(half_life_s, hl_unc_s)

        # Extract transitions from raw dataset blocks
        transitions = [self._parse_dataset(b) for b in blocks]

        return NuclideData(
            symbol      = symbol,
            meta        = meta,
            mass_num    = mass_num,
            stable      = is_stable,
            half_life   = half_life_s,
            decay_const = decay_const,
            decay_unc   = decay_unc,
            transitions = transitions
        )