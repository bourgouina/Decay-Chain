from __future__ import annotations

import re
from pathlib import Path

from shared import NuclideData, TransitionData, ParseError


# ----- LARA Text File Field Key Constants --------------------
_FIELD_SEP = ";"

_KEY_NUCLIDE            = "Nuclide"
_KEY_DAUGHTERS          = "Daughter(s)"
_KEY_HALFLIFE_SECONDS   = "Half-life (s)"
_KEY_DECAY_CONST        = "Decay constant (1/s)"

_EMISSIONS_MARKER_PREFIX    = "Emissions"
_NO_EMISSIONS_MARKER        = "No emissions"

# Each daughter nuclide record has 3 fields: decay mode, daughter nuclide identifier, branching 
# ratio
_DAUGHTER_GROUP_SIZE = 3


# ----- Regexes --------------------
_NUCLIDE_ID_PATTERN = re.compile(r'^([A-Za-z]+)-(\d+)(m)?$')    # <SYMBOL>-<MASS>[m]
_MODE_TOKEN_PATTERN = re.compile(r'^\((.+)\)$')                 # (<DECAY_MODE>)


class LARA_TxtParser:
    """
    """

    # ----- Private Methods --------------------
    def _read_file(self, filepath: Path) -> str:
        """
        Reads content of a LARA text file.

        Parameter
        ---------
        - `filepath`: File path of LARA text file

        Returns
        -------
        Text content of LARA text file.
        """

        return filepath.read_text(encoding="ascii", errors="replace")


    def _split_into_lines(self, text: str) -> list[str]:
        """
        Normalizes line endings and strips blank lines from raw file text.

        Parameter
        ---------
        - `text`: LARA text file content

        Returns
        -------
        List of non-blank lines.
        """

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        return [l.strip() for l in lines if l.strip() != ""]


    def _split_fields(self, line: str) -> list[str]:
        """
        Splits a raw semicolon-delimited record line into its raw fields.

        Parameter
        ---------
        - `line`: Raw record line

        Returns
        -------
        List of raw fields.
        """

        fields = [f.strip() for f in line.split(_FIELD_SEP)]

        # If an empty string after the last semi-colon is extracted as a field, remove it
        if fields and fields[-1] == "":
            fields.pop()

        return fields


    def _parse_nuclide_id(self, raw: str) -> tuple[str, int, str]:
        """
        Parses a raw nuclide identifier string into its identifying components.

        Parameter
        ---------
        - `raw`: Raw nuclide identifier string

        Returns
        -------
        Tuple of the form `(symbol, mass no., meta)`
        """

        m = _NUCLIDE_ID_PATTERN.match(raw.strip())

        # If pattern does not match, nuclide identifier cannot be extracted (malformed data)
        if not m:
            raise ParseError(f"Malformed nuclide identifier: {raw!r}")

        symbol      = m.group(1).capitalize()
        mass_num    = int(m.group(2))
        meta        = (m.group(3) or "").lower()

        return (symbol, mass_num, meta)


    def _parse_daughters_record(self, fields: list[str]) -> list[TransitionData]:
        """
        Parses daughter nuclide data into a list of `TransitionData` objects.
        Raises `ParseError` if field data does not contain detectable daughter nuclide data.

        Parameter
        ---------
        - `fields`: Fields of the raw line containing daughter nuclide data (including key field)

        Returns
        -------
        List of `TransitionData`objects (one per decay branch) containing:
        - Daughter nuclide identifier
        - Decay type
        - Branching ratio + uncertainty
        """

        # If line fields does not contain daughter data, it cannot be parsed
        if fields[0] != _KEY_DAUGHTERS:
            raise ParseError(f'Invalid field type. Expected key to be "{_KEY_DAUGHTERS}".')

        branch_fields = fields[1:]

        # If the branch field count is not a multiple of the group size, the record is malformed
        if len(branch_fields) == 0 or len(branch_fields) % _DAUGHTER_GROUP_SIZE != 0:
            raise ParseError(f"Malformed Daughter(s) record: {fields!r}")

        transitions: list[TransitionData] = []

        # Iterate through daughter data groups
        for i in range(0, len(branch_fields), _DAUGHTER_GROUP_SIZE):
            mode_str, daughter_str, pct_str = branch_fields[i:i + _DAUGHTER_GROUP_SIZE]

            m = _MODE_TOKEN_PATTERN.match(mode_str)

            # If pattern does not match, decay type cannot be extracted (malformed data)
            if not m:
                raise ParseError(f"Malformed decay-mode token: {mode_str!r}")

            # Extract decay type, daughter nuclide identifier and branching ratio + uncertainty
            decay_type = m.group(1) if m.group(1) != "alpha" else "A"
            symbol, mass_num, meta = self._parse_nuclide_id(daughter_str)
            branch_pct = self._parse_float(pct_str, context=f"branch_pct ({daughter_str})")

            transitions.append(TransitionData(
                symbol      = symbol,
                meta        = meta,
                mass_num    = mass_num,
                decay_type  = decay_type,
                branch_pct  = branch_pct,
                branch_unc  = None
            ))
        
        TransitionData.order(transitions)

        return transitions


    def _parse_value_unc_record(self, fields: list[str], context: str) -> tuple[float, float | None]:
        """
        Parses the value and uncertainty of a "`<KEY>` ; `<VALUE>` ; `<UNCERTAINTY>`" record.

        Parameters
        ----------
        - `fields`:   Fields of the record
        - `context`:  Description of the record being parsed, for error messages

        Returns
        -------
        Tuple of the form `(value, uncertainty)`.
        """

        # If the value field is missing, the record is malformed
        if len(fields) < 2:
            raise ParseError(f"Malformed {context} record: {fields!r}")

        # Parse value
        value   = self._parse_float(fields[1], context=context)
        
        # Parse uncertainty
        unc_str = fields[2] if len(fields) >= 3 else ""
        unc = None

        try: 
            unc = self._parse_float(unc_str, context=f"{context}_unc")
        except ParseError:
            unc = None

        return (value, unc)


    def _parse_float(self, raw: str, context: str) -> float:
        """
        Parses a numeric field.

        Parameters
        ----------
        - `raw`:      Raw numeric field string
        - `context`:  Description of the field being parsed, for error messages

        Returns
        -------
        Parsed value.
        """

        raw = raw.strip()

        # If raw value string is empty, it cannot be parsed
        if raw == "":
            raise ParseError(f"Missing required numeric value for {context}")

        try:
            return float(raw)
        except ValueError:
            raise ParseError(f"Malformed numeric value {raw!r} for {context}")


    # ----- Public Methods --------------------
    def parse(self, filepath: Path) -> NuclideData:
        """
        Parses a single LARA text file obtained from LARAWeb into a `NuclideData` object.
        Raises `ParseError` for malformed data.

        Parameter
        ---------
        - `filepath`: File path of LARA text file

        Returns
        -------
        A `NuclideData` object containing:
        - Nuclide identifier
        - Nuclide half-life (in s)
        - Nuclide decay constant + uncertainty (in 1/s)
        - If the nuclide is stable (always `False`)
        - Its transitions which contain:
            - Daughter nuclide identifier
            - Decay type
            - Branching ratio + uncertainty (uncertainty always `None` as not provided)
        """

        text  = self._read_file(filepath)
        lines = self._split_into_lines(text)

        symbol: str | None       = None
        mass_num: int | None     = None
        meta: str | None         = None
        transitions: list[TransitionData] | None = None
        half_life_s: float | None = None
        decay_const: float | None = None
        decay_unc: float | None   = None
        saw_emissions_marker      = False

        for line in lines:
            # If this is the Emissions/No-emissions marker, the header block has ended.
            if line.startswith(_EMISSIONS_MARKER_PREFIX) or line == _NO_EMISSIONS_MARKER:
                saw_emissions_marker = True
                break

            fields  = self._split_fields(line)
            key     = fields[0]

            # CASE: Nuclide record
            if key == _KEY_NUCLIDE:
                if len(fields) < 2:
                    raise ParseError(f"Malformed Nuclide record: {line!r}")
                
                symbol, mass_num, meta = self._parse_nuclide_id(fields[1])

            # CASE: Daughter(s) record
            elif key == _KEY_DAUGHTERS:
                transitions = self._parse_daughters_record(fields)

            # CASE: Half-life (s) record
            elif key == _KEY_HALFLIFE_SECONDS:
                half_life_s, _ = self._parse_value_unc_record(fields, context="half_life")

            # CASE: Decay constant (1/s) record -- given directly, not derived from half-life
            elif key == _KEY_DECAY_CONST:
                decay_const, decay_unc = self._parse_value_unc_record(fields, context="decay_const")

        # If the file has no Emissions/No-emissions marker, it looks truncated (malformed data)
        if not saw_emissions_marker:
            raise ParseError(f'No "Emissions"/"No-emissions" marker found (malformed/truncated data)')

        # If any required record was not found, the file cannot be parsed
        if symbol is None:
            raise ParseError(f"No {_KEY_NUCLIDE!r} record found.")
        if transitions is None:
            raise ParseError(f"No {_KEY_DAUGHTERS!r} record found.")
        if half_life_s is None:
            raise ParseError(f"No {_KEY_HALFLIFE_SECONDS!r} record found.")
        if decay_const is None:
            raise ParseError(f"No {_KEY_DECAY_CONST!r} record found.")

        return NuclideData(
            symbol      = symbol,
            meta        = meta,
            mass_num    = mass_num,
            stable      = False,
            half_life   = half_life_s,
            decay_const = decay_const,
            decay_unc   = decay_unc,
            transitions = transitions
        )