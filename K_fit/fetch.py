# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Download structure files from the RCSB PDB by PDB ID.

Uses only the Python standard library (urllib). To use a different mirror
(for example PDBe), change ``DOWNLOAD_URL``.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request

from .errors import KFitError

# {pdb_id} is the upper-case 4-character ID, {ext} is "cif" or "pdb".
DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.{ext}"

# Classic 4-character PDB IDs: a digit followed by three letters/digits.
PDB_ID_PATTERN = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


class FetchError(KFitError):
    """Raised when a PDB entry cannot be downloaded."""


def fetch_pdb_entry(pdb_id: str, out_dir: str, file_format: str = "cif",
                    timeout: float = 60.0) -> str:
    """Download one PDB entry and return the path of the saved file.

    Parameters
    ----------
    pdb_id : 4-character PDB ID, e.g. "1UBQ" (case does not matter).
    out_dir : directory to save into (created if missing).
    file_format : "cif" (default; available for every entry) or "pdb"
        (not available for very large entries).
    timeout : network timeout in seconds.

    The file is saved as ``<ID>.<ext>``, e.g. ``1UBQ.cif``.
    """
    pdb_id = pdb_id.strip()
    if not PDB_ID_PATTERN.match(pdb_id):
        raise FetchError(f"'{pdb_id}' is not a valid 4-character PDB ID")
    if file_format not in ("cif", "pdb"):
        raise FetchError(f"unsupported download format: {file_format}")

    pdb_id = pdb_id.upper()
    url = DOWNLOAD_URL.format(pdb_id=pdb_id, ext=file_format)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pdb_id}.{file_format}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as err:
        raise FetchError(f"{pdb_id}: download failed (HTTP {err.code}) from {url}") from err
    except urllib.error.URLError as err:
        raise FetchError(f"{pdb_id}: download failed ({err.reason}) from {url}") from err

    with open(out_path, "wb") as handle:
        handle.write(data)
    return out_path
