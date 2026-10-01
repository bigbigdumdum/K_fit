# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Pack output files into a zip archive (standard library only)."""

from __future__ import annotations

import os
import zipfile


def zip_outputs(paths: list, zip_path: str) -> str:
    """Write ``paths`` into ``zip_path`` (flat, by base name) and return ``zip_path``.

    Missing or None entries are skipped, so the output lists can be passed
    straight in.
    """
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path and os.path.isfile(path):
                archive.write(path, arcname=os.path.basename(path))
    return zip_path
