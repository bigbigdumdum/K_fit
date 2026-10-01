# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""K_fit: superimpose protein structures by Kearsley fitting of selected atoms.

Modules
-------
parser    : load PDB / mmCIF files into Structure objects (Biopython based)
fetch     : download entries from the RCSB PDB by PDB ID
checker   : validate atom selections and read the pairs CSV
fitter    : Kearsley fit (kearsley-numba) and coordinate transformation
writer    : write moved structures, the text report and the CSV files
archive   : zip all output files
pipeline  : run a complete superposition job (used by the Colab notebook)

Every atom is addressed by its ``matomid`` = f"{model_number}-{atom_id}".
"""

__version__ = "0.1.0"
