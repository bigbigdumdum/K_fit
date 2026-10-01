Copyright (C) 2026 Mukundan S

# K_fit

A Python library that superimposes protein structures (PDB or mmCIF) onto a
reference structure by Kearsley fitting of atoms you choose, and writes the
fitted structures with a report.

For a point-and-click interface, use the
[K_fit-colab](https://github.com/bigbigdumdum/K_fit-colab) notebook.

## Installation

```bash
pip install git+https://github.com/bigbigdumdum/K_fit.git
```

Requires Python 3.10+, Biopython, NumPy and
[kearsley-numba](https://github.com/bigbigdumdum/kearsley_numbajit).

## Example

```python
from K_fit.parser import load_structure
from K_fit.fetch import fetch_pdb_entry
from K_fit.pipeline import FitJob, run_superposition

reference = load_structure("ref.pdb")
target = load_structure(fetch_pdb_entry("1ABC", "inputs"), source="pdb_id")

job = FitJob(target=target,
             reference_ids=["1-10", "1-18", "1-25"],   # paired by position
             target_ids=["1-12", "1-20", "1-27"])
result = run_superposition(reference, [job], out_dir="out")
print(result.summary_text())
```

## Atom IDs

Atoms are identified by a **matomid**, `<model number>-<atom id>`, for
example `1-245`. The model number is the one in the file (1 if there is
none); the atom id is the PDB serial number or the mmCIF `_atom_site.id`.
Alternate locations are separate atoms.

The same entry can have different atom ids in its PDB and mmCIF files, so
choose atoms from the file you actually load.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `so_cutoff` (SOdc) | 1.5 Å | Every fitted atom must lie within this distance of its reference atom. |
| `rmsd_cutoff` (RMSDc) | 1.5 Å | Maximum RMSD for a fit to be reported as FIT. |
| `fit_all_models` | `False` | Fit every model of a multi-model target separately, instead of only the selected model. |

## Output files

| File | Content |
|---|---|
| `<name>_fit.pdb` / `.cif` | Fitted structure in the input format, coordinates only: header records (cell, symmetry, assemblies, ...) are removed because they no longer match. Not written if no fit passed; with `fit_all_models`, NO FIT models are left out |
| `report.txt` | Inputs with SHA-256 checksums, cutoffs, and for each fit the atom count, RMSD, structure overlap, FIT / NO FIT and the transformation |
| `<name>__pairs.csv` | The exact atom pairs used |

`<name>` is the target name without its extension. If two targets share it
(e.g. `x.pdb` and `x.cif`), the extension is kept: `x_pdb`, `x_cif`.
| `transforms.csv` | Rotation R and translation t for each fit |
| `K_fit_results.zip` | All of the above |

Transformations are given as `x' = R·x + t`, so they can be used directly in
PyMOL or ChimeraX.

At least 3 atom pairs, not all on one straight line, are required.
All K_fit errors derive from `K_fit.errors.KFitError`.

## Known limitations

- Biopython cannot read some mmCIF files with microheterogeneity (e.g.
  1EJG); the PDB-format file of the same entry usually works.
- PDB format cannot hold coordinates outside −999.999 to 9999.999 Å; use
  mmCIF for such cases.

## License

MIT License. See [LICENSE](LICENSE).
