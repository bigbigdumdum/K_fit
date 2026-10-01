Copyright (C) 2026 Mukundan S

# K_fit: development notes

Notes for people changing the engine. User documentation is in
[README.md](README.md); the implementation plan and status are in
[PLAN.md](PLAN.md).

## Modules

| Module | Purpose |
|---|---|
| `parser` | `load_structure()`, `Structure`, `AtomRecord`, `make_unique_name()` |
| `fetch` | `fetch_pdb_entry()`: download from RCSB (standard library only) |
| `checker` | `check_atom_pairs()`, `read_pairs_csv()`, `parse_matomid_list()` |
| `fitter` | `fit_pair()`, `apply_transform()`, `FitResult` |
| `writer` | structure files, `report.txt`, pairs and transforms CSVs |
| `archive` | `zip_outputs()` |
| `pipeline` | `run_superposition()`, `FitJob`, `self_fit_job()` |

The modules can be used one by one (for example `checker.check_atom_pairs`
then `fitter.fit_pair` then `writer.write_structure`); `pipeline` chains them.
This repository must stay free of notebook or Colab code.

## Setup and tests

```bash
pip install -e ".[test]"
pytest -q
```

Test data in `tests/data` are excerpts of RCSB entries: 1UBQ (PDB and mmCIF),
the first 3 NMR models of 1L2Y, and residues 1–8 of 1EJG (altlocs and
ANISOU). `tests/test_gfp.py` is the CLAUDE.md test case (6L26 fitted onto
1EMA by the 15 chromophore carbons, RMSD 0.124 Å); it downloads both entries
and is skipped when offline.

## Behaviour details

- **matomid:** Biopython reports PDB files without `MODEL` records as model 0;
  `parser._model_number` makes that model 1, matching mmCIF. In PDB files each
  `TER` record consumes a serial number, so ids after a `TER` differ from the
  mmCIF file of the same entry.
- **Fit all models:** atoms are matched across models by (chain, residue
  number, insertion code, residue name, atom name, altloc), because PDB files
  restart atom serials in every model. A model missing a selected atom is NO FIT.
- **Without fit all models,** only the selected model is moved; other models
  are written unchanged.
- **Pass criteria:** `RMSD <= rmsd_cutoff` and SO == 100 %. SO comes from
  kearsley-numba and counts distances strictly below `so_cutoff`.
- **NO FIT:** no structure file is written unless at least one fit of that
  target passed; the report, pairs CSV and transforms row are always written.
- **Checks:** errors for empty selections, count mismatch, unknown or
  duplicate matomids, and mixed models; warnings for atom name or element
  mismatches and fewer than 3 pairs.

## Transformation convention

kearsley-numba returns `(rotation, translation)` with
`x' = rotation · (x − translation)`. The fitter keeps these values unchanged
and moves coordinates with `kearsley_numba.transform`. Only the writer
converts them to the standard form for the output files
(`writer.to_standard_transform`): `R = rotation`, `t = −rotation · translation`.

## Writing structure files

- **PDB:** the input text is copied; only columns 31–54 (x, y, z) of moved
  `ATOM`/`HETATM` records are replaced, and `ANISOU` tensors are rotated
  (R U Rᵀ). All other lines are byte-identical.
- **mmCIF:** `_atom_site.Cartn_x/y/z` (and `_atom_site_anisotrop` U or B
  tensors) are replaced through Biopython's `MMCIF2Dict` and written with
  `MMCIFIO`. All categories are kept, but spacing and quoting may change.
