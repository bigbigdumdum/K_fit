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
ANISOU). `tests/test_gfp.py` is the test case defined in CLAUDE.md; it
downloads its entries and is skipped when offline.

## Behaviour details

- **matomid:** Biopython reports PDB files without `MODEL` records as model 0;
  `parser._model_number` makes that model 1, matching mmCIF. In PDB files each
  `TER` record consumes a serial number, so ids after a `TER` differ from the
  mmCIF file of the same entry.
- **Fit all models:** atoms are matched across models by (chain, residue
  number, insertion code, residue name, atom name, altloc), because PDB files
  restart atom serials in every model. A model missing a selected atom is NO FIT.
- **Without fit all models,** only the selected model is moved; other models
  are written unchanged. **With fit all models,** the fitted file holds only
  the models whose fit passed; NO FIT models are left out and listed in the report.
- **Pass criteria:** `RMSD <= rmsd_cutoff` and SO == 100 %. SO comes from
  kearsley-numba and counts distances strictly below `so_cutoff`. With
  RMSDc >= SOdc the RMSD test can never fail on its own (every distance is
  below SOdc, so the RMSD is too); it decides only when RMSDc is set lower.
  Both are kept because RMSDc is the standard cutoff in the field.
- **NO FIT:** no structure file is written unless at least one fit of that
  target passed; the report, pairs CSV and transforms row are always written.
- **Checks:** errors for empty selections, count mismatch, unknown or
  duplicate matomids, mixed models, fewer than 3 pairs, and atoms on one
  straight line (`checker.COLLINEAR_TOLERANCE`); warnings for atom name or
  element mismatches.
- **Errors:** every K_fit error class derives from `errors.KFitError`, so a
  caller can report all user mistakes with one `except`.
- **Output names:** `<stem>_fit<ext>` and `<stem>__pairs.csv`, where stem is
  the target name without its extension. Targets sharing a stem keep the
  extension in it (`x_pdb`, `x_cif`; `pipeline.output_stems`).

## Transformation convention

kearsley-numba returns `(rotation, translation)` with
`x' = rotation · (x − translation)`. The fitter keeps these values unchanged
and moves coordinates with `kearsley_numba.transform`. Only the writer
converts them to the standard form for the output files
(`writer.to_standard_transform`): `R = rotation`, `t = −rotation · translation`.

## Writing structure files

Fitted files hold coordinates only. Header records (cell, symmetry,
assemblies, secondary structure, links, ...) are removed, because they no
longer match the moved coordinates. Both writers read the input text kept in
`Structure.raw_text`, so the input file may be deleted after loading.

- **PDB:** only `MODEL`, `ATOM`, `HETATM`, `ANISOU`, `TER`, `ENDMDL` and `END`
  lines are kept (`writer.PDB_KEPT_RECORDS`). Columns 31–54 (x, y, z) of moved
  `ATOM`/`HETATM` records are replaced and `ANISOU` tensors rotated (R U Rᵀ);
  all other columns are unchanged.
- **mmCIF:** only the block name, `_entry.id`, `_atom_site` and
  `_atom_site_anisotrop` are kept (`writer.CIF_KEPT_PREFIXES`).
  `_atom_site.Cartn_x/y/z` (and U or B tensors) are replaced through
  Biopython's `MMCIF2Dict` and written with `MMCIFIO`; spacing and quoting
  may change.
