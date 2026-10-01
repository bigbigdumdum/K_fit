Copyright (C) 2026 Mukundan S

# K_fit: implementation plan

Engine repository (repo 1). Requirements come from `CLAUDE.md` and the
decisions log `DESIGN_NOTES.md` in the parent project folder.

## Rules

- No Colab or widget code in this repo.
- Dependencies: `biopython`, `numpy`, `kearsley-numba`. Anything else needs a reason.
- Names are standard and describe what the code does.
- Every function and class has a docstring explaining what it does, its
  inputs and outputs, and what to change when modifying it.
- Nothing is committed without the owner's permission.
- License: MIT. Each source file starts with
  `# Copyright (C) 2026 Mukundan S` and `# SPDX-License-Identifier: MIT`.

## Package layout

```
K_fit/
├── pyproject.toml
├── README.md, PLAN.md, LICENSE
├── K_fit/
│   ├── __init__.py
│   ├── parser.py      # load PDB/mmCIF, Structure / AtomRecord, unique names
│   ├── fetch.py       # download a PDB ID from RCSB (urllib only)
│   ├── checker.py     # validate atom pairs, read pairs CSV
│   ├── fitter.py      # Kearsley fit, apply transformation, cutoffs
│   ├── writer.py      # structure files, report, CSVs
│   ├── archive.py     # zip all outputs
│   └── pipeline.py    # run a full job (check, fit, write, zip)
└── tests/
    ├── data/          # 1UBQ (pdb+cif), 1L2Y 3 NMR models, 1EJG residues 1-8
    └── test_*.py
```

## Milestones

| | Milestone | Status |
|---|---|---|
| M1 | Parser and fetch | done |
| M2 | Checker and pairs CSV | done |
| M3 | Fitter (one model, or every model with fit_all_models) | done |
| M4 | Writer (PDB text swap + ANISOU rotation, mmCIF via MMCIF2Dict), report, CSVs, zip | done |
| M5 | Packaging (`pyproject.toml`, MIT) and pipeline | done |

38 tests pass (`pytest`; the 2 tests of the CLAUDE.md test case download their entries and skip when offline), with Biopython 1.88, numpy 2.5, numba 0.67 and
kearsley-numba 0.1.0 on Python 3.13.

## Behaviour decided during implementation

- **NO FIT:** no structure file is written for a target unless at least one
  of its fits passed. The pairs CSV, the `transforms.csv` rows and the report
  entry (with RMSD, SO and the reason) are always written.
- **Fit all models:** each model of the target is fitted separately. Atoms
  are matched across models by (chain, residue, insertion code, residue name,
  atom name, altloc), because PDB files restart atom serials in every model.
  A model lacking a selected atom is reported as NO FIT.
- **Without fit all models:** only the selected model is moved; other models
  are written unchanged. With fit all models, NO FIT models are left out of
  the fitted file.
- **Fitted files** hold coordinates only; header records are removed.
- **Same file twice:** `Structure.copy()` and `pipeline.self_fit_job()` make a
  copy named like a duplicate input (`<stem>_<n><ext>`).
- **Errors** (not warnings) for fewer than 3 pairs and for atoms on one straight line.
- A pipeline run with selection errors raises `PipelineError` and writes nothing.

## Open items

1. Package name: `K_fit` works, but Python style prefers lowercase `k_fit`.
