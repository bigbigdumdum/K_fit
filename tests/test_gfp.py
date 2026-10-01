# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""CLAUDE.md test case: fit GFP 6L26 onto GFP 1EMA by the chromophore carbons.

Both entries have the chromophore as residue CRO 66 of chain A. 1EMA has 15
chromophore carbons. 6L26 has the same 15 names, but CB1 and CG1 have
altlocs A (occupancy 0.85) and B (0.15); the higher-occupancy altloc is used.

The structures are downloaded from RCSB, so the test is skipped when offline.
"""

import csv

import numpy as np
import pytest

from K_fit.fetch import FetchError, fetch_pdb_entry
from K_fit.parser import load_structure
from K_fit.pipeline import FitJob, run_superposition

CHROMOPHORE = ("A", 66, " ", "CRO")


def chromophore_carbons(structure) -> dict:
    """Return {atom name: AtomRecord} for CRO 66 carbons, highest-occupancy altloc."""
    best = {}
    for atom in structure.atoms_in_residue(1, CHROMOPHORE):
        if atom.element != "C":
            continue
        if atom.atom_name not in best or atom.occupancy > best[atom.atom_name].occupancy:
            best[atom.atom_name] = atom
    return best


@pytest.fixture(scope="module")
def download_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("gfp"))


def load_entry(pdb_id, file_format, download_dir):
    """Download and load one entry; skip the test if the download fails."""
    try:
        path = fetch_pdb_entry(pdb_id, download_dir, file_format)
    except FetchError as err:
        pytest.skip(f"cannot download {pdb_id}: {err}")
    return load_structure(path, source="pdb_id")


@pytest.mark.parametrize("file_format", ["pdb", "cif"])
def test_gfp_chromophore_fit(file_format, download_dir, tmp_path):
    reference = load_entry("1EMA", file_format, download_dir)
    target = load_entry("6L26", file_format, download_dir)
    ref_c, tgt_c = chromophore_carbons(reference), chromophore_carbons(target)
    assert len(ref_c) == 15 and set(ref_c) == set(tgt_c)
    names = list(ref_c)
    job = FitJob(target, [ref_c[n].matomid for n in names], [tgt_c[n].matomid for n in names])

    result = run_superposition(reference, [job], str(tmp_path))
    fit = result.fits[0]
    assert result.checks[0].ok and not result.checks[0].warnings
    assert fit.passed and fit.n_atoms == 15 and fit.so == 100.0
    assert fit.rmsd == pytest.approx(0.124, abs=0.002)

    # Independent Kabsch (SVD) RMSD on the same pairs.
    u = reference.coords(job.reference_ids)
    v = target.coords(job.target_ids)
    uc, vc = u - u.mean(0), v - v.mean(0)
    left, _s, right_t = np.linalg.svd(vc.T @ uc)
    sign = np.sign(np.linalg.det(right_t.T @ left.T))
    rotation = right_t.T @ np.diag([1, 1, sign]) @ left.T
    kabsch_rmsd = np.sqrt(((vc @ rotation.T - uc) ** 2).sum(axis=1).mean())
    assert fit.rmsd == pytest.approx(kabsch_rmsd, abs=1e-4)

    # R and t in transforms.csv reproduce the written structure file.
    with open(result.transforms_path) as handle:
        row = next(csv.DictReader(handle))
    r = np.array([[float(row[f"R{i}{j}"]) for j in (1, 2, 3)] for i in (1, 2, 3)])
    t = np.array([float(row[f"t{i}"]) for i in (1, 2, 3)])
    fitted = load_structure(result.output_structures[target.name])
    ids = list(target.atoms)
    assert np.abs(target.coords(ids) @ r.T + t - fitted.coords(ids)).max() < 2e-3
