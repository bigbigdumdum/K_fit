# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Tests for K_fit.fitter, K_fit.writer, K_fit.archive and K_fit.pipeline."""

import csv
import os
import zipfile

import numpy as np
import pytest
from Bio.PDB import PDBParser

from K_fit.fitter import apply_transform, fit_pair
from K_fit.parser import load_structure
from K_fit.pipeline import FitJob, PipelineError, run_superposition, self_fit_job
from K_fit.writer import to_standard_transform, write_structure

from conftest import data_path


def moved_copy(path, model_transforms, out_path, source_name=None):
    """Write a moved copy of a structure file and load it."""
    structure = load_structure(path)
    write_structure(structure, model_transforms, str(out_path))
    return load_structure(str(out_path), name=source_name)


def ca_ids(structure, model=1):
    return [a.matomid for a in structure.atoms_in_model(model) if a.atom_name == "CA"]


def test_standard_transform_matches_library(known_move):
    rotation, translation = known_move
    xyz = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
    r, t = to_standard_transform(rotation, translation)
    assert np.allclose(apply_transform(xyz, rotation, translation), xyz @ r.T + t)


def test_recover_known_move(tmp_path, known_move):
    ref = load_structure(data_path("1UBQ.pdb"))
    target = moved_copy(data_path("1UBQ.pdb"), {1: known_move}, tmp_path / "moved.pdb")
    ids = ca_ids(ref)
    fit = fit_pair(ref, target, ids, ids)
    assert fit.passed and fit.rmsd < 1e-3 and fit.so == 100.0
    # Fitted target coordinates (all atoms) return to the reference.
    all_ids = list(ref.atoms)
    back = apply_transform(target.coords(all_ids), fit.rotation, fit.translation)
    assert np.abs(back - ref.coords(all_ids)).max() < 3e-3


def test_pdb_output_keeps_other_lines(tmp_path, known_move):
    ref = load_structure(data_path("1UBQ.pdb"))
    out = tmp_path / "out.pdb"
    write_structure(ref, {1: known_move}, str(out))
    before = ref.raw_text.splitlines()
    after = out.read_text().splitlines()
    assert len(before) == len(after)
    for old, new in zip(before, after):
        if old.startswith(("ATOM  ", "HETATM")):
            assert old[:30] == new[:30] and old[54:] == new[54:]
        else:
            assert old == new


def test_cif_round_trip(tmp_path, known_move):
    ref = load_structure(data_path("1UBQ.cif"))
    target = moved_copy(data_path("1UBQ.cif"), {1: known_move}, tmp_path / "moved.cif")
    assert target.file_format == "cif" and list(target.atoms) == list(ref.atoms)
    ids = list(ref.atoms)
    expected = apply_transform(ref.coords(ids), *known_move)
    assert np.abs(target.coords(ids) - expected).max() < 1e-3


def test_anisou_is_rotated(tmp_path, known_move):
    src = data_path("1EJG_res1-8.pdb")
    out = tmp_path / "moved.pdb"
    write_structure(load_structure(src), {1: known_move}, str(out))
    parser = PDBParser(QUIET=True)
    old_atom = next(parser.get_structure("a", src).get_atoms())
    new_atom = next(parser.get_structure("b", str(out)).get_atoms())

    def tensor(u):
        u11, u22, u33, u12, u13, u23 = u
        return np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]])

    rotation = known_move[0]
    expected = rotation @ tensor(old_atom.get_anisou()) @ rotation.T
    assert np.abs(tensor(new_atom.get_anisou()) - expected).max() < 2e-4


def test_pipeline_fit_writes_all_outputs(tmp_path, known_move):
    ref = load_structure(data_path("1UBQ.pdb"))
    target = moved_copy(data_path("1UBQ.pdb"), {1: known_move}, tmp_path / "moved.pdb",
                        "1UBQ_1.pdb")
    ids = ca_ids(ref)
    result = run_superposition(ref, [FitJob(target, ids, ids)], str(tmp_path / "out"))
    names = sorted(os.listdir(tmp_path / "out"))
    assert names == ["1UBQ_1__pairs.csv", "1UBQ_1_fit.pdb", "K_fit_results.zip",
                     "report.txt", "transforms.csv"]
    report = open(result.report_path).read()
    assert "SHA-256" in report and ref.sha256 in report and ": FIT" in report
    assert "Atoms superimposed: 76" in report
    with open(result.transforms_path) as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "FIT" and row["n_atoms"] == "76"
    with zipfile.ZipFile(result.zip_path) as archive:
        assert sorted(archive.namelist()) == [n for n in names if n != "K_fit_results.zip"]
    fitted = load_structure(result.output_structures["1UBQ_1.pdb"])
    assert np.abs(fitted.coords(list(ref.atoms)) - ref.coords(list(ref.atoms))).max() < 3e-3


def test_pipeline_no_fit(tmp_path):
    ref = load_structure(data_path("1UBQ.pdb"))
    target = ref.copy("1UBQ_1.pdb")
    ids = ca_ids(ref)
    for m in ids[:5]:                        # push 5 atoms 3 A away
        target.atoms[m].coord = target.atoms[m].coord + np.array([3.0, 0.0, 0.0])
    result = run_superposition(ref, [FitJob(target, ids, ids)], str(tmp_path),
                               make_zip=False)
    fit = result.fits[0]
    assert not fit.passed and fit.so < 100.0
    assert result.output_structures["1UBQ_1.pdb"] is None
    report = open(result.report_path).read()
    assert "NO FIT" in report and f"{fit.rmsd:.4f}" in report and "not written (NO FIT)" in report


def test_pipeline_rejects_bad_selection(tmp_path):
    ref = load_structure(data_path("1UBQ.pdb"))
    ids = ca_ids(ref)
    with pytest.raises(PipelineError, match="mismatch"):
        run_superposition(ref, [FitJob(ref.copy("b.pdb"), ids, ids[:-1])], str(tmp_path))
    assert os.listdir(tmp_path) == []
    with pytest.raises(PipelineError):
        run_superposition(ref, [], str(tmp_path))


def test_fit_all_models_single_file(tmp_path):
    nmr = load_structure(data_path("1L2Y_3models.pdb"))
    ids = ca_ids(nmr, 1)
    job = self_fit_job(nmr, ids, {nmr.name})
    assert job.target.name == "1L2Y_3models_1.pdb"
    result = run_superposition(nmr, [job], str(tmp_path), rmsd_cutoff=5.0, so_cutoff=5.0,
                               fit_all_models=True, make_zip=False)
    assert [f.target_model for f in result.fits] == [1, 2, 3]
    assert result.fits[0].rmsd < 1e-3
    assert all(f.passed for f in result.fits)
    # Every model of the output is superimposed on model 1 (CA atoms).
    out = load_structure(result.output_structures[job.target.name])
    ref_xyz = nmr.coords(ids)
    for model in (2, 3):
        moved = out.coords([a.matomid for a in out.equivalent_atoms(ids, model)])
        rmsd = np.sqrt(((moved - ref_xyz) ** 2).sum(axis=1).mean())
        assert abs(rmsd - result.fits[model - 1].rmsd) < 1e-3


def test_default_moves_only_selected_model(tmp_path):
    nmr = load_structure(data_path("1L2Y_3models.pdb"))
    target = nmr.copy("t.pdb")
    ref_ids = ca_ids(nmr, 1)
    tgt_ids = ca_ids(nmr, 2)
    result = run_superposition(nmr, [FitJob(target, ref_ids, tgt_ids)], str(tmp_path),
                               rmsd_cutoff=5.0, so_cutoff=5.0, make_zip=False)
    assert len(result.fits) == 1 and result.fits[0].target_model == 2
    out = load_structure(result.output_structures["t.pdb"])
    model3 = [a.matomid for a in nmr.atoms_in_model(3)]
    assert np.allclose(out.coords(model3), nmr.coords(model3))
