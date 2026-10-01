# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Tests for K_fit.parser."""

import numpy as np
import pytest

from K_fit.parser import (StructureError, detect_format, load_structure,
                          make_unique_name, sha256_of_file)

from conftest import data_path


def test_pdb_counts_and_matomid():
    s = load_structure(data_path("1UBQ.pdb"))
    summary = s.summary()
    assert s.models() == [1]                      # no MODEL record -> model 1
    assert "1-1" in s.atoms and s.atoms["1-1"].atom_name == "N"
    assert summary.n_atoms == 660
    assert summary.n_chains == 1
    assert summary.water_count == 58
    assert summary.n_residues == 76 + 58
    assert summary.hetero_residues == []


def test_pdb_and_cif_agree():
    pdb = load_structure(data_path("1UBQ.pdb"))
    cif = load_structure(data_path("1UBQ.cif"))
    assert pdb.file_format == "pdb" and cif.file_format == "cif"
    assert len(pdb.atoms) == len(cif.atoms)
    # Protein atoms share ids; waters differ by one because the PDB TER
    # record consumes serial 603. Compare every atom by identity instead.
    by_key = {a.model_independent_key(): a.coord for a in cif.atoms.values()}
    for atom in pdb.atoms.values():
        assert np.allclose(atom.coord, by_key[atom.model_independent_key()])
    assert pdb.atoms["1-602"].model_independent_key() == cif.atoms["1-602"].model_independent_key()


def test_nmr_models_and_equivalent_atoms():
    s = load_structure(data_path("1L2Y_3models.pdb"))
    assert s.models() == [1, 2, 3]
    assert s.summary().n_models == 3
    # PDB serials restart in each model; matomids stay unique.
    assert s.atoms["1-1"].atom_id == s.atoms["2-1"].atom_id == 1
    ca = [a.matomid for a in s.atoms_in_model(1) if a.atom_name == "CA"]
    matched = s.equivalent_atoms(ca, 3)
    assert all(m is not None and m.model_number == 3 and m.atom_name == "CA" for m in matched)


def test_altlocs_are_separate_atoms():
    s = load_structure(data_path("1EJG_res1-8.pdb"))
    thr1 = s.atoms_in_residue(1, ("A", 1, " ", "THR"))
    n_atoms = [a for a in thr1 if a.atom_name == "N"]
    assert {a.altloc for a in n_atoms} == {"A", "B"}
    assert "altloc A (occ 0.82)" in n_atoms[0].label()


def test_navigation_helpers():
    s = load_structure(data_path("1UBQ.pdb"))
    assert s.chains(1) == ["A"]
    residues = s.residues(1, "A")
    assert residues[0] == ("A", 1, " ", "MET")
    assert [a.atom_name for a in s.atoms_in_residue(1, residues[0])][:2] == ["N", "CA"]


def test_make_unique_name():
    taken = {"model.pdb", "model_1.pdb"}
    assert make_unique_name("other.pdb", taken) == "other.pdb"
    assert make_unique_name("model.pdb", taken) == "model_2.pdb"


def test_copy_is_independent():
    s = load_structure(data_path("1UBQ.pdb"))
    c = s.copy("1UBQ_1.pdb")
    c.atoms["1-1"].coord[0] += 5.0
    assert c.name == "1UBQ_1.pdb" and s.name == "1UBQ.pdb"
    assert s.atoms["1-1"].coord[0] != c.atoms["1-1"].coord[0]


def test_sha256_and_detect_format(tmp_path):
    s = load_structure(data_path("1UBQ.pdb"))
    assert s.sha256 == sha256_of_file(data_path("1UBQ.pdb")) and len(s.sha256) == 64
    unknown_ext = tmp_path / "structure.txt"
    unknown_ext.write_text(open(data_path("1UBQ.cif")).read())
    assert detect_format(str(unknown_ext)) == "cif"


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.pdb"
    empty.write_text("REMARK nothing here\nEND\n")
    with pytest.raises(StructureError):
        load_structure(str(empty))
