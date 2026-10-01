# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Tests for K_fit.checker."""

import numpy as np
import pytest

from K_fit.checker import (PairsCsvError, check_atom_pairs, distance_from_line,
                            parse_matomid_list, read_pairs_csv)
from K_fit.errors import KFitError
from K_fit.parser import load_structure

from conftest import data_path


@pytest.fixture(scope="module")
def ubq():
    return load_structure(data_path("1UBQ.pdb"))


@pytest.fixture(scope="module")
def nmr():
    return load_structure(data_path("1L2Y_3models.pdb"))


def ca_ids(structure, model=1, n=None):
    ids = [a.matomid for a in structure.atoms_in_model(model) if a.atom_name == "CA"]
    return ids[:n] if n else ids


def test_valid_pairs(ubq):
    ids = ca_ids(ubq, n=10)
    result = check_atom_pairs(ubq, ubq, ids, ids)
    assert result.ok and len(result.pairs) == 10 and not result.warnings


def test_count_mismatch_is_error(ubq):
    ids = ca_ids(ubq, n=10)
    result = check_atom_pairs(ubq, ubq, ids, ids[:9])
    assert not result.ok and "mismatch" in result.errors[0]


def test_unknown_and_duplicate_ids(ubq):
    ids = ca_ids(ubq, n=4)
    result = check_atom_pairs(ubq, ubq, ids, ids[:3] + ["1-99999"])
    assert any("unknown" in e for e in result.errors)
    result = check_atom_pairs(ubq, ubq, ids, ids[:3] + ids[:1])
    assert any("twice" in e for e in result.errors)


def test_mixed_models_is_error(nmr):
    mixed = ca_ids(nmr, 1, 3) + ca_ids(nmr, 2, 3)[:1]
    result = check_atom_pairs(nmr, nmr, ca_ids(nmr, 1, 4), mixed)
    assert any("several models" in e for e in result.errors)


def test_element_and_name_warnings(ubq):
    ca = ca_ids(ubq, n=3)
    n_atoms = [a.matomid for a in ubq.atoms_in_model(1) if a.atom_name == "N"][:3]
    cb = [a.matomid for a in ubq.atoms_in_model(1) if a.atom_name == "CB"][:3]
    assert any("element differs" in w for w in check_atom_pairs(ubq, ubq, ca, n_atoms).warnings)
    assert any("atom name differs" in w for w in check_atom_pairs(ubq, ubq, ca, cb).warnings)


def test_too_few_pairs_is_error(ubq):
    ids = ca_ids(ubq, n=2)
    result = check_atom_pairs(ubq, ubq, ids, ids)
    assert not result.ok and "only 2 pair(s)" in result.errors[0]


def test_collinear_selection_is_error(ubq):
    ids = ca_ids(ubq, n=4)
    line = ubq.copy("line.pdb")
    for i, m in enumerate(ids):                  # put the 4 atoms on one line
        line.atoms[m].coord = np.array([1.5 * i, 2.0 * i, -1.0 * i])
    result = check_atom_pairs(ubq, line, ids, ids)
    assert not result.ok and "one straight line" in result.errors[0]
    assert "target line.pdb" in result.errors[0]
    assert distance_from_line(ubq.coords(ids)) > 0.5    # a real beta strand passes


def test_parse_matomid_list():
    assert parse_matomid_list("1-10, 1-11\n1-12  1-13") == ["1-10", "1-11", "1-12", "1-13"]


def test_read_pairs_csv():
    text = "ref.pdb,a.pdb,b.cif\n1-1,1-2,1-3\n\n1-4,1-5,1-6\n"
    ref, selections = read_pairs_csv(text, ["ref.pdb", "a.pdb", "b.cif"])
    assert ref == "ref.pdb"
    assert selections == {"a.pdb": (["1-1", "1-4"], ["1-2", "1-5"]),
                          "b.cif": (["1-1", "1-4"], ["1-3", "1-6"])}


@pytest.mark.parametrize("text", [
    "ref.pdb\n1-1\n",                        # one column
    "ref.pdb,x.pdb\n1-1,1-2\n",              # unknown header
    "ref.pdb,a.pdb\n1-1,\n",                 # empty cell
    "ref.pdb,a.pdb\n",                       # no rows
])
def test_read_pairs_csv_errors(text):
    assert issubclass(PairsCsvError, KFitError)
    with pytest.raises(PairsCsvError):
        read_pairs_csv(text, ["ref.pdb", "a.pdb"])
