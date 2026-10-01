# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Shared test fixtures.

Test structures in tests/data are excerpts of RCSB PDB entries:
  1UBQ.pdb / 1UBQ.cif   ubiquitin, complete entry (waters, no altlocs)
  1L2Y_3models.pdb      Trp-cage NMR ensemble, first 3 models only
  1EJG_res1-8.pdb       crambin at 0.54 A, residues 1-8 (altlocs + ANISOU)
"""

import os

import numpy as np
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def data_path(name: str) -> str:
    """Return the absolute path of a file in tests/data."""
    return os.path.join(DATA_DIR, name)


def rotation_about_axis(axis, angle_deg: float) -> np.ndarray:
    """Return a 3x3 rotation matrix (Rodrigues formula)."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    a = np.radians(angle_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


@pytest.fixture
def known_move():
    """A (rotation, translation) pair in kearsley-numba convention."""
    return rotation_about_axis([1, 2, 3], 37.0), np.array([4.0, -7.5, 12.25])
