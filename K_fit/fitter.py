# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Kearsley superposition of selected atom pairs with kearsley-numba.

Convention: ``rotation`` and ``translation`` in ``FitResult`` are kept exactly
as kearsley-numba returns them, and coordinates are moved with
``kearsley_numba.transform``:

    x_fitted = rotation @ (x - translation)

The writer module converts them to the standard form (x' = R x + t) for the
output files only (see ``writer.to_standard_transform``).

A fit passes when both cutoffs are met:
    RMSD <= rmsd_cutoff  and  structure overlap (SO) == 100 %
SO is the percentage of fitted target atoms closer than ``so_cutoff`` to
their reference atom (strict "<", as in kearsley-numba).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from kearsley_numba import fit_transform, transform

from .parser import Structure

DEFAULT_SO_CUTOFF = 1.5    # Angstrom (SOdc)
DEFAULT_RMSD_CUTOFF = 1.5  # Angstrom (RMSDc)


@dataclass
class FitResult:
    """Result of fitting one target selection onto the reference selection.

    ``rmsd``, ``so``, ``rotation``, ``translation`` and ``pair_distances`` are
    None when the fit could not be run (see ``failure_reasons``).
    ``pair_distances`` holds the distance of each pair after fitting.
    """

    reference_name: str
    target_name: str
    reference_model: int
    target_model: int
    reference_ids: list
    target_ids: list
    so_cutoff: float
    rmsd_cutoff: float
    rmsd: float | None = None
    so: float | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    pair_distances: np.ndarray | None = None
    passed: bool = False
    failure_reasons: list = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        """Number of atom pairs used for the fit."""
        return len(self.target_ids)

    @property
    def status(self) -> str:
        """"FIT" or "NO FIT"."""
        return "FIT" if self.passed else "NO FIT"


def apply_transform(coords: np.ndarray, rotation: np.ndarray,
                    translation: np.ndarray) -> np.ndarray:
    """Move an (M, 3) coordinate array with a kearsley-numba transformation.

    Returns a new array: rotation @ (x - translation) for every row.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
    return transform(coords, np.ascontiguousarray(rotation, dtype=np.float64),
                     np.ascontiguousarray(translation, dtype=np.float64))


def fit_pair(reference: Structure, target: Structure, reference_ids: list,
             target_ids: list, so_cutoff: float = DEFAULT_SO_CUTOFF,
             rmsd_cutoff: float = DEFAULT_RMSD_CUTOFF) -> FitResult:
    """Fit the target atoms onto the reference atoms and apply the cutoffs.

    The i-th target matomid is paired with the i-th reference matomid.
    Selections must already have passed ``checker.check_atom_pairs``: both
    lists are the same length and come from one model each.
    """
    result = FitResult(
        reference_name=reference.name,
        target_name=target.name,
        reference_model=reference.atoms[reference_ids[0]].model_number,
        target_model=target.atoms[target_ids[0]].model_number,
        reference_ids=list(reference_ids),
        target_ids=list(target_ids),
        so_cutoff=float(so_cutoff),
        rmsd_cutoff=float(rmsd_cutoff),
    )
    u = np.ascontiguousarray(reference.coords(reference_ids))
    v = np.ascontiguousarray(target.coords(target_ids))
    fitted, rmsd, so, rotation, translation = fit_transform(u, v, float(so_cutoff))

    result.rmsd = float(rmsd)
    result.so = float(so)
    result.rotation = np.array(rotation)
    result.translation = np.array(translation)
    result.pair_distances = np.linalg.norm(u - fitted, axis=1)

    if result.rmsd > rmsd_cutoff:
        result.failure_reasons.append(
            f"RMSD {result.rmsd:.3f} A > RMSDc {rmsd_cutoff:.3f} A")
    if result.so < 100.0:
        n_out = int(np.sum(result.pair_distances >= so_cutoff))
        result.failure_reasons.append(
            f"SO {result.so:.1f} % < 100 % ({n_out} atom(s) not within SOdc {so_cutoff:.3f} A)")
    result.passed = not result.failure_reasons
    return result


def unfittable_result(reference: Structure, target: Structure, reference_ids: list,
                      target_model: int, reason: str, so_cutoff: float,
                      rmsd_cutoff: float) -> FitResult:
    """Return a NO FIT result for a model that could not be fitted at all.

    Used by the "fit all models" option when a model lacks some of the
    selected atoms.
    """
    return FitResult(
        reference_name=reference.name,
        target_name=target.name,
        reference_model=reference.atoms[reference_ids[0]].model_number,
        target_model=target_model,
        reference_ids=list(reference_ids),
        target_ids=[],
        so_cutoff=float(so_cutoff),
        rmsd_cutoff=float(rmsd_cutoff),
        failure_reasons=[reason],
    )
