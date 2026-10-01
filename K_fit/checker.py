# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Check atom selections before fitting, and read the pairs CSV.

Rules applied by ``check_atom_pairs``:

Errors (fitting must not run):
  * a selection is empty
  * the reference and target selections have different numbers of atoms
  * a matomid does not exist in its structure
  * the same matomid appears twice in one selection
  * one selection mixes atoms from more than one model
  * fewer than 3 pairs (the rotation is then not defined)
  * the atoms of a selection lie on one straight line (the rotation about
    that line is then not defined)

Warnings (fitting may run):
  * a pair has different atom names or different elements

To add a rule, append to ``errors`` or ``warnings`` in ``check_atom_pairs``.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import numpy as np

from .errors import KFitError
from .parser import AtomRecord, Structure

# Minimum number of pairs for a well-defined rotation.
MIN_PAIRS_FOR_ROTATION = 3

# A selection counts as collinear when the RMS distance of its atoms from
# their best-fit straight line is below this value (Angstrom). Raise it to
# also reject nearly collinear selections.
COLLINEAR_TOLERANCE = 0.05


@dataclass
class AtomPair:
    """One reference atom and the target atom fitted onto it."""

    reference: AtomRecord
    target: AtomRecord


@dataclass
class PairCheckResult:
    """Outcome of ``check_atom_pairs``.

    ``pairs`` is filled only when there are no errors.
    """

    reference_name: str
    target_name: str
    pairs: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no errors."""
        return not self.errors

    def as_text(self) -> str:
        """Return the pair table, errors and warnings as printable text."""
        lines = [f"Pairs {self.reference_name} <- {self.target_name}: {len(self.pairs)}"]
        for i, pair in enumerate(self.pairs, start=1):
            lines.append(f"  {i:>4}  {pair.reference.label():<45} <- {pair.target.label()}")
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def parse_matomid_list(text: str) -> list:
    """Split user-typed matomids (comma, space or newline separated) into a list.

    Example: "1-10, 1-11\\n1-12" -> ["1-10", "1-11", "1-12"].
    """
    return [token for token in text.replace(",", " ").split() if token]


def _check_one_selection(structure: Structure, ids: list, role: str) -> list:
    """Return error messages for one selection (unknown ids, duplicates, mixed models)."""
    errors = []
    if not ids:
        return [f"{role} selection for {structure.name} is empty"]
    unknown = [m for m in ids if m not in structure.atoms]
    if unknown:
        errors.append(f"{role} {structure.name}: unknown matomid(s) {', '.join(unknown[:10])}"
                      + (" ..." if len(unknown) > 10 else ""))
    seen, duplicates = set(), []
    for m in ids:
        if m in seen:
            duplicates.append(m)
        seen.add(m)
    if duplicates:
        errors.append(f"{role} {structure.name}: matomid(s) selected twice: "
                      f"{', '.join(dict.fromkeys(duplicates))}")
    models = {structure.atoms[m].model_number for m in ids if m in structure.atoms}
    if len(models) > 1:
        errors.append(f"{role} {structure.name}: atoms come from several models "
                      f"{sorted(models)}; select atoms from one model only")
    return errors


def distance_from_line(coords: np.ndarray) -> float:
    """Return the RMS distance (A) of (N, 3) points from their best-fit line.

    The centred coordinates are decomposed by SVD: the first singular
    value belongs to the best-fit line, the other two measure the spread
    away from it. Returns 0.0 for fewer than 3 points.
    """
    if len(coords) < 3:
        return 0.0
    centred = coords - coords.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    return float(np.sqrt((singular_values[1:] ** 2).sum() / len(coords)))


def check_atom_pairs(reference: Structure, target: Structure,
                     reference_ids: list, target_ids: list) -> PairCheckResult:
    """Validate two ordered matomid lists and pair them up by position.

    The i-th reference atom is paired with the i-th target atom.
    See the module docstring for the list of errors and warnings.
    """
    result = PairCheckResult(reference_name=reference.name, target_name=target.name)
    result.errors += _check_one_selection(reference, reference_ids, "reference")
    result.errors += _check_one_selection(target, target_ids, "target")
    if len(reference_ids) != len(target_ids):
        result.errors.append(
            f"atom count mismatch: reference {reference.name} has {len(reference_ids)}, "
            f"target {target.name} has {len(target_ids)}")
    if result.errors:
        return result
    if len(reference_ids) < MIN_PAIRS_FOR_ROTATION:
        result.errors.append(
            f"only {len(reference_ids)} pair(s); at least {MIN_PAIRS_FOR_ROTATION} "
            "atoms not on one straight line are needed to define a rotation")
        return result
    for structure, ids, role in ((reference, reference_ids, "reference"),
                                 (target, target_ids, "target")):
        spread = distance_from_line(structure.coords(ids))
        if spread < COLLINEAR_TOLERANCE:
            result.errors.append(
                f"{role} {structure.name}: the selected atoms lie on one straight line "
                f"(RMS distance from the line {spread:.3f} A < {COLLINEAR_TOLERANCE} A); "
                "add an atom off that line")
    if result.errors:
        return result

    for ref_id, tgt_id in zip(reference_ids, target_ids):
        pair = AtomPair(reference.atoms[ref_id], target.atoms[tgt_id])
        result.pairs.append(pair)
        if pair.reference.element != pair.target.element:
            result.warnings.append(
                f"element differs: {pair.reference.label()} ({pair.reference.element}) vs "
                f"{pair.target.label()} ({pair.target.element})")
        elif pair.reference.atom_name != pair.target.atom_name:
            result.warnings.append(
                f"atom name differs: {pair.reference.label()} vs {pair.target.label()}")
    return result


class PairsCsvError(KFitError):
    """Raised when a pairs CSV file is malformed."""


def read_pairs_csv(text: str, structure_names: list) -> tuple:
    """Read a pairs CSV and return (reference_name, {target_name: (ref_ids, tgt_ids)}).

    Format: the header row holds structure names (file names, after
    renaming of duplicates). The first column is the reference, each further
    column one target. Each following row is one atom pair of matomids:

        ref.pdb,model_a.pdb,1ABC.cif
        1-10,1-12,1-305
        1-18,1-20,1-313

    ``text`` is the CSV content (not a path), which works for both files on
    disk and Colab uploads. ``structure_names`` lists the loaded names used
    to validate the header. Blank lines are ignored. Every row must have a
    value in every column.
    """
    rows = [row for row in csv.reader(io.StringIO(text))
            if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise PairsCsvError("pairs CSV needs a header row and at least one atom row")
    header = [cell.strip() for cell in rows[0]]
    if len(header) < 2:
        raise PairsCsvError("pairs CSV needs at least two columns (reference and one target)")
    unknown = [h for h in header if h not in structure_names]
    if unknown:
        raise PairsCsvError(f"pairs CSV header names not loaded: {', '.join(unknown)}; "
                            f"loaded structures: {', '.join(structure_names)}")
    if len(set(header)) != len(header):
        raise PairsCsvError("pairs CSV header repeats a structure name")

    columns = [[] for _ in header]
    for line_no, row in enumerate(rows[1:], start=2):
        cells = [cell.strip() for cell in row]
        if len(cells) != len(header) or not all(cells):
            raise PairsCsvError(f"pairs CSV row {line_no}: expected {len(header)} non-empty values")
        for column, value in zip(columns, cells):
            column.append(value)

    reference_name = header[0]
    selections = {name: (columns[0], column)
                  for name, column in zip(header[1:], columns[1:])}
    return reference_name, selections
