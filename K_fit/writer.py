# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Write K_fit outputs: moved structures, the text report and CSV files.

Structure files keep the input format but hold only coordinates. Header
records (crystal cell, symmetry operators, biological assemblies, secondary
structure, links and so on) are removed, because after the coordinates are
moved they would no longer match.

* PDB: only MODEL, ATOM, HETATM, ANISOU, TER, ENDMDL and END lines are kept,
  in their original order. The x, y, z columns (31-54) of ATOM/HETATM
  records in moved models are replaced and the ANISOU tensors rotated; all
  other columns are copied unchanged.
* mmCIF: the input text is read with Biopython's ``MMCIF2Dict``. Only the
  data block name, ``_entry.id``, ``_atom_site`` and ``_atom_site_anisotrop``
  are kept. ``Cartn_x/y/z`` (and the anisotropic tensors) of moved models are
  replaced and the result is written with ``MMCIFIO``.

To keep another PDB record or mmCIF category, add it to
``PDB_KEPT_RECORDS`` or ``CIF_KEPT_PREFIXES``.

Transformations are given to this module in kearsley-numba's convention
(x' = rotation @ (x - translation)). ``to_standard_transform`` converts them
to x' = R x + t for the report and ``transforms.csv``; that conversion is done
only here.
"""

from __future__ import annotations

import csv
import datetime
import io
import os

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO

from . import __version__
from .errors import KFitError
from .fitter import FitResult, apply_transform
from .parser import Structure

# PDB records written to fitted structure files (record name padded to 6).
PDB_KEPT_RECORDS = ("MODEL ", "ATOM  ", "HETATM", "ANISOU", "TER   ", "ENDMDL", "END   ")

# mmCIF items written to fitted structure files ("data_" is the block name).
CIF_KEPT_PREFIXES = ("data_", "_entry.id", "_atom_site.", "_atom_site_anisotrop.")


class WriterError(KFitError):
    """Raised when an output file cannot be written."""


# Transformation helpers

def to_standard_transform(rotation: np.ndarray, translation: np.ndarray) -> tuple:
    """Convert kearsley-numba (rotation, translation) to standard (R, t).

    kearsley-numba: x' = rotation @ (x - translation)
    standard:       x' = R @ x + t   with R = rotation, t = -rotation @ translation
    """
    rotation = np.asarray(rotation, dtype=float)
    return rotation, -rotation @ np.asarray(translation, dtype=float)


def _rotate_tensor(u6: list, rotation: np.ndarray) -> list:
    """Rotate an anisotropic tensor given as [U11, U22, U33, U12, U13, U23].

    Returns the rotated six values (R U R^T) in the same order.
    """
    u11, u22, u33, u12, u13, u23 = u6
    tensor = np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]], dtype=float)
    r = rotation @ tensor @ rotation.T
    return [r[0, 0], r[1, 1], r[2, 2], r[0, 1], r[0, 2], r[1, 2]]


def _file_model_number(serial: int) -> int:
    """Return the K_fit model number for a model serial read from a file.

    Same rule as ``parser._model_number``: a missing or 0 serial is model 1.
    """
    return serial or 1


# Structure files

def output_file_name(structure: Structure, stem: str | None = None) -> str:
    """Return the output file name for a moved structure: ``<stem>_fit<ext>``.

    ``stem`` defaults to the structure name without its extension; the
    pipeline passes its own when two targets share a stem.
    """
    name_stem, ext = os.path.splitext(structure.name)
    ext = ext or (".cif" if structure.file_format == "cif" else ".pdb")
    return f"{stem or name_stem}_fit{ext}"


def write_structure(structure: Structure, model_transforms: dict, out_path: str,
                    models_to_write=None) -> str:
    """Write ``structure`` with the listed models moved, in its input format.

    ``model_transforms`` maps model number -> (rotation, translation) in
    kearsley-numba convention. Models not listed keep their coordinates.
    ``models_to_write`` is a set of model numbers to include in the file;
    None writes every model. Only coordinate data is written (see module
    doc). Returns ``out_path``.
    """
    if structure.file_format == "pdb":
        _write_pdb(structure, model_transforms, out_path, models_to_write)
    elif structure.file_format == "cif":
        _write_cif(structure, model_transforms, out_path, models_to_write)
    else:
        raise WriterError(f"cannot write format {structure.file_format}")
    return out_path


def _write_pdb(structure: Structure, model_transforms: dict, out_path: str,
               models_to_write) -> None:
    """Write the coordinate records of the PDB text, moving listed models (see module doc)."""
    model_number = 1          # PDB files without MODEL records are model 1
    out_lines = []
    for line in structure.raw_text.splitlines(keepends=True):
        original = line.rstrip("\r\n")
        ending = line[len(original):]
        record = original[:6].ljust(6)
        if record not in PDB_KEPT_RECORDS:
            continue
        if record == "MODEL ":
            fields = original[6:].split()
            model_number = _file_model_number(
                int(original[10:14].strip() or (fields[0] if fields else 0)))
        if record != "END   " and models_to_write is not None \
                and model_number not in models_to_write:
            continue
        if record in ("ATOM  ", "HETATM", "ANISOU") and model_number in model_transforms:
            rotation, translation = model_transforms[model_number]
            width = 70 if record == "ANISOU" else 54   # last replaced column
            body = original.ljust(width)
            try:
                if record == "ANISOU":
                    u6 = [int(body[28 + 7 * i:35 + 7 * i]) for i in range(6)]
                    new_u = _rotate_tensor(u6, rotation)
                    body = body[:28] + "".join(f"{round(v):7d}" for v in new_u) + body[70:]
                else:
                    xyz = np.array([float(body[30:38]), float(body[38:46]),
                                    float(body[46:54])])
                    x, y, z = apply_transform(xyz, rotation, translation)[0]
                    coords = f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    if len(coords) != 24:
                        raise WriterError(
                            f"{structure.name}: coordinate out of PDB range after fitting "
                            f"({x:.3f}, {y:.3f}, {z:.3f}); use mmCIF input instead")
                    body = body[:30] + coords + body[54:]
            except ValueError as err:
                raise WriterError(f"{structure.name}: unreadable {record.strip()} line: "
                                  f"{original!r}") from err
            original = body
        out_lines.append(original + ending)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.writelines(out_lines)


# mmCIF anisotropic tensor tags, in the order used by _rotate_tensor.
_ANISO_ORDER = ["[1][1]", "[2][2]", "[3][3]", "[1][2]", "[1][3]", "[2][3]"]


def _keep_rows(data: dict, prefix: str, keep: list) -> None:
    """Keep only the loop rows of category ``prefix`` whose index is in ``keep``."""
    for key in data:
        if key.startswith(prefix):
            data[key] = [data[key][row] for row in keep]


def _write_cif(structure: Structure, model_transforms: dict, out_path: str,
               models_to_write) -> None:
    """Write _atom_site (and anisotropic tensors), moving listed models (see module doc).

    The input is read from ``structure.raw_text``, so the input file does
    not need to exist any more.
    """
    full = MMCIF2Dict(io.StringIO(structure.raw_text))
    data = {key: value for key, value in full.items()
            if key.startswith(CIF_KEPT_PREFIXES)}
    ids = data["_atom_site.id"]
    n_rows = len(ids)
    models = data.get("_atom_site.pdbx_PDB_model_num", ["1"] * n_rows)
    xs, ys, zs = (data[f"_atom_site.Cartn_{c}"] for c in "xyz")

    kept_rows = []
    kept_ids = {}                 # atom id -> model number, for anisotropic tensors
    for row in range(n_rows):
        model_number = _file_model_number(
            int(models[row]) if models[row] not in ("?", ".") else 0)
        if models_to_write is not None and model_number not in models_to_write:
            continue
        kept_rows.append(row)
        kept_ids[ids[row]] = model_number
        if model_number not in model_transforms:
            continue
        rotation, translation = model_transforms[model_number]
        xyz = np.array([float(xs[row]), float(ys[row]), float(zs[row])])
        x, y, z = apply_transform(xyz, rotation, translation)[0]
        xs[row], ys[row], zs[row] = f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"
    if len(kept_rows) != n_rows:
        _keep_rows(data, "_atom_site.", kept_rows)

    if "_atom_site_anisotrop.id" in data:
        aniso_ids = data["_atom_site_anisotrop.id"]
        for prefix in ("U", "B"):
            tags = [f"_atom_site_anisotrop.{prefix}{suffix}" for suffix in _ANISO_ORDER]
            if not all(t in data for t in tags):
                continue
            for row, atom_id in enumerate(aniso_ids):
                if kept_ids.get(atom_id) not in model_transforms:
                    continue
                rotation, _ = model_transforms[kept_ids[atom_id]]
                u6 = [float(data[t][row]) for t in tags]
                for tag, value in zip(tags, _rotate_tensor(u6, rotation)):
                    data[tag][row] = f"{value:.4f}"
        _keep_rows(data, "_atom_site_anisotrop.",
                   [row for row, atom_id in enumerate(aniso_ids) if atom_id in kept_ids])

    io_writer = MMCIFIO()
    io_writer.set_dict(data)
    io_writer.save(out_path)


# Report and CSV files

def pair_rows(reference: Structure, target: Structure, fit: FitResult) -> list:
    """Return CSV rows (dicts) describing each atom pair of one fit."""
    rows = []
    for i, (ref_id, tgt_id) in enumerate(zip(fit.reference_ids, fit.target_ids), start=1):
        ref_atom, tgt_atom = reference.atoms[ref_id], target.atoms[tgt_id]
        row = {"target_model": fit.target_model, "pair": i}
        for side, atom in (("reference", ref_atom), ("target", tgt_atom)):
            row.update({
                f"{side}_matomid": atom.matomid,
                f"{side}_chain": atom.chain_id,
                f"{side}_residue_name": atom.residue_name,
                f"{side}_residue_number": f"{atom.residue_number}{atom.insertion_code.strip()}",
                f"{side}_atom_name": atom.atom_name,
                f"{side}_element": atom.element,
                f"{side}_altloc": atom.altloc,
            })
        row["distance_after_fit"] = (f"{fit.pair_distances[i - 1]:.3f}"
                                     if fit.pair_distances is not None else "")
        rows.append(row)
    return rows


PAIRS_COLUMNS = (["target_model", "pair"]
                 + [f"{side}_{col}" for side in ("reference", "target")
                    for col in ("matomid", "chain", "residue_name", "residue_number",
                                "atom_name", "element", "altloc")]
                 + ["distance_after_fit"])


def write_pairs_csv(reference: Structure, target: Structure, fits: list,
                    out_path: str) -> str:
    """Write the exact atom pairs used for every fit of one target.

    One row per atom pair; ``target_model`` tells the fits apart when all
    models were fitted. Returns ``out_path``.
    """
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRS_COLUMNS)
        writer.writeheader()
        for fit in fits:
            writer.writerows(pair_rows(reference, target, fit))
    return out_path


TRANSFORMS_COLUMNS = (["reference", "reference_model", "target", "target_model",
                       "n_atoms", "rmsd", "so_percent", "status"]
                      + [f"R{i}{j}" for i in range(1, 4) for j in range(1, 4)]
                      + ["t1", "t2", "t3"])


def write_transforms_csv(fits: list, out_path: str) -> str:
    """Write one row per fit with RMSD, SO, status and the standard R and t.

    R and t follow x' = R x + t (see ``to_standard_transform``). They are
    empty for fits that could not be run. Returns ``out_path``.
    """
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRANSFORMS_COLUMNS)
        writer.writeheader()
        for fit in fits:
            row = {
                "reference": fit.reference_name, "reference_model": fit.reference_model,
                "target": fit.target_name, "target_model": fit.target_model,
                "n_atoms": fit.n_atoms,
                "rmsd": f"{fit.rmsd:.4f}" if fit.rmsd is not None else "",
                "so_percent": f"{fit.so:.2f}" if fit.so is not None else "",
                "status": fit.status,
            }
            if fit.rotation is not None:
                r, t = to_standard_transform(fit.rotation, fit.translation)
                row.update({f"R{i + 1}{j + 1}": f"{r[i, j]:.6f}"
                            for i in range(3) for j in range(3)})
                row.update({f"t{i + 1}": f"{t[i]:.4f}" for i in range(3)})
            writer.writerow(row)
    return out_path


def _format_matrix(fit: FitResult) -> list:
    """Return report lines with the standard R and t of one fit."""
    r, t = to_standard_transform(fit.rotation, fit.translation)
    lines = ["    Rotation R and translation t (x' = R x + t):"]
    for i in range(3):
        lines.append("      [" + " ".join(f"{r[i, j]:10.6f}" for j in range(3))
                     + f" ]   [ {t[i]:10.4f} ]")
    return lines


def write_report(reference: Structure, targets: list, fits: list, checks: list,
                 so_cutoff: float, rmsd_cutoff: float, fit_all_models: bool,
                 output_files: dict, out_path: str) -> str:
    """Write the detailed text report and return ``out_path``.

    Sections: inputs (mode, file name, SHA-256), parameters, then one block per
    fit (atoms, RMSD, SO, FIT / NO FIT with reasons, warnings, R and t) and
    the list of output files.

    ``checks`` are the PairCheckResults (their warnings are repeated here).
    ``output_files`` maps target name -> written structure file or None.
    """
    mode_text = {"upload": "user upload", "pdb_id": "PDB ID"}
    lines = [
        "K_fit superposition report",
        "=" * 26,
        f"K_fit version {__version__}",
        f"Date (UTC): {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}",
        "",
        "Inputs",
        "------",
    ]
    for role, structure in [("reference", reference)] + [("target", t) for t in targets]:
        lines.append(f"  {role:<9} {structure.name}")
        lines.append(f"            input mode: {mode_text.get(structure.source, structure.source)}")
        lines.append(f"            file name:  {structure.file_name}")
        lines.append(f"            SHA-256:    {structure.sha256}")
    lines += [
        "",
        "Parameters",
        "----------",
        f"  SOdc (structure overlap distance cutoff): {so_cutoff:.3f} A",
        f"  RMSDc (maximum RMSD):                     {rmsd_cutoff:.3f} A",
        f"  Fit all models:                           {'yes' if fit_all_models else 'no'}",
        "  A fit is reported as FIT only if RMSD <= RMSDc and every fitted atom",
        "  lies within SOdc of its reference atom (SO = 100 %).",
        "",
        "Superpositions",
        "--------------",
    ]
    for fit in fits:
        lines.append(f"  {fit.target_name} model {fit.target_model} -> "
                     f"{fit.reference_name} model {fit.reference_model}: {fit.status}")
        lines.append(f"    Atoms superimposed: {fit.n_atoms}")
        if fit.rmsd is not None:
            lines.append(f"    RMSD: {fit.rmsd:.4f} A")
            lines.append(f"    Structure overlap (SO): {fit.so:.2f} %")
        for reason in fit.failure_reasons:
            lines.append(f"    NO FIT reason: {reason}")
        if fit.rotation is not None:
            lines += _format_matrix(fit)
        lines.append("")
    warnings = [(c.target_name, w) for c in checks for w in c.warnings]
    if warnings:
        lines += ["Warnings", "--------"]
        lines += [f"  {name}: {w}" for name, w in warnings]
        lines.append("")
    lines += ["Output structure files (coordinates only; header records removed)",
              "------------------------------------------------------------------"]
    for name, path in output_files.items():
        if not path:
            lines.append(f"  {name}: not written (NO FIT)")
            continue
        line = f"  {name}: {os.path.basename(path)}"
        if fit_all_models:
            fitted = [f.target_model for f in fits if f.target_name == name and f.passed]
            failed = [f.target_model for f in fits if f.target_name == name and not f.passed]
            line += f" (models {', '.join(map(str, fitted))}"
            line += f"; NO FIT models {', '.join(map(str, failed))} left out)" if failed else ")"
        lines.append(line)
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return out_path
