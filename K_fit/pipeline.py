# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Run a complete superposition job: check, fit, write, zip.

This is the single entry point used by the Colab notebook, and can be used
by any other program:

    from K_fit.parser import load_structure
    from K_fit.pipeline import FitJob, run_superposition

    ref = load_structure("ref.pdb")
    tgt = load_structure("model.pdb")
    job = FitJob(target=tgt, reference_ids=["1-2", "1-10", "1-18"],
                 target_ids=["1-5", "1-13", "1-21"])
    result = run_superposition(ref, [job], out_dir="out")
    print(result.summary_text())

Fit all models (``fit_all_models=True``): every model of each target is
fitted separately onto the reference selection. The atoms of the other models
are found by chain, residue, atom name and altloc (see
``Structure.equivalent_atoms``). A model that lacks any selected atom is
reported as NO FIT. To fit the models of a single file onto one of its own
models, pass a job made by ``self_fit_job``.

Without fit_all_models only the selected model of each target is moved;
other models are written unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .archive import zip_outputs
from .checker import check_atom_pairs
from .fitter import DEFAULT_RMSD_CUTOFF, DEFAULT_SO_CUTOFF, fit_pair, unfittable_result
from .parser import Structure, make_unique_name
from .writer import (output_file_name, write_pairs_csv, write_report, write_structure,
                     write_transforms_csv)

REPORT_NAME = "report.txt"
TRANSFORMS_NAME = "transforms.csv"
ZIP_NAME = "K_fit_results.zip"


class PipelineError(Exception):
    """Raised when a job cannot run (e.g. atom selections fail the checks)."""


@dataclass
class FitJob:
    """One target structure and the atoms to fit (paired by position)."""

    target: Structure
    reference_ids: list
    target_ids: list


@dataclass
class RunResult:
    """Everything produced by ``run_superposition``.

    ``output_structures`` maps target name -> written file path, or None
    when no fit of that target passed. ``pairs_files`` maps target name ->
    pairs CSV path.
    """

    reference: Structure
    targets: list
    checks: list
    fits: list
    output_structures: dict = field(default_factory=dict)
    pairs_files: dict = field(default_factory=dict)
    report_path: str = ""
    transforms_path: str = ""
    zip_path: str = ""

    def all_files(self) -> list:
        """Return every written file except the zip."""
        return ([p for p in self.output_structures.values() if p]
                + list(self.pairs_files.values())
                + [self.transforms_path, self.report_path])

    def summary_text(self) -> str:
        """Return one line per fit: target, model, atoms, RMSD, SO, status."""
        lines = [f"{'target':<28}{'model':>6}{'atoms':>7}{'RMSD':>9}{'SO %':>8}  status"]
        for fit in self.fits:
            rmsd = f"{fit.rmsd:.3f}" if fit.rmsd is not None else "-"
            so = f"{fit.so:.1f}" if fit.so is not None else "-"
            line = f"{fit.target_name:<28}{fit.target_model:>6}{fit.n_atoms:>7}{rmsd:>9}{so:>8}  {fit.status}"
            if fit.failure_reasons:
                line += "  (" + "; ".join(fit.failure_reasons) + ")"
            lines.append(line)
        return "\n".join(lines)


def self_fit_job(structure: Structure, selected_ids: list, taken_names) -> FitJob:
    """Return a job that fits a copy of ``structure`` onto ``structure``.

    The copy is named like a duplicate input (``<stem>_<n><ext>``). Used
    for "fit all models" on a single input file: the selected model is the
    reference and every model of the copy is fitted onto it. Also usable when
    two atom sets come from the same file (pass the second set as the job's
    ``target_ids`` afterwards).
    """
    copy = structure.copy(make_unique_name(structure.name, taken_names))
    return FitJob(target=copy, reference_ids=list(selected_ids), target_ids=list(selected_ids))


def check_jobs(reference: Structure, jobs: list) -> list:
    """Run ``check_atom_pairs`` on every job and return the PairCheckResults."""
    return [check_atom_pairs(reference, job.target, job.reference_ids, job.target_ids)
            for job in jobs]


def _fit_job(reference: Structure, job: FitJob, so_cutoff: float, rmsd_cutoff: float,
             fit_all_models: bool) -> list:
    """Return the FitResults of one job (one per fitted model)."""
    if not fit_all_models:
        return [fit_pair(reference, job.target, job.reference_ids, job.target_ids,
                         so_cutoff, rmsd_cutoff)]
    fits = []
    for model_number in job.target.models():
        matched = job.target.equivalent_atoms(job.target_ids, model_number)
        missing = [m for m, atom in zip(job.target_ids, matched) if atom is None]
        if missing:
            reason = (f"model {model_number} lacks {len(missing)} selected atom(s), "
                      f"e.g. {job.target.atoms[missing[0]].label()}")
            fits.append(unfittable_result(reference, job.target, job.reference_ids,
                                          model_number, reason, so_cutoff, rmsd_cutoff))
            continue
        fits.append(fit_pair(reference, job.target, job.reference_ids,
                             [atom.matomid for atom in matched], so_cutoff, rmsd_cutoff))
    return fits


def _file_stem(name: str) -> str:
    """Return the name without extension, for output file names."""
    return os.path.splitext(name)[0]


def run_superposition(reference: Structure, jobs: list, out_dir: str,
                      so_cutoff: float = DEFAULT_SO_CUTOFF,
                      rmsd_cutoff: float = DEFAULT_RMSD_CUTOFF,
                      fit_all_models: bool = False, make_zip: bool = True) -> RunResult:
    """Check, fit and write every job; return a RunResult.

    Raises PipelineError (with the check messages) if any selection has
    errors; nothing is written in that case. Outputs in ``out_dir``:
    ``<stem>_fit.<ext>`` per target with at least one passing fit,
    ``<stem>__pairs.csv`` per target, ``transforms.csv``, ``report.txt``, and
    ``K_fit_results.zip`` (if ``make_zip``).
    """
    if not jobs:
        raise PipelineError("at least one target (a second set of atoms) is required")
    checks = check_jobs(reference, jobs)
    failed = [c for c in checks if not c.ok]
    if failed:
        raise PipelineError("atom selection errors:\n" + "\n".join(c.as_text() for c in failed))

    os.makedirs(out_dir, exist_ok=True)
    result = RunResult(reference=reference, targets=[j.target for j in jobs],
                       checks=checks, fits=[])
    for job in jobs:
        fits = _fit_job(reference, job, so_cutoff, rmsd_cutoff, fit_all_models)
        result.fits += fits
        target = job.target
        passed = {f.target_model: (f.rotation, f.translation) for f in fits if f.passed}
        out_path = None
        if passed:
            out_path = write_structure(target, passed,
                                       os.path.join(out_dir, output_file_name(target)))
        result.output_structures[target.name] = out_path
        result.pairs_files[target.name] = write_pairs_csv(
            reference, target, [f for f in fits if f.target_ids],
            os.path.join(out_dir, f"{_file_stem(target.name)}__pairs.csv"))

    result.transforms_path = write_transforms_csv(
        result.fits, os.path.join(out_dir, TRANSFORMS_NAME))
    result.report_path = write_report(
        reference, result.targets, result.fits, checks, so_cutoff, rmsd_cutoff,
        fit_all_models, result.output_structures, os.path.join(out_dir, REPORT_NAME))
    if make_zip:
        result.zip_path = zip_outputs(result.all_files(), os.path.join(out_dir, ZIP_NAME))
    return result
