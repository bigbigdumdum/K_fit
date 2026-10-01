# Copyright (C) 2026 Mukundan S
# SPDX-License-Identifier: MIT
"""Parse PDB and mmCIF files into K_fit ``Structure`` objects.

Parsing is done with Biopython (``Bio.PDB``). Each atom becomes an
``AtomRecord`` identified by its ``matomid`` = f"{model_number}-{atom_id}":

* ``model_number`` is the model number written in the file (``MODEL`` record
  in PDB, ``pdbx_PDB_model_num`` in mmCIF). A PDB file without ``MODEL``
  records is model 1. Biopython reports such files as model 0, and
  ``_model_number`` corrects that.
* ``atom_id`` is the atom serial number (PDB) or ``_atom_site.id`` (mmCIF).

Alternate locations (altlocs) are kept as separate atoms, so every altloc has
its own ``matomid``.

To support another file format, add its extension to ``FORMAT_BY_EXTENSION``
and a Biopython parser to ``_make_biopython_parser``. The writer module must
then also learn how to write that format.
"""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from .errors import KFitError

# File extension (lower case) -> K_fit format name.
FORMAT_BY_EXTENSION = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "cif",
    ".mmcif": "cif",
}

# Residue names treated as water in the summary (listed apart from ligands).
WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "D2O"}


class StructureError(KFitError):
    """Raised when a structure file cannot be read or is inconsistent."""


@dataclass
class AtomRecord:
    """One atom (one altloc) of a structure.

    Attributes are copied from Biopython at parse time so that the rest of
    K_fit never needs Biopython objects. ``coord`` is a numpy array of shape
    (3,), in Angstrom.
    """

    matomid: str
    model_number: int
    atom_id: int
    chain_id: str
    residue_name: str
    residue_number: int
    insertion_code: str
    is_hetero: bool
    atom_name: str
    element: str
    altloc: str
    occupancy: float
    coord: np.ndarray

    def residue_key(self) -> tuple:
        """Return (chain, residue number, insertion code, residue name).

        This identifies the residue within one model.
        """
        return (self.chain_id, self.residue_number, self.insertion_code,
                self.residue_name)

    def model_independent_key(self) -> tuple:
        """Return a key that finds the same atom in another model.

        Used by the "fit all models" option. Atom ids cannot be used for this,
        because PDB files restart the serial numbers in every model while
        mmCIF files keep counting.
        """
        return self.residue_key() + (self.atom_name, self.altloc)

    def label(self) -> str:
        """Return a readable label, e.g. ``A ALA 42 CA altloc B (occ 0.40) [1-245]``.

        Used in dropdowns, reports and error messages.
        """
        residue = f"{self.residue_number}{self.insertion_code.strip()}"
        text = f"{self.chain_id} {self.residue_name} {residue} {self.atom_name}"
        if self.altloc:
            text += f" altloc {self.altloc} (occ {self.occupancy:.2f})"
        return f"{text} [{self.matomid}]"


@dataclass
class StructureSummary:
    """Counts and hetero residues of one structure (Stage 3 input checks).

    ``hetero_residues`` holds (chain, residue name, residue number) for
    non-water hetero residues of the first model. ``water_count`` is the
    number of water residues in the first model.
    """

    name: str
    n_models: int
    n_atoms: int
    n_residues: int
    n_chains: int
    hetero_residues: list
    water_count: int

    def as_text(self) -> str:
        """Return the summary as printable lines of text."""
        lines = [
            f"{self.name}: {self.n_models} model(s), {self.n_chains} chain(s), "
            f"{self.n_residues} residue(s), {self.n_atoms} atom(s)",
        ]
        if self.hetero_residues:
            lines.append("  Hetero residues (name, chain, residue id):")
            for chain_id, name, number in self.hetero_residues:
                lines.append(f"    {name:>4}  chain {chain_id}  {number}")
        else:
            lines.append("  Hetero residues: none")
        lines.append(f"  Waters: {self.water_count}")
        return "\n".join(lines)


@dataclass
class Structure:
    """A parsed structure file plus everything needed to write it back.

    Attributes
    ----------
    name : unique display name, used as CSV header and output file stem.
        Usually the file name. Duplicates become ``<stem>_<n><ext>``.
    file_name : the original file name (without directory).
    file_format : "pdb" or "cif".
    source : "upload" or "pdb_id" (the input mode shown in the report).
    path : path of the input file on disk.
    sha256 : SHA-256 checksum of the input file bytes.
    raw_text : full text of the input file, used by the writer.
    atoms : dict matomid -> AtomRecord, in file order.
    """

    name: str
    file_name: str
    file_format: str
    source: str
    path: str
    sha256: str
    raw_text: str
    atoms: dict = field(default_factory=dict)

    # Navigation helpers (used by the Colab dropdowns)

    def models(self) -> list:
        """Return the model numbers in file order."""
        return list(dict.fromkeys(a.model_number for a in self.atoms.values()))

    def chains(self, model_number: int) -> list:
        """Return the chain ids of one model in file order."""
        return list(dict.fromkeys(
            a.chain_id for a in self.atoms.values()
            if a.model_number == model_number))

    def residues(self, model_number: int, chain_id: str) -> list:
        """Return residue keys (chain, number, icode, name) of one chain."""
        return list(dict.fromkeys(
            a.residue_key() for a in self.atoms.values()
            if a.model_number == model_number and a.chain_id == chain_id))

    def atoms_in_residue(self, model_number: int, residue_key: tuple) -> list:
        """Return the AtomRecords of one residue (all altlocs)."""
        return [a for a in self.atoms.values()
                if a.model_number == model_number
                and a.residue_key() == residue_key]

    def atoms_in_model(self, model_number: int) -> list:
        """Return all AtomRecords of one model."""
        return [a for a in self.atoms.values() if a.model_number == model_number]

    # Lookups

    def get_atom(self, matomid: str) -> AtomRecord:
        """Return the atom with this matomid or raise KeyError."""
        return self.atoms[matomid]

    def coords(self, matomids: list) -> np.ndarray:
        """Return an (N, 3) float array with the coordinates of ``matomids``."""
        return np.array([self.atoms[m].coord for m in matomids], dtype=float)

    def equivalent_atoms(self, matomids: list, model_number: int) -> list:
        """Return the atoms matching ``matomids`` in another model.

        Matching uses ``AtomRecord.model_independent_key`` (chain, residue,
        atom name, altloc). The result has one entry per input matomid:
        the matching AtomRecord, or None if that model lacks the atom.
        """
        index = {a.model_independent_key(): a for a in self.atoms_in_model(model_number)}
        return [index.get(self.atoms[m].model_independent_key()) for m in matomids]

    # Other helpers

    def summary(self) -> StructureSummary:
        """Count models, chains, residues and atoms and list hetero residues.

        Counts are for the first model. For a multi-model file the number of
        models is reported separately.
        """
        models = self.models()
        first = self.atoms_in_model(models[0]) if models else []
        residue_keys = list(dict.fromkeys(a.residue_key() for a in first))
        hetero = {}          # residue key -> (chain, name, number), file order
        water_keys = set()
        for atom in first:
            if not atom.is_hetero:
                continue
            key = atom.residue_key()
            if atom.residue_name in WATER_NAMES:
                water_keys.add(key)
                continue
            number = f"{atom.residue_number}{atom.insertion_code.strip()}"
            hetero.setdefault(key, (atom.chain_id, atom.residue_name, number))
        return StructureSummary(
            name=self.name,
            n_models=len(models),
            n_atoms=len(first),
            n_residues=len(residue_keys),
            n_chains=len(dict.fromkeys(a.chain_id for a in first)),
            hetero_residues=list(hetero.values()),
            water_count=len(water_keys),
        )

    def copy(self, new_name: str) -> "Structure":
        """Return an independent copy with a new display name.

        Used when two atom sets come from the same input file: the copy is
        handled like a separate input.
        """
        clone = copy.deepcopy(self)
        clone.name = new_name
        return clone


def detect_format(path: str) -> str:
    """Return "pdb" or "cif" for a structure file.

    The extension is used first. For unknown extensions the content is
    checked: mmCIF files start with a ``data_`` block.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in FORMAT_BY_EXTENSION:
        return FORMAT_BY_EXTENSION[ext]
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                return "cif" if line.lstrip().startswith("data_") else "pdb"
    raise StructureError(f"{path}: file is empty")


def sha256_of_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_unique_name(name: str, taken) -> str:
    """Return ``name`` or, if already taken, ``<stem>_<n><ext>`` with the lowest free n.

    Example: "model.pdb" -> "model_1.pdb" -> "model_2.pdb".
    ``taken`` is any container of names already in use.
    """
    if name not in taken:
        return name
    stem, ext = os.path.splitext(name)
    n = 1
    while f"{stem}_{n}{ext}" in taken:
        n += 1
    return f"{stem}_{n}{ext}"


def _make_biopython_parser(file_format: str):
    """Return a quiet Biopython parser for "pdb" or "cif"."""
    if file_format == "pdb":
        return PDBParser(QUIET=True)
    if file_format == "cif":
        return MMCIFParser(QUIET=True)
    raise StructureError(f"unsupported format: {file_format}")


def _model_number(bio_model) -> int:
    """Return the model number as written in the file.

    Biopython's ``serial_num`` is 0 for PDB files without MODEL records;
    K_fit calls that model 1.
    """
    serial = getattr(bio_model, "serial_num", None)
    return serial if serial else 1


def load_structure(path: str, source: str = "upload", name: str | None = None) -> Structure:
    """Parse a PDB or mmCIF file and return a ``Structure``.

    Parameters
    ----------
    path : file to read.
    source : "upload" or "pdb_id". Stored for the report only.
    name : display name. Defaults to the file name. Callers loading several
        files should pass a name from ``make_unique_name``.

    Raises StructureError if the file cannot be parsed, contains no atoms or
    contains the same matomid twice.
    """
    file_format = detect_format(path)
    file_name = os.path.basename(path)
    with open(path, encoding="utf-8", errors="replace") as handle:
        raw_text = handle.read()

    try:
        bio_structure = _make_biopython_parser(file_format).get_structure(
            file_name, path)
    except (PDBConstructionException, ValueError, KeyError) as err:
        raise StructureError(f"{file_name}: Biopython could not parse the file ({err})") from err

    atoms = {}
    for bio_model in bio_structure:
        model_number = _model_number(bio_model)
        for chain in bio_model:
            # get_unpacked_list() expands disordered residues and atoms, so
            # every altloc (and point mutation) is kept.
            for residue in chain.get_unpacked_list():
                het_flag, res_number, icode = residue.id
                for bio_atom in residue.get_unpacked_list():
                    atom_id = bio_atom.get_serial_number()
                    matomid = f"{model_number}-{atom_id}"
                    if matomid in atoms:
                        raise StructureError(
                            f"{file_name}: duplicate matomid {matomid}; atom ids must be "
                            "unique within a model")
                    altloc = bio_atom.get_altloc().strip()
                    occupancy = bio_atom.get_occupancy()
                    atoms[matomid] = AtomRecord(
                        matomid=matomid,
                        model_number=model_number,
                        atom_id=atom_id,
                        chain_id=chain.id,
                        residue_name=residue.get_resname(),
                        residue_number=res_number,
                        insertion_code=icode,
                        is_hetero=het_flag != " ",
                        atom_name=bio_atom.get_name(),
                        element=(bio_atom.element or "").strip().upper(),
                        altloc=altloc,
                        occupancy=float(occupancy) if occupancy is not None else 1.0,
                        coord=np.array(bio_atom.get_coord(), dtype=float),
                    )
    if not atoms:
        raise StructureError(f"{file_name}: no atoms found")

    return Structure(
        name=name or file_name,
        file_name=file_name,
        file_format=file_format,
        source=source,
        path=os.path.abspath(path),
        sha256=sha256_of_file(path),
        raw_text=raw_text,
        atoms=atoms,
    )
