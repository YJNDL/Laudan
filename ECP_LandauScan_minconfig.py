#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landau-mode guided subphase scanner (with robust Plan-C auto-fill).

What this script does (high level):
- Read a parent POSCAR/CONTCAR (VASP) structure.
- Use spglib to:
  1) identify the parent space group (number),
  2) obtain symmetry operations for the given cell (NO axis swapping / no standardization).
- Build a mechanical representation D(g) on the 3N displacement space.
- For parent point group mm2 (C2v), build projector operators for irreps A1/A2/B1/B2.
- Optionally split each irrep sector into parity sectors (@G/@R) using a detected
  non-primitive centering translation (e.g., A/B/C/F/I-centering).
- Scan:
  - single-mode amplitudes α_i + strains (ε_a, ε_b, ε_c)
  - multi-mode combinations using ρ * OP-direction + strains
- "Plan-C": if USE_ISOTROPY_CONFIRMED_COMBOS=True but the list is empty, the script
  will try to auto-fill combos by a robust brute-force confirmation (NO crash).
  If Plan-C still finds nothing, it automatically falls back to auto-generated combos.

This script is designed to NOT crash in the scenario shown in your log:
- spgrep_modulation IsotropyEnumerator failed ("Unreachable!") so combos remained empty
- your old code raised RuntimeError and aborted
Here we: warn -> fallback -> continue scanning.

Dependencies:
- numpy (required)
- spglib (strongly recommended / required for symmetry ops and SG id)

Optional:
- spgrep, spgrep-modulation: NOT required in this robust version (we don't depend on
  IsotropyEnumerator; we do a brute-force "confirmation" Plan-C instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable, Set, Union

import numpy as np
import re


# =========================
# ECP program identity
# =========================
# A friendly name that echoes the Exfoliating Crystallographic Plane (ECP) program.
PROGRAM_NAME = "ECP-LandauScan"
PROGRAM_TAG = "ECP-LS"
PROGRAM_VERSION = "0.7.0"


# =========================
# User configuration region
# =========================
"""Quick-start: edit only the few knobs in this block.

Design goal:
- Keep the public configuration surface small and stable.
- Hide rarely-touched knobs as internal defaults further down.

If you need full control (custom grids / direction pools / Plan-C details),
edit the internal defaults in the section below (rarely necessary).
"""

# ---- Required input ----
PARENT_POSCAR = "GaSb_39.vasp"   # POSCAR/CONTCAR path

# ---- Output ----
OUTPUT_DIR = "ECP_LandauScan_outputs"
SAVE_POLICY = "all"  # "all" (recommended) or "hits"
MAX_STRUCTURES_PER_SG: Optional[int] = None  # e.g. 50; None -> unlimited

# Create a unique subfolder per run (recommended; prevents overwriting).
CREATE_RUN_SUBDIR = True
RUN_TAG: Optional[str] = None  # optional label in run folder name

# ---- Symmetry tolerances (Å) ----
SYMPREC_PARENT = 1e-3      # parent symmetry detection / ops
SYMPREC_IDENTIFY = 2e-2    # identify distorted structures (looser)
ANGLE_TOLERANCE = -1.0     # spglib default if negative

# ---- Scan preset ----
# fast: quick sanity check; standard: default; dense: heavier scan
SCAN_PROFILE = "standard"  # "fast" | "standard" | "dense"

# ---- Optional: target-driven combo confirmation (Plan‑C) ----
# Leave empty to explore broadly (no target).
TARGET_SGS: List[int] = []  # e.g. [26]

# ---- Optional: manually confirmed combos ----
# If non-empty, we will ONLY scan these combos (most reproducible).
# Format:
# {
#   "target_sgs": {26},                      # set[int], optional
#   "irrep_modes": [("A1@R", 0), ...],       # list[tuple[str,int]]
#   "op_directions": [[1,1], [1,-1], ...],   # list[list[float]] each inner length = len(irrep_modes)
# }
CONFIRMED_COMBOS: List[Dict[str, Any]] = [
    # Example:
    # {"target_sgs": {26}, "irrep_modes": [("A2@R", 0)], "op_directions": [[1.0]]},
]

# ---- Optional: parent symmetry hints / safety checks ----
# (strongly recommended: keep point-group hint as None and let spglib decide)
PARENT_POINT_GROUP_HINT: Optional[str] = None
PARENT_LAYER_GROUP_NO_HINT: Optional[int] = None   # 1–80
PARENT_SPACE_GROUP_NO_HINT: Optional[int] = None   # 1–230
STRICT_POINTGROUP_MATCH = True
ENABLE_PARITY_SECTORS = True


# =========================
# Internal defaults (rarely edited)
# =========================

_SCAN_PROFILES = {
    "fast": {
        "AMP_GRID": [-0.6, 0.6],
        "RHO_GRID": [0.6, 0.9, 1.2],
        "STRAIN_A_GRID": [1.0],
        "STRAIN_B_GRID": [1.0],
        "STRAIN_C_GRID": [1.0],
        "N_RANDOM_DIRS": 0,
    },
    "standard": {
        "AMP_GRID": [-0.8, -0.4, 0.4, 0.8],
        "RHO_GRID": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        "STRAIN_A_GRID": [float(x) for x in np.round(np.arange(0.80, 0.91, 0.02), 4)],
        "STRAIN_B_GRID": [float(x) for x in np.round(np.arange(0.88, 0.97, 0.02), 4)],
        "STRAIN_C_GRID": [float(x) for x in np.round(np.arange(1.00, 1.04, 0.01), 4)],
        "N_RANDOM_DIRS": 30,
    },
    "dense": {
        "AMP_GRID": [-1.0, -0.8, -0.6, -0.4, 0.4, 0.6, 0.8, 1.0],
        "RHO_GRID": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2],
        "STRAIN_A_GRID": [float(x) for x in np.round(np.arange(0.78, 0.93, 0.01), 4)],
        "STRAIN_B_GRID": [float(x) for x in np.round(np.arange(0.86, 0.99, 0.01), 4)],
        "STRAIN_C_GRID": [float(x) for x in np.round(np.arange(0.98, 1.06, 0.01), 4)],
        "N_RANDOM_DIRS": 60,
    },
}

if SCAN_PROFILE not in _SCAN_PROFILES:
    raise ValueError(
        f"Unknown SCAN_PROFILE='{SCAN_PROFILE}'. "
        f"Choose from {sorted(_SCAN_PROFILES.keys())}"
    )

# Apply profile -> internal variables
AMP_GRID = _SCAN_PROFILES[SCAN_PROFILE]["AMP_GRID"]
RHO_GRID = _SCAN_PROFILES[SCAN_PROFILE]["RHO_GRID"]
STRAIN_A_GRID = _SCAN_PROFILES[SCAN_PROFILE]["STRAIN_A_GRID"]
STRAIN_B_GRID = _SCAN_PROFILES[SCAN_PROFILE]["STRAIN_B_GRID"]
STRAIN_C_GRID = _SCAN_PROFILES[SCAN_PROFILE]["STRAIN_C_GRID"]

# Random OP direction sampling (optional; 0 disables)
ENABLE_RANDOM_OP_DIRECTIONS = _SCAN_PROFILES[SCAN_PROFILE]["N_RANDOM_DIRS"] > 0
N_RANDOM_OP_DIRECTIONS_PER_COMBO = int(_SCAN_PROFILES[SCAN_PROFILE]["N_RANDOM_DIRS"])
RANDOM_OP_DIRECTION_SEED = 12345

# Kept for backward compatibility (internal algorithm may use this as a pool size)
RANDOM_OP_DIRECTION_CANDIDATES = max(500, 50 * max(1, N_RANDOM_OP_DIRECTIONS_PER_COMBO))

# Runtime: old/new irrep label aliases (for backward-compatible combos)
IRREP_LABEL_ALIASES: Dict[str, str] = {}

# Combo policy unification:
# - If CONFIRMED_COMBOS is non-empty -> scan only confirmed combos.
# - Else, if TARGET_SGS non-empty -> run Plan‑C confirmation to auto-fill combos.
# - Else -> fall back to auto-generated combos.
USE_ISOTROPY_CONFIRMED_COMBOS = True
ISOTROPY_CONFIRMED_COMBOS = CONFIRMED_COMBOS
PLAN_C_TARGET_SGS = TARGET_SGS
ENABLE_PLAN_C_AUTOFILL_COMBOS = (len(ISOTROPY_CONFIRMED_COMBOS) == 0 and len(PLAN_C_TARGET_SGS) > 0)

PLAN_C_TEST_RHO = 0.6
PLAN_C_MAX_COMBO_SIZE = 2
PLAN_C_DIR_POOL_2D = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0, -1.0],
]

AUTO_COMBO_MAX_SIZE = 2
AUTO_DIR_POOL_2D = [
    [1.0, 1.0],
    [1.0, -1.0],
]

# Structure writing policy
WRITE_ALL_TRIAL_STRUCTURES = (SAVE_POLICY.lower() == "all")
WRITE_HIT_STRUCTURES = True
WRITE_LIMIT_PER_SG = MAX_STRUCTURES_PER_SG
WRITE_RESULTS_CSV = True
# Utility: Periodic table
# =========================

_PERIODIC_TABLE = [
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
    "Nh","Fl","Mc","Lv","Ts","Og",
]
SYMBOL_TO_Z = {sym: i + 1 for i, sym in enumerate(_PERIODIC_TABLE)}


# =========================
# Data structures
# =========================

@dataclass(frozen=True)
class Crystal:
    """Simple crystal container in spglib conventions.

    lattice: (3,3) with ROWS being a,b,c lattice vectors in Cartesian (Å)
    frac: (N,3) fractional coordinates in [0,1)
    numbers: (N,) atomic numbers (species)
    symbols: element symbols list (optional, for writing POSCAR)
    """
    lattice: np.ndarray
    frac: np.ndarray
    numbers: np.ndarray
    symbols: List[str]

    @property
    def nsites(self) -> int:
        return int(self.frac.shape[0])

    def cart_coords(self) -> np.ndarray:
        return self.frac @ self.lattice

    def to_spglib_cell(self):
        return (np.array(self.lattice, float), np.array(self.frac, float), np.array(self.numbers, int))


# =========================
# VASP POSCAR IO
# =========================

def _is_all_int(tokens: List[str]) -> bool:
    if not tokens:
        return False
    for t in tokens:
        try:
            int(t)
        except Exception:
            return False
    return True


def read_poscar(path: str | Path) -> Crystal:
    path = Path(path)
    txt = path.read_text().splitlines()
    if len(txt) < 8:
        raise ValueError(f"POSCAR too short: {path}")

    comment = txt[0].strip()
    scale = float(txt[1].split()[0])

    lattice = np.array([[float(x) for x in txt[i].split()[:3]] for i in range(2, 5)], float) * scale

    line5 = txt[5].split()
    if _is_all_int(line5):
        # VASP4 style: counts immediately
        symbols = [f"X{i+1}" for i in range(len(line5))]
        counts = [int(x) for x in line5]
        idx = 6
    else:
        symbols = line5
        counts = [int(x) for x in txt[6].split()]
        idx = 7

    # Optional "Selective dynamics"
    if txt[idx].strip().lower().startswith("s"):
        idx += 1

    coord_type = txt[idx].strip().lower()
    direct = coord_type.startswith("d")
    cartesian = coord_type.startswith("c") or coord_type.startswith("k")
    if not (direct or cartesian):
        raise ValueError(f"Cannot parse coordinate type line: '{txt[idx]}' in {path}")
    idx += 1

    n = sum(counts)
    pos_lines = txt[idx: idx + n]
    if len(pos_lines) < n:
        raise ValueError(f"Not enough coordinate lines in POSCAR: need {n}, got {len(pos_lines)}")

    coords = np.array([[float(x) for x in ln.split()[:3]] for ln in pos_lines], float)
    if direct:
        frac = coords % 1.0
    else:
        # cart -> frac using row-lattice convention: cart = frac @ lattice  => frac = cart @ inv(lattice)
        frac = coords @ np.linalg.inv(lattice)
        frac = frac % 1.0

    # atomic numbers
    numbers = []
    for sym, c in zip(symbols, counts):
        z = SYMBOL_TO_Z.get(sym, None)
        if z is None:
            raise ValueError(f"Unknown element symbol '{sym}' in POSCAR. Add it to periodic table list.")
        numbers.extend([z] * c)

    return Crystal(lattice=lattice, frac=frac, numbers=np.array(numbers, int), symbols=symbols)


def write_poscar(crys: Crystal, path: str | Path, comment: str = "generated by landau scan") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build symbol->count in the same order as crys.symbols if possible
    # If crys.symbols doesn't match counts, fall back to Z grouping.
    symbols = crys.symbols if crys.symbols else []
    if symbols:
        # Count by provided symbol order (assumes same species ordering as input POSCAR)
        z_by_sym = [SYMBOL_TO_Z[s] for s in symbols]
        counts = [int(np.sum(crys.numbers == z)) for z in z_by_sym]
    else:
        # group by Z
        uniq = sorted(set(int(z) for z in crys.numbers))
        symbols = [_PERIODIC_TABLE[z - 1] for z in uniq]
        counts = [int(np.sum(crys.numbers == z)) for z in uniq]

    lines = []
    lines.append(comment)
    lines.append("1.0")
    for v in crys.lattice:
        lines.append(f"{v[0]:16.10f} {v[1]:16.10f} {v[2]:16.10f}")
    lines.append(" ".join(symbols))
    lines.append(" ".join(str(c) for c in counts))
    lines.append("Direct")
    for f in crys.frac % 1.0:
        lines.append(f"{f[0]:16.10f} {f[1]:16.10f} {f[2]:16.10f}")
    path.write_text("\n".join(lines) + "\n")


# =========================
# spglib helpers
# =========================

def _require_spglib():
    try:
        import spglib  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "This script requires spglib for symmetry operations and SG identification.\n"
            "Install it with: pip install spglib"
        ) from e


def get_spglib_dataset(crys: Crystal, symprec: float, angle_tolerance: float):
    _require_spglib()
    import spglib  # type: ignore

    cell = crys.to_spglib_cell()
    if angle_tolerance is not None and angle_tolerance > 0:
        ds = spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance)
    else:
        ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return ds


def dataset_get(ds: Any, key: str, default=None):
    # spglib 2.x returns an object with attributes; older returns dict
    if isinstance(ds, dict):
        return ds.get(key, default)
    return getattr(ds, key, default)


def identify_spacegroup(crys: Crystal, symprec: float, angle_tolerance: float) -> Tuple[int, Optional[str], Optional[str]]:
    ds = get_spglib_dataset(crys, symprec=symprec, angle_tolerance=angle_tolerance)
    sg_num = int(dataset_get(ds, "number", -1))
    sg_symbol = dataset_get(ds, "international", None)
    pg = dataset_get(ds, "pointgroup", None)
    return sg_num, sg_symbol, pg


def get_symmetry_ops(crys: Crystal, symprec: float, angle_tolerance: float):
    _require_spglib()
    import spglib  # type: ignore

    cell = crys.to_spglib_cell()
    if angle_tolerance is not None and angle_tolerance > 0:
        symm = spglib.get_symmetry(cell, symprec=symprec, angle_tolerance=angle_tolerance)
    else:
        symm = spglib.get_symmetry(cell, symprec=symprec)
    rotations = np.array(symm["rotations"], int)
    translations = np.array(symm["translations"], float)
    return rotations, translations


# =========================
# Symmetry representation on 3N displacements
# =========================

def wrap_frac(x: np.ndarray) -> np.ndarray:
    """Wrap fractional coords to [0,1)."""
    return x - np.floor(x)


def wrap_diff(d: np.ndarray) -> np.ndarray:
    """Wrap a fractional difference to [-0.5,0.5)."""
    return d - np.rint(d)

def wrap_to_mhalf_half(x: np.ndarray) -> np.ndarray:
    """Wrap a fractional vector to [-0.5,0.5) component-wise."""
    return x - np.rint(x)



def find_permutation(crys: Crystal, R: np.ndarray, t: np.ndarray, map_tol: float) -> np.ndarray:
    """Find permutation perm such that atom i maps to atom perm[i] under {R|t}."""
    frac = crys.frac
    lattice = crys.lattice
    Z = crys.numbers
    N = crys.nsites

    # Apply operation to fractional (row-vector form): f' = f @ R^T + t
    mapped = wrap_frac(frac @ R.T + t)

    used: Set[int] = set()
    perm = np.empty(N, dtype=int)

    for i in range(N):
        best_j = None
        best_dist = 1e30
        for j in range(N):
            if j in used:
                continue
            if Z[j] != Z[i]:
                continue
            df = wrap_diff(frac[j] - mapped[i])
            dist = float(np.linalg.norm(df @ lattice))
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is None or best_dist > map_tol:
            # Debug hint: loosen map_tol, or check structure symmetry / symprec.
            raise RuntimeError(
                f"Cannot find atom mapping for i={i} under operation.\n"
                f"best_dist={best_dist:.3e} (map_tol={map_tol:.3e}).\n"
                f"Try increasing SYMPREC_PARENT or using a conventional cell."
            )
        perm[i] = int(best_j)
        used.add(int(best_j))

    return perm


def rotation_cartesian(lattice: np.ndarray, R_frac: np.ndarray) -> np.ndarray:
    """Convert a fractional rotation (3x3 int) to Cartesian rotation (3x3 float).

    spglib convention: lattice rows are a,b,c vectors.
    Use column-vector formula with A = lattice^T (columns are lattice vectors):
        R_cart = A * R_frac * A^{-1}
    """
    A = lattice.T
    A_inv = np.linalg.inv(A)
    return A @ R_frac @ A_inv


def build_representation_matrix(crys: Crystal, R: np.ndarray, t: np.ndarray, map_tol: float) -> np.ndarray:
    """Build 3N x 3N representation matrix D({R|t}) acting on Cartesian displacements (column vector)."""
    N = crys.nsites
    dim = 3 * N
    D = np.zeros((dim, dim), dtype=float)

    perm = find_permutation(crys, R, t, map_tol=map_tol)
    R_cart = rotation_cartesian(crys.lattice, R)

    for i in range(N):
        j = int(perm[i])
        D[3 * j: 3 * j + 3, 3 * i: 3 * i + 3] = R_cart

    return D


# =========================
# Point group mm2 (C2v) characters and operation classification
# =========================

def classify_mm2_op(R: np.ndarray) -> str:
    """Classify a rotation matrix into one of {'E','C2','Mx','My'} for mm2.

    Assumes conventional orthorhombic-like diagonal ±1 matrices.
    Falls back to determinant/trace if not diagonal.
    """
    R = np.array(R, dtype=int)
    I = np.eye(3, dtype=int)
    if np.array_equal(R, I):
        return "E"

    det = int(round(np.linalg.det(R)))
    tr = int(np.trace(R))

    # 180° rotation: det=+1, tr=-1
    if det == 1 and tr == -1:
        return "C2"

    # mirror: det=-1, tr=+1
    if det == -1 and tr == 1:
        # Diagonal fast-path:
        if np.array_equal(R, np.diag([-1, 1, 1])):
            return "Mx"  # mirror plane normal to x (yz plane)
        if np.array_equal(R, np.diag([1, -1, 1])):
            return "My"  # mirror plane normal to y (xz plane)
        # If it's not purely diagonal, guess by the eigenvector with eigenvalue -1.
        w, v = np.linalg.eig(R.astype(float))
        # Find eigenvalue closest to -1
        idx = int(np.argmin(np.abs(w + 1.0)))
        n = np.real(v[:, idx])
        # Choose axis by largest component
        k = int(np.argmax(np.abs(n)))
        return "Mx" if k == 0 else ("My" if k == 1 else "Mz")

    raise RuntimeError(f"Rotation not compatible with mm2 classification: det={det}, trace={tr}, R=\n{R}")


MM2_CHAR_TABLE: Dict[str, Dict[str, int]] = {
    # Operation order: E, C2(z), σ(yz)=Mx, σ(xz)=My
    "A1": {"E": 1, "C2": 1, "Mx": 1, "My": 1},
    "A2": {"E": 1, "C2": 1, "Mx": -1, "My": -1},
    "B1": {"E": 1, "C2": -1, "Mx": 1, "My": -1},
    "B2": {"E": 1, "C2": -1, "Mx": -1, "My": 1},
}


# =========================
# Parity translation detection (centering translation)
# =========================

def detect_centering_translation(rotations: np.ndarray, translations: np.ndarray, tol: float = 1e-3) -> Optional[np.ndarray]:
    """Detect a non-primitive centering translation tau (e.g. A/B/C/I/F centering).

    Why we need this
    ---------------
    For a centered lattice, spglib symmetry operations typically appear in pairs that differ by
    a centering translation tau:

        (R|t) and (R|t + tau)

    If we naively build point-group irrep projectors by summing over *all* space-group operations,
    the paired sum introduces a factor (I + D_tau) and **kills the odd-parity sector**.

    This function returns tau so we can explicitly build parity projectors:
        P_G = (I + D_tau)/2,  P_R = (I - D_tau)/2

    Strategy
    --------
    1) First try to find a *pure translation* element: R = I but t is non-integer.
    2) If not found, fall back to translation *differences* among operations with the same rotation.

    Returns
    -------
    tau in fractional coordinates wrapped to [0,1), or None if not detected.
    """
    I = np.eye(3, dtype=int)

    def _is_nontrivial(v: np.ndarray) -> bool:
        w = wrap_to_mhalf_half(v)
        n = float(np.linalg.norm(w))
        return (n > tol) and (n < 1.0 - tol)

    # --- (1) pure translations ---
    pure = []
    for R, t in zip(rotations, translations):
        if np.array_equal(R, I) and _is_nontrivial(t):
            pure.append(wrap_frac(t))

    if pure:
        # pick the one with smallest |wrap_to_mhalf_half| norm (usually the centering translation)
        pure.sort(key=lambda v: float(np.linalg.norm(wrap_to_mhalf_half(v))))
        tau = pure[0]
        # canonicalize small numerical noise
        tau = np.where(np.isclose(tau, 0.0, atol=tol), 0.0, tau)
        tau = np.where(np.isclose(tau, 1.0, atol=tol), 0.0, tau)
        return wrap_frac(tau)

    # --- (2) fallback: diffs among same-rotation operations ---
    # group translations by rotation key
    by_rot = {}
    for R, t in zip(rotations, translations):
        key = tuple(int(x) for x in R.flatten())
        by_rot.setdefault(key, []).append(wrap_frac(t))

    cand = []
    for ts in by_rot.values():
        if len(ts) < 2:
            continue
        # compare all pairs (usually 2)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                dt = wrap_frac(ts[j] - ts[i])
                if _is_nontrivial(dt):
                    cand.append(dt)

    if not cand:
        return None

    # cluster by rounding
    from collections import Counter
    keys = [tuple(np.round(wrap_frac(v), 6)) for v in cand]
    most_key, _ = Counter(keys).most_common(1)[0]
    tau = np.array(most_key, float)
    tau = np.where(np.isclose(tau, 0.0, atol=tol), 0.0, tau)
    tau = np.where(np.isclose(tau, 1.0, atol=tol), 0.0, tau)
    return wrap_frac(tau)


def select_representative_ops_by_rotation(rotations: np.ndarray, translations: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Select ONE representative (R|t) per unique rotation matrix.

    This is critical for generating the @R (odd) sector:
    ---------------------------------------------------
    If the lattice is centered, operations come in pairs differing by a centering translation tau.
    Summing over both in the projector introduces (I + D_tau) and annihilates odd parity.

    Therefore, for the *point-group* projector we intentionally keep only one translation coset
    for each rotation.

    Returns a deterministic list. For mm2 we sort by (E, C2, Mx, My) for readability;
    for other point groups we fall back to a stable matrix-key sort.
    """
    reps = {}

    def score(t: np.ndarray) -> Tuple[float, float, float]:
        w = wrap_to_mhalf_half(t)
        return (float(np.linalg.norm(w)), float(np.sum(np.abs(w))), float(np.max(np.abs(w))))

    for R, t in zip(rotations, translations):
        key = tuple(int(x) for x in R.flatten())
        s = score(t)
        if (key not in reps) or (s < reps[key][0]):
            reps[key] = (s, R.copy(), wrap_frac(t))

    ops = [(v[1], v[2]) for v in reps.values()]

    def _rot_key(R: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(x) for x in np.array(R, dtype=int).flatten())

    # stable order for readability (mm2 if possible, otherwise deterministic by matrix key)
    I = np.eye(3, dtype=int)
    try:
        order = {"E": 0, "C2": 1, "Mx": 2, "My": 3, "Mz": 4}
        ops.sort(key=lambda Rt: order.get(classify_mm2_op(Rt[0]), 99))
    except Exception:
        ops.sort(key=lambda Rt: (0 if np.array_equal(np.array(Rt[0], dtype=int), I) else 1, _rot_key(Rt[0])))
    return ops


def basis_from_projector(P: np.ndarray, eig_threshold: float = 0.5) -> List[np.ndarray]:
    """Extract an orthonormal basis (as vectors) from a (near-)projector matrix."""
    # Ensure symmetric
    P = 0.5 * (P + P.T)
    w, v = np.linalg.eigh(P)
    idxs = [i for i, val in enumerate(w) if val > eig_threshold]
    # Return eigenvectors associated with large eigenvalues
    return [v[:, i].copy() for i in idxs]




def normalize_mode(vec, nsites: int, eps: float = 1e-12):
    """Remove rigid translation and normalize a 3N displacement vector.

    Parameters
    ----------
    vec
        (3N,) displacement vector in Cartesian coordinates (Å).
    nsites
        Number of sites N.

    Returns
    -------
    (3N,) numpy array
        Translation-removed and normalized mode vector. The normalization is
        chosen so that the *maximum* atomic displacement magnitude equals 1 Å.

    Why max-normalization?
    ----------------------
    With L2-normalization, amplitudes (e.g. 0.4, 0.8) get "diluted" when N is
    not small, making distortions too tiny to break the centering translation.
    Max-normalization makes the scan amplitude directly correspond to a typical
    per-atom displacement scale.
    """
    v = (np.array(vec, dtype=float).reshape((int(nsites), 3))).copy()

    # Remove rigid translation component
    v -= np.mean(v, axis=0, keepdims=True)

    norms = np.linalg.norm(v, axis=1)
    maxn = float(np.max(norms)) if norms.size else 0.0
    if maxn < eps:
        return np.zeros((3 * int(nsites),), dtype=float)

    v /= maxn

    # Deterministic sign: make first non-trivial component positive
    flat = v.reshape(-1)
    for x in flat:
        if abs(x) > eps:
            if x < 0:
                flat = -flat
            break
    return flat
def build_landau_basis_mm2(
    crys: Crystal,
    rotations: np.ndarray,
    translations: np.ndarray,
    symprec_map: float,
    enable_parity: bool = True,
) -> Dict[str, List[np.ndarray]]:
    """Build Landau mode basis grouped by (irrep @ parity)."""
    N = crys.nsites
    dim = 3 * N
    map_tol = max(symprec_map * 5.0, 1e-5)

    # Build representation matrices for POINT-GROUP operations (unique rotations only).
    # IMPORTANT: do NOT sum over both centering-related translation cosets here,
    # otherwise (I + D_tau) gets baked into the projector and @R becomes identically empty.
    rep_ops = select_representative_ops_by_rotation(rotations, translations)

    D_ops: List[np.ndarray] = []
    op_classes: List[str] = []
    for R, t in rep_ops:
        D_ops.append(build_representation_matrix(crys, R, t, map_tol=map_tol))
        op_classes.append(classify_mm2_op(R))

    # For diagnostics
    if len(rep_ops) != len(rotations):
        print(f"[INFO] spglib ops: {len(rotations)} (space-group), unique rotations used for projector: {len(rep_ops)}")

    # Detect centering translation and build parity projectors
    parity_projectors: Dict[str, np.ndarray] = {"G": np.eye(dim)}
    tau = None
    if enable_parity:
        tau = detect_centering_translation(rotations, translations)
        if tau is not None:
            # Build D(tau) as a pure translation operator
            D_tau = build_representation_matrix(crys, np.eye(3, dtype=int), tau, map_tol=map_tol)
            parity_projectors["G"] = 0.5 * (np.eye(dim) + D_tau)
            parity_projectors["R"] = 0.5 * (np.eye(dim) - D_tau)

    landau_basis: Dict[str, List[np.ndarray]] = {}

    for irrep, chars in MM2_CHAR_TABLE.items():
        P_ir = np.zeros((dim, dim), dtype=float)
        for D, cls in zip(D_ops, op_classes):
            chi = float(chars.get(cls, 0))
            P_ir += chi * D
        P_ir /= float(len(D_ops))

        for ptag, Pp in parity_projectors.items():
            # Combined projector: restrict to parity sector
            P = Pp @ P_ir @ Pp
            basis = basis_from_projector(P, eig_threshold=0.5)
            basis = [normalize_mode(b, nsites=N) for b in basis]
            key = f"{irrep}@{ptag}"
            if basis:
                landau_basis[key] = basis

    # If parity split was requested but R is empty, still keep only @G.
    return landau_basis


# =========================
# Generic point-group decomposition (no character table)
# =========================

def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _cluster_sorted_eigenvalues(w: np.ndarray, tol: float) -> List[List[int]]:
    """Cluster *sorted* eigenvalues into groups within tolerance."""
    clusters: List[List[int]] = []
    if w.size == 0:
        return clusters
    cur = [0]
    w0 = float(w[0])
    for i in range(1, int(w.size)):
        if abs(float(w[i]) - w0) <= tol:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
            w0 = float(w[i])
    clusters.append(cur)
    return clusters


def _round_character(x: float, atol: float = 1e-6, ndigits: int = 6) -> Union[float, int]:
    """Round traces that should be near small integers (0, ±1, ±2, …).

    We keep them as int when close, to stabilize signature comparisons.
    """
    r = float(np.rint(float(x)))
    if abs(float(x) - r) <= float(atol):
        return int(r)
    return float(np.round(float(x), int(ndigits)))


def _subspace_character_signature(
    V: np.ndarray,
    D_ops: List[np.ndarray],
    atol: float = 1e-6,
    ndigits: int = 6,
) -> Tuple[Union[float, int], ...]:
    """Character signature χ(g)=Tr(V^T D(g) V) across all group elements."""
    sig: List[Union[float, int]] = []
    for D in D_ops:
        M = V.T @ D @ V
        tr = float(np.trace(M))
        sig.append(_round_character(tr, atol=atol, ndigits=ndigits))
    return tuple(sig)


def _decompose_by_commutant(
    D_ops: List[np.ndarray],
    seed: int = 20250101,
    eig_cluster_tol: float = 1e-6,
    char_round_atol: float = 1e-6,
    char_round_ndigits: int = 6,
) -> Dict[Tuple[Union[float, int], ...], List[np.ndarray]]:
    """Return {signature: [V_block, ...]} where each V_block is an (n×d) basis.

    This avoids hard-coding character tables:
    1) Build a random symmetric matrix M.
    2) Average it over the group: A = Σ D(g) M D(g)^T.
       A commutes with all D(g).
    3) Eigen-decompose A; each eigenspace is (with high probability) an irrep copy.
    4) Group copies by their character signature.
    """
    if not D_ops:
        return {}
    n = int(D_ops[0].shape[0])
    rng = np.random.default_rng(int(seed))
    M = rng.standard_normal((n, n))
    M = _symmetrize(M)

    A = np.zeros((n, n), dtype=float)
    for D in D_ops:
        A += D @ M @ D.T
    A /= float(len(D_ops))
    A = _symmetrize(A)

    w, V = np.linalg.eigh(A)
    # tolerance scaled by spectral magnitude
    w_scale = float(np.max(np.abs(w))) if w.size else 1.0
    tol = max(float(eig_cluster_tol), float(eig_cluster_tol) * w_scale)

    clusters = _cluster_sorted_eigenvalues(w, tol=tol)
    by_sig: Dict[Tuple[Union[float, int], ...], List[np.ndarray]] = {}
    for idxs in clusters:
        block = V[:, idxs]
        sig = _subspace_character_signature(block, D_ops, atol=char_round_atol, ndigits=char_round_ndigits)
        by_sig.setdefault(sig, []).append(block)
    return by_sig


# =========================
# Mulliken irrep labelling utilities (for layer-group axial point groups)
# =========================

# 80 layer groups only realize these 27 axial crystallographic point groups.
AXIAL_POINT_GROUPS_27: Set[str] = {
    "1", "-1",
    "2", "m", "2/m",
    "222", "mm2", "mmm",
    "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
    "3", "-3", "32", "3m", "-3m",
    "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm",
}

# Layer-group ↔ space-group correspondence (Table 20 in the user's reference image).
# We use this only for sanity checks / logging to avoid mixing up:
#   - "layer-group #39" (1–80)  vs  "space-group #39" (1–230).
#
# NOTE: Some space-group numbers correspond to multiple layer-group numbers in Table 20.
#       For our purposes (consistency checks & point-group expectation), this is fine.
LAYERGROUP_NO_TO_SPACEGROUP_NO: Dict[int, int] = {
    1: 1,
    2: 2,
    3: 3,
    4: 6,
    5: 7,
    6: 10,
    7: 13,
    8: 3,
    9: 4,
    10: 5,
    11: 6,
    12: 7,
    13: 8,
    14: 10,
    15: 11,
    16: 12,
    17: 13,
    18: 14,
    19: 16,
    20: 17,
    21: 18,
    22: 21,
    23: 25,
    24: 25,
    25: 26,
    26: 26,
    27: 27,
    28: 28,
    29: 28,
    30: 29,
    31: 30,
    32: 31,
    33: 32,
    34: 35,
    35: 38,
    36: 39,
    37: 47,
    38: 49,
    39: 50,
    40: 51,
    41: 51,
    42: 53,
    43: 54,
    44: 55,
    45: 57,
    46: 59,
    47: 65,
    48: 67,
    49: 75,
    50: 81,
    51: 83,
    52: 85,
    53: 89,
    54: 90,
    55: 99,
    56: 100,
    57: 111,
    58: 113,
    59: 115,
    60: 117,
    61: 123,
    62: 125,
    63: 127,
    64: 129,
    65: 143,
    66: 147,
    67: 149,
    68: 150,
    69: 156,
    70: 157,
    71: 162,
    72: 164,
    73: 168,
    74: 174,
    75: 175,
    76: 177,
    77: 183,
    78: 187,
    79: 189,
    80: 191,
}


def pointgroup_from_spacegroup_number(sg_no: int) -> Optional[str]:
    """Return the Hermann–Mauguin point-group symbol (crystal class) for a 3D space-group number."""
    s = int(sg_no)
    if s == 1:
        return "1"
    if s == 2:
        return "-1"
    if 3 <= s <= 5:
        return "2"
    if 6 <= s <= 9:
        return "m"
    if 10 <= s <= 15:
        return "2/m"
    if 16 <= s <= 24:
        return "222"
    if 25 <= s <= 46:
        return "mm2"
    if 47 <= s <= 74:
        return "mmm"
    if 75 <= s <= 80:
        return "4"
    if 81 <= s <= 82:
        return "-4"
    if 83 <= s <= 88:
        return "4/m"
    if 89 <= s <= 98:
        return "422"
    if 99 <= s <= 110:
        return "4mm"
    if 111 <= s <= 122:
        return "-42m"
    if 123 <= s <= 142:
        return "4/mmm"
    if 143 <= s <= 146:
        return "3"
    if 147 <= s <= 148:
        return "-3"
    if 149 <= s <= 155:
        return "32"
    if 156 <= s <= 161:
        return "3m"
    if 162 <= s <= 167:
        return "-3m"
    if 168 <= s <= 173:
        return "6"
    if s == 174:
        return "-6"
    if 175 <= s <= 176:
        return "6/m"
    if 177 <= s <= 182:
        return "622"
    if 183 <= s <= 186:
        return "6mm"
    if 187 <= s <= 188:
        return "-6m2"
    if 189 <= s <= 194:
        return "6/mmm"
    if 195 <= s <= 199:
        return "23"
    if 200 <= s <= 206:
        return "m-3"
    if 207 <= s <= 214:
        return "432"
    if 215 <= s <= 220:
        return "-43m"
    if 221 <= s <= 230:
        return "m-3m"
    return None


# Derived: layer-group expected point-group (by crystal class of the corresponding 3D space group).
# This is used only for sanity checks when you provide PARENT_LAYER_GROUP_NO_HINT.
LAYERGROUP_NO_TO_POINTGROUP: Dict[int, str] = {
    int(lg): (pointgroup_from_spacegroup_number(int(sg)) or "?")
    for lg, sg in LAYERGROUP_NO_TO_SPACEGROUP_NO.items()
}

# Inverse: space-group number -> possible layer-group numbers
SPACEGROUP_NO_TO_LAYERGROUP_NOS: Dict[int, List[int]] = {}
for _lg, _sg in LAYERGROUP_NO_TO_SPACEGROUP_NO.items():
    SPACEGROUP_NO_TO_LAYERGROUP_NOS.setdefault(int(_sg), []).append(int(_lg))
for _sg, _lgs in SPACEGROUP_NO_TO_LAYERGROUP_NOS.items():
    _lgs.sort()
del _lg, _sg, _lgs


def normalize_pointgroup_symbol(pg: Optional[str]) -> Optional[str]:
    """Normalize point-group strings (Hermann–Mauguin) into a compact form.

    - Remove spaces
    - Convert unicode minus to '-'
    - Lowercase
    - Keep backward compatibility: "mm4"->"4mm", "mm6"->"6mm"
    """
    if pg is None:
        return None
    s = str(pg).strip().replace("−", "-")
    s = s.replace(" ", "")
    s = s.lower()
    # Backward compatibility: mm4/mm6
    if s.startswith("mm") and len(s) == 3 and s[2].isdigit() and s[2] in {"4", "6"}:
        s = f"{s[2]}mm"
    # Some people write 3mm for C3v
    if s == "3mm":
        s = "3m"
    return s


def layergroup_expected_pointgroup(layergroup_no: Optional[int]) -> Optional[str]:
    if layergroup_no is None:
        return None
    return normalize_pointgroup_symbol(LAYERGROUP_NO_TO_POINTGROUP.get(int(layergroup_no)))

def layergroup_expected_spacegroup_number(layergroup_no: Optional[int]) -> Optional[int]:
    if layergroup_no is None:
        return None
    return LAYERGROUP_NO_TO_SPACEGROUP_NO.get(int(layergroup_no))



@dataclass
class AxesFrame:
    """Right-handed orthonormal frame used to classify symmetry elements.

    For layer groups, we define:
    - z = unit normal of the (a,b) plane
    - x = projection of a onto the plane
    - y = z × x
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


def make_layer_axes_from_lattice(lattice: np.ndarray, eps: float = 1e-12) -> AxesFrame:
    # Defensive typing: this function expects a numeric (3,3) lattice matrix.
    # If you see this error, you likely passed an AxesFrame (x,y,z) object by mistake.
    if isinstance(lattice, AxesFrame):
        raise TypeError(
            "make_layer_axes_from_lattice expects a (3,3) lattice matrix (crys.lattice), "
            "but got an AxesFrame. Please pass the lattice, not the axes frame."
        )
    lat = np.asarray(lattice, dtype=float)
    if lat.shape != (3, 3):
        raise TypeError(
            f"make_layer_axes_from_lattice expects lattice shape (3,3), got {lat.shape}."
        )

    a = np.asarray(lat[0], dtype=float)
    b = np.asarray(lat[1], dtype=float)
    z = np.cross(a, b)
    nz = float(np.linalg.norm(z))
    if nz < eps:
        # Fallback: use c if a,b nearly collinear
        z = np.asarray(lat[2], dtype=float)
        nz = float(np.linalg.norm(z))
    z = z / nz
    # x: a projected to plane
    x = a - float(np.dot(a, z)) * z
    nx = float(np.linalg.norm(x))
    if nx < eps:
        x = b - float(np.dot(b, z)) * z
        nx = float(np.linalg.norm(x))
    x = x / nx
    y = np.cross(z, x)
    ny = float(np.linalg.norm(y))
    if ny < eps:
        # fallback: gram-schmidt with b
        y = b - float(np.dot(b, z)) * z - float(np.dot(b, x)) * x
        y = y / float(np.linalg.norm(y))
    else:
        y = y / ny
    return AxesFrame(x=x, y=y, z=z)


def rotation_order_int(R: np.ndarray, max_order: int = 12) -> int:
    """Small-integer order for an integer rotation matrix R."""
    I = np.eye(3, dtype=int)
    M = np.eye(3, dtype=int)
    for n in range(1, max_order + 1):
        M = M @ R
        if np.array_equal(M, I):
            return n
    return max_order + 1


def frac_rot_to_cart(R_frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional rotation to Cartesian rotation: R_cart = L^T R L^{-T}."""
    LT = np.asarray(lattice, dtype=float).T
    return LT @ np.asarray(R_frac, dtype=float) @ np.linalg.inv(LT)


def _unit_vector(v: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def _eigvec_near_eigval(R: np.ndarray, target: float, atol: float = 1e-6) -> Optional[np.ndarray]:
    """Return a real unit eigenvector whose eigenvalue is closest to `target`."""
    w, V = np.linalg.eig(R)
    idx = int(np.argmin(np.abs(w - target)))
    if float(np.abs(w[idx] - target)) > 1e-2 and target in (1.0, -1.0):
        # still accept; some numerical noise is expected for skew lattices
        pass
    v = np.real(V[:, idx])
    v = _unit_vector(v)
    if v is None:
        return None
    # sanity: check residual
    if float(np.linalg.norm(R @ v - target * v)) > atol:
        # try least-squares nullspace for (R-target I)
        A = R - target * np.eye(3)
        _, _, VT = np.linalg.svd(A)
        v2 = np.real(VT[-1, :])
        v2 = _unit_vector(v2)
        if v2 is None:
            return v
        return v2
    return v


@dataclass
class OpInfo:
    idx: int
    R_frac: np.ndarray
    R_cart: np.ndarray
    det: int
    order: int
    axis: Optional[np.ndarray]
    mirror_normal: Optional[np.ndarray]


def build_opinfo_list(rep_ops: List[Tuple[np.ndarray, np.ndarray]], lattice: np.ndarray) -> List[OpInfo]:
    infos: List[OpInfo] = []
    I = np.eye(3, dtype=int)
    minusI = -np.eye(3, dtype=int)
    for i, (R, _) in enumerate(rep_ops):
        R_int = np.array(R, dtype=int)
        R_cart = frac_rot_to_cart(R_int, lattice)
        det = int(round(float(np.linalg.det(R_cart))))
        order = 1 if np.array_equal(R_int, I) else rotation_order_int(R_int, max_order=12)

        axis: Optional[np.ndarray] = None
        mirror_normal: Optional[np.ndarray] = None

        is_inversion = np.array_equal(R_int, minusI)
        if is_inversion:
            axis = None
            mirror_normal = None
        else:
            if det == 1 and order > 1:
                axis = _eigvec_near_eigval(R_cart, 1.0)
            elif det == -1 and order > 2:
                # rotoinversion axis corresponds to eigenvalue -1 (since it's (-I)*Cn)
                axis = _eigvec_near_eigval(R_cart, -1.0)
            if det == -1 and order == 2:
                # mirror: unique normal with eigenvalue -1
                mirror_normal = _eigvec_near_eigval(R_cart, -1.0)

        infos.append(
            OpInfo(
                idx=i,
                R_frac=R_int,
                R_cart=R_cart,
                det=det,
                order=order,
                axis=axis,
                mirror_normal=mirror_normal,
            )
        )
    return infos


def _best_aligned_index(
    infos: List[OpInfo],
    *,
    predicate,
    vec_getter,
    target: np.ndarray,
) -> Optional[int]:
    best_i: Optional[int] = None
    best_s = -1.0
    t = np.asarray(target, dtype=float)
    for op in infos:
        if not predicate(op):
            continue
        v = vec_getter(op)
        if v is None:
            continue
        s = abs(float(np.dot(v, t)))
        if s > best_s:
            best_s = s
            best_i = op.idx
    return best_i


def infer_generator_indices(pg_symbol: str, rep_ops: List[Tuple[np.ndarray, np.ndarray]], lattice: np.ndarray) -> Dict[str, int]:
    """Pick a small set of concrete generators (indices into rep_ops) to label irreps."""
    pg = normalize_pointgroup_symbol(pg_symbol) or ""
    if pg not in AXIAL_POINT_GROUPS_27:
        raise RuntimeError(f"Unsupported / non-axial point group for layer groups: {pg_symbol}")

    axes = make_layer_axes_from_lattice(lattice)
    infos = build_opinfo_list(rep_ops, lattice)
    gens: Dict[str, int] = {}

    # Identity
    for op in infos:
        if np.array_equal(op.R_frac, np.eye(3, dtype=int)):
            gens["E"] = op.idx
            break
    if "E" not in gens:
        gens["E"] = 0

    # Inversion
    if pg in {"-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm"}:
        for op in infos:
            if np.array_equal(op.R_frac, -np.eye(3, dtype=int)):
                gens["i"] = op.idx
                break
        if "i" not in gens:
            raise RuntimeError(f"Point group {pg_symbol} requires inversion, but no -I found in ops")

    # Principal-axis rotations / rotoinversions
    if pg in {"2", "2/m", "mm2", "222", "mmm"}:
        idx = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == 1 and o.order == 2 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.z,
        )
        if idx is not None:
            gens["C2z"] = idx

    if pg in {"3", "-3", "32", "3m", "-3m", "-6", "-6m2"}:
        idx = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == 1 and o.order == 3 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.z,
        )
        if idx is not None:
            gens["C3z"] = idx

    if pg in {"4", "4/m", "422", "4mm", "4/mmm"}:
        idx = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == 1 and o.order == 4 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.z,
        )
        if idx is not None:
            gens["C4z"] = idx

    if pg in {"-4", "-42m"}:
        idx = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == -1 and o.order == 4 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.z,
        )
        if idx is not None:
            gens["S4z"] = idx

    if pg in {"6", "6/m", "622", "6mm", "6/mmm"}:
        idx = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == 1 and o.order == 6 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.z,
        )
        if idx is not None:
            gens["C6z"] = idx

    # In-plane C2 axes for D2 / D3 / D4 / D6 families
    if pg in {"222", "mmm"}:
        ix = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == 1 and o.order == 2 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.x,
        )
        iy = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == 1 and o.order == 2 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.y,
        )
        iz = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == 1 and o.order == 2 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.z,
        )
        if ix is not None:
            gens["C2x"] = ix
        if iy is not None:
            gens["C2y"] = iy
        if iz is not None:
            gens["C2z"] = iz

    if pg in {"32", "-3m", "-6m2"}:
        ix = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == 1 and o.order == 2 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.x,
        )
        if ix is not None:
            gens["C2x"] = ix

    if pg in {"422", "4/mmm", "-42m", "622", "6/mmm"}:
        ix = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == 1 and o.order == 2 and o.axis is not None),
            vec_getter=lambda o: o.axis,
            target=axes.x,
        )
        if ix is not None:
            gens["C2x"] = ix

    # Mirrors
    if pg == "m":
        # any mirror
        for op in infos:
            if op.det == -1 and op.order == 2 and op.mirror_normal is not None:
                gens["sigma"] = op.idx
                break

    if pg in {"-6", "-6m2"}:
        # horizontal mirror sigma_h: normal || z
        idx = _best_aligned_index(
            infos,
            predicate=lambda o: (o.det == -1 and o.order == 2 and o.mirror_normal is not None),
            vec_getter=lambda o: o.mirror_normal,
            target=axes.z,
        )
        if idx is not None:
            gens["sigma_h"] = idx

    if pg in {"mm2", "3m", "4mm", "6mm"}:
        # vertical mirror sigma_v: plane contains z and x -> normal || y
        candidates: List[OpInfo] = [
            o
            for o in infos
            if (o.det == -1 and o.order == 2 and o.mirror_normal is not None)
        ]
        best_idx = None
        best_s = -1.0
        for o in candidates:
            n = o.mirror_normal
            if n is None:
                continue
            # vertical: normal ⟂ z
            if abs(float(np.dot(n, axes.z))) > 0.2:
                continue
            s = abs(float(np.dot(n, axes.y)))
            if s > best_s:
                best_s = s
                best_idx = o.idx
        if best_idx is not None:
            gens["sigma_v"] = best_idx

    return gens


def _sig_int(sig: Tuple[float, ...], idx: int) -> int:
    return int(round(float(sig[idx])))


def mulliken_label_from_signature(pg_symbol: str, sig: Tuple[float, ...], gens: Dict[str, int]) -> str:
    """Map a character signature to a Mulliken irrep label for the given point group."""
    pg = normalize_pointgroup_symbol(pg_symbol) or ""
    if pg not in AXIAL_POINT_GROUPS_27:
        raise RuntimeError(f"Unsupported / non-axial point group for layer groups: {pg_symbol}")

    dim = _sig_int(sig, gens.get("E", 0))

    def gu_suffix() -> str:
        if "i" not in gens:
            return ""
        chi_i = _sig_int(sig, gens["i"])
        if chi_i == dim:
            return "g"
        if chi_i == -dim:
            return "u"
        return "?"

    def prime_suffix() -> str:
        if "sigma_h" not in gens:
            return ""
        chi = _sig_int(sig, gens["sigma_h"])
        if chi == dim:
            return "'"
        if chi == -dim:
            return "''"
        return "?"

    if pg == "1":
        return "A"

    if pg == "-1":
        return f"A{gu_suffix()}"

    if pg == "2":
        chi = _sig_int(sig, gens["C2z"])
        return "A" if chi == 1 else "B"

    if pg == "m":
        chi = _sig_int(sig, gens["sigma"])
        return "A'" if chi == 1 else "A''"

    if pg == "2/m":
        # C2 x Ci
        chi_c2 = _sig_int(sig, gens["C2z"])
        base = "A" if chi_c2 == 1 else "B"
        return f"{base}{gu_suffix()}"

    if pg == "222":
        cx = _sig_int(sig, gens["C2x"])
        cy = _sig_int(sig, gens["C2y"])
        cz = _sig_int(sig, gens["C2z"])
        if (cx, cy, cz) == (1, 1, 1):
            return "A"
        if (cx, cy, cz) == (-1, -1, 1):
            return "B1"
        if (cx, cy, cz) == (-1, 1, -1):
            return "B2"
        if (cx, cy, cz) == (1, -1, -1):
            return "B3"
        return "IR?"

    if pg == "mm2":
        chi_c2 = _sig_int(sig, gens["C2z"])
        chi_sv = _sig_int(sig, gens["sigma_v"])
        if chi_c2 == 1:
            return "A1" if chi_sv == 1 else "A2"
        return "B1" if chi_sv == 1 else "B2"

    if pg == "mmm":
        base = mulliken_label_from_signature("222", sig, gens)
        return f"{base}{gu_suffix()}"

    if pg == "3":
        return "A" if dim == 1 else "E"

    if pg == "-3":
        base = "A" if dim == 1 else "E"
        return f"{base}{gu_suffix()}"

    if pg == "32":
        if dim == 2:
            return "E"
        chi = _sig_int(sig, gens["C2x"])
        return "A1" if chi == 1 else "A2"

    if pg == "3m":
        if dim == 2:
            return "E"
        chi = _sig_int(sig, gens["sigma_v"])
        return "A1" if chi == 1 else "A2"

    if pg == "-3m":
        base = mulliken_label_from_signature("32", sig, gens)
        return f"{base}{gu_suffix()}"

    if pg == "4":
        if dim == 2:
            return "E"
        chi = _sig_int(sig, gens["C4z"])
        return "A" if chi == 1 else "B"

    if pg == "-4":
        if dim == 2:
            return "E"
        chi = _sig_int(sig, gens["S4z"])
        return "A" if chi == 1 else "B"

    if pg == "4/m":
        base = mulliken_label_from_signature("4", sig, gens)
        return f"{base}{gu_suffix()}"

    if pg == "422":
        if dim == 2:
            return "E"
        chi_c4 = _sig_int(sig, gens["C4z"])
        chi_c2x = _sig_int(sig, gens["C2x"])
        if chi_c4 == 1:
            return "A1" if chi_c2x == 1 else "A2"
        return "B1" if chi_c2x == 1 else "B2"

    if pg == "4mm":
        if dim == 2:
            return "E"
        chi_c4 = _sig_int(sig, gens["C4z"])
        chi_sv = _sig_int(sig, gens["sigma_v"])
        if chi_c4 == 1:
            return "A1" if chi_sv == 1 else "A2"
        return "B1" if chi_sv == 1 else "B2"

    if pg == "-42m":
        if dim == 2:
            return "E"
        chi_s4 = _sig_int(sig, gens["S4z"])
        chi_c2x = _sig_int(sig, gens["C2x"])
        if chi_s4 == 1:
            return "A1" if chi_c2x == 1 else "A2"
        return "B1" if chi_c2x == 1 else "B2"

    if pg == "4/mmm":
        base = mulliken_label_from_signature("422", sig, gens)
        return f"{base}{gu_suffix()}"

    if pg == "6":
        if dim == 1:
            chi = _sig_int(sig, gens["C6z"])
            return "A" if chi == 1 else "B"
        chi = _sig_int(sig, gens["C6z"])
        return "E1" if chi == 1 else "E2"

    if pg == "-6":
        # C3h = C3 x Cs (sigma_h)
        base = "A" if dim == 1 else "E"
        return f"{base}{prime_suffix()}"

    if pg == "6/m":
        base = mulliken_label_from_signature("6", sig, gens)
        return f"{base}{gu_suffix()}"

    if pg == "622":
        if dim == 2:
            chi = _sig_int(sig, gens["C6z"])
            return "E1" if chi == 1 else "E2"
        chi_c6 = _sig_int(sig, gens["C6z"])
        chi_c2x = _sig_int(sig, gens["C2x"])
        if chi_c6 == 1:
            return "A1" if chi_c2x == 1 else "A2"
        return "B1" if chi_c2x == 1 else "B2"

    if pg == "6mm":
        if dim == 2:
            chi = _sig_int(sig, gens["C6z"])
            return "E1" if chi == 1 else "E2"
        chi_c6 = _sig_int(sig, gens["C6z"])
        chi_sv = _sig_int(sig, gens["sigma_v"])
        if chi_c6 == 1:
            return "A1" if chi_sv == 1 else "A2"
        return "B1" if chi_sv == 1 else "B2"

    if pg == "-6m2":
        # D3h = D3 x Cs (sigma_h)
        if dim == 2:
            base = "E"
        else:
            chi = _sig_int(sig, gens["C2x"])
            base = "A1" if chi == 1 else "A2"
        return f"{base}{prime_suffix()}"

    if pg == "6/mmm":
        base = mulliken_label_from_signature("622", sig, gens)
        return f"{base}{gu_suffix()}"

    raise RuntimeError(f"Unhandled point group: {pg_symbol}")


def build_landau_basis_auto_pointgroup(
    crys: Crystal,
    rotations: np.ndarray,
    translations: np.ndarray,
    symprec_map: float,
    point_group: Optional[str] = None,
    enable_parity: bool = True,
    seed: int = 20250101,
) -> Dict[str, List[np.ndarray]]:
    """Build Landau mode basis for *any* point group (including all layer-group point groups).

    统一输出 **Mulliken/晶体学** irrep 标签（如 A1/A2/B1/B2/E/Eg/Eu/E1/E2/...），
    并保留 @G/@R（中心化平移 τ 的偶/奇）分扇区。

    同时，在运行 log 里打印：IRxx_d?@{G,R} -> Mulliken@{G,R}，便于你核对/追溯。
    """
    N = crys.nsites
    dim = 3 * N
    map_tol = max(symprec_map * 5.0, 1e-5)

    pg = normalize_pointgroup_symbol(point_group)
    if pg is None:
        raise RuntimeError(
            "Point group is None; cannot map IRxx -> Mulliken labels. "
            "Please set PARENT_POINT_GROUP_HINT or ensure spglib dataset provides pointgroup."
        )
    if pg not in AXIAL_POINT_GROUPS_27:
        raise RuntimeError(
            f"Unsupported / non-axial point group for layer-group workflow: '{pg}'. "
            f"Supported axial PGs: {sorted(AXIAL_POINT_GROUPS_27)}"
        )

    # One (R|t) representative per rotation matrix (i.e., per point-group element)
    rep_ops = select_representative_ops_by_rotation(rotations, translations)
    D_full_ops: List[np.ndarray] = [build_representation_matrix(crys, R, t, map_tol=map_tol) for R, t in rep_ops]

    # Prepare Mulliken labelling helpers from concrete operations
    # NOTE: infer_generator_indices expects the representative (R|t) list plus lattice,
    # and will internally build OpInfo and the layer-axes frame.
    gens = infer_generator_indices(pg, rep_ops, crys.lattice)

    # Detect centering translation and build parity projectors
    parity_projectors: Dict[str, np.ndarray] = {"G": np.eye(dim)}
    if enable_parity:
        tau = detect_centering_translation(rotations, translations)
        if tau is not None:
            D_tau = build_representation_matrix(crys, np.eye(3, dtype=int), tau, map_tol=map_tol)
            parity_projectors["G"] = 0.5 * (np.eye(dim) + D_tau)
            parity_projectors["R"] = 0.5 * (np.eye(dim) - D_tau)

    # First pass: compute per-parity {signature -> mode vectors}
    sig_parity_modes: Dict[Tuple[Union[float, int], ...], Dict[str, List[np.ndarray]]] = {}

    for ptag, Pp in parity_projectors.items():
        # If Pp is identity, skip reduction (fast path)
        if np.allclose(Pp, np.eye(dim)):
            Q = np.eye(dim)
            D_ops = D_full_ops
            reduced = False
        else:
            basis_p = basis_from_projector(Pp, eig_threshold=0.5)
            if not basis_p:
                continue
            Q = np.column_stack(basis_p)  # (dim, d_p)
            D_ops = [Q.T @ D @ Q for D in D_full_ops]
            reduced = True

        # Decompose in this parity sector
        by_sig = _decompose_by_commutant(
            D_ops=D_ops,
            seed=int(seed) + (0 if ptag == "G" else 99991),
        )

        for sig, blocks in by_sig.items():
            out_list: List[np.ndarray] = []
            for Vblk in blocks:
                if reduced:
                    full = Q @ Vblk  # (dim, k)
                else:
                    full = Vblk  # already (dim, k)
                # Each column is a basis vector in this irrep copy
                for j in range(full.shape[1]):
                    v = normalize_mode(full[:, j], nsites=N)
                    if np.linalg.norm(v) > 1e-12:
                        out_list.append(v)

            if out_list:
                sig_parity_modes.setdefault(sig, {})[ptag] = out_list

    if not sig_parity_modes:
        return {}

    # Assign stable IR labels from signature ordering (for alias/traceability)
    sigs_sorted = sorted(sig_parity_modes.keys())
    sig_to_label = {sig: i for i, sig in enumerate(sigs_sorted)}

    landau_basis: Dict[str, List[np.ndarray]] = {}
    for sig, by_p in sig_parity_modes.items():
        idx = sig_to_label[sig]
        # irrep dimension = character of identity element, which is signature[0] because identity is sorted first.
        d_ir = int(sig[0]) if isinstance(sig[0], (int, np.integer)) else int(round(float(sig[0])))
        mull = mulliken_label_from_signature(pg, sig, gens)
        for ptag, vecs in by_p.items():
            ir_key = f"IR{idx:02d}_d{d_ir}@{ptag}"
            mull_key = f"{mull}@{ptag}"

            # Save canonical Mulliken key
            if mull_key in landau_basis:
                # Should not happen; if it does, keep both deterministically.
                alt_key = f"{mull}_IR{idx:02d}_d{d_ir}@{ptag}"
                print(f"[WARN] Mulliken key collision: '{mull_key}' already exists; using '{alt_key}' instead")
                mull_key = alt_key
            landau_basis[mull_key] = vecs

            # Record alias mapping for backward compatibility
            IRREP_LABEL_ALIASES[ir_key] = mull_key
            IRREP_LABEL_ALIASES[mull_key] = mull_key

            print(f"[IRREP] {ir_key} -> {mull_key}")

    return landau_basis


# =========================
# Distortion / strain application
# =========================

def apply_strain(lattice: np.ndarray, sa: float, sb: float, sc: float) -> np.ndarray:
    """Scale lattice vectors a,b,c (row vectors)."""
    L = np.array(lattice, float).copy()
    L[0, :] *= float(sa)
    L[1, :] *= float(sb)
    L[2, :] *= float(sc)
    return L


def make_distorted_crystal(
    parent: Crystal,
    landau_basis: Dict[str, List[np.ndarray]],
    mode_amplitudes: Dict[Tuple[str, int], float],
    sa: float,
    sb: float,
    sc: float,
) -> Crystal:
    """Return a new Crystal with strain and atomic displacements applied.

    mode_amplitudes keys: (irrep_tag, mode_idx) e.g. ("B2@R", 0)
    amplitudes in Å units (since mode vectors are normalized to max displacement 1).
    """
    N = parent.nsites
    L_new = apply_strain(parent.lattice, sa, sb, sc)

    base_cart = parent.frac @ L_new
    disp = np.zeros((N, 3), dtype=float)

    for (irrep_tag, midx), amp in mode_amplitudes.items():
        # Backward compatibility: allow legacy IRxx keys (or older saved combo specs)
        # to resolve to the canonical Mulliken-labelled keys.
        tag_use = irrep_tag
        vecs = landau_basis.get(tag_use, None)
        if vecs is None and IRREP_LABEL_ALIASES:
            ali = IRREP_LABEL_ALIASES.get(tag_use, None)
            if ali is not None:
                tag_use = ali
                vecs = landau_basis.get(tag_use, None)
        if vecs is None:
            raise KeyError(
                f"Mode irrep '{irrep_tag}' not found in landau_basis (after alias). "
                f"Available: {list(landau_basis.keys())}"
            )
        if not (0 <= midx < len(vecs)):
            raise IndexError(f"Mode index out of range for {tag_use}: {midx}, n={len(vecs)}")
        v = vecs[midx].reshape(N, 3)
        disp += float(amp) * v

    cart_new = base_cart + disp
    frac_new = cart_new @ np.linalg.inv(L_new)
    frac_new = wrap_frac(frac_new)

    return Crystal(lattice=L_new, frac=frac_new, numbers=parent.numbers.copy(), symbols=parent.symbols.copy())


# =========================
# Combo generation / Plan-C
# =========================

def normalize_direction(d: List[float], eps: float = 1e-12) -> List[float]:
    dn = float(np.linalg.norm(np.array(d, float)))
    if dn < eps:
        return d
    return [float(x) / dn for x in d]



# ---------------------- 随机方向 + 近似正交化采样 ----------------------

def _canonicalize_unit_direction(vec, eps: float = 1e-12):
    """Normalize vec to unit length, and canonicalize overall sign.

    Canonical rule: find the first component whose |value|>eps;
    if it is negative, multiply the whole vector by -1.

    Returns:
        np.ndarray of shape (dim,) if valid; otherwise None.
    """
    v = np.asarray(vec, dtype=float)
    if v.ndim != 1:
        v = v.reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    v = v / n
    # Canonicalize overall sign so that v and -v are treated as the same "direction"
    for x in v:
        if abs(float(x)) > eps:
            if float(x) < 0.0:
                v = -v
            break
    v[np.abs(v) < eps] = 0.0
    return v


def _dir_key(v: np.ndarray, ndigits: int = 8):
    return tuple(float(x) for x in np.round(v, ndigits))


def generate_spread_unit_directions(dim: int,
                                    n_random: int,
                                    rng: np.random.Generator,
                                    fixed_dirs: Optional[List[List[float]]] = None,
                                    n_candidates: int = 2000,
                                    eps: float = 1e-12,
                                    ndigits: int = 8) -> List[List[float]]:
    """Generate unit directions on S^{dim-1}.

    - Start from fixed_dirs (if any), normalize & de-duplicate (up to overall sign).
    - Add n_random directions picked from random candidates with a greedy
      "most orthogonal to current set" criterion (minimize max |dot|).

    This is NOT strict pairwise orthogonality (impossible when n >> dim),
    but it makes the set much less clustered than plain random draws.
    """
    selected: List[np.ndarray] = []
    seen = set()

    if fixed_dirs:
        for d in fixed_dirs:
            v = _canonicalize_unit_direction(d, eps=eps)
            if v is None:
                continue
            k = _dir_key(v, ndigits=ndigits)
            if k in seen:
                continue
            selected.append(v)
            seen.add(k)

    target_total = len(selected) + max(0, int(n_random))
    if target_total == len(selected):
        return [v.tolist() for v in selected]

    # Build candidate pool
    cand: List[np.ndarray] = []
    cand_seen = set()
    # Oversample slightly to avoid duplicates
    max_draws = max(n_candidates * 5, target_total * 200)
    for _ in range(max_draws):
        v = _canonicalize_unit_direction(rng.normal(size=dim), eps=eps)
        if v is None:
            continue
        k = _dir_key(v, ndigits=ndigits)
        if (k in seen) or (k in cand_seen):
            continue
        cand.append(v)
        cand_seen.add(k)
        if len(cand) >= n_candidates:
            break

    # Greedy selection: each time pick candidate that minimizes max |dot| with selected
    while len(selected) < target_total and cand:
        if not selected:
            v = cand.pop(0)
            selected.append(v)
            seen.add(_dir_key(v, ndigits=ndigits))
            continue

        best_idx = None
        best_score = 1e9
        # A small optimization: evaluate a subset if cand is huge
        # (keep deterministic order)
        for i, v in enumerate(cand):
            # score = max absolute overlap with current selected set
            score = max(abs(float(np.dot(v, s))) for s in selected)
            if score < best_score:
                best_score = score
                best_idx = i
                # early stop if already very orthogonal
                if best_score < 1e-3:
                    break
        v = cand.pop(best_idx)
        selected.append(v)
        seen.add(_dir_key(v, ndigits=ndigits))

    return [v.tolist() for v in selected]


def build_op_directions_for_combo(m: int,
                                  base_dirs: List[List[float]],
                                  rng: np.random.Generator) -> List[List[float]]:
    """Merge base op_directions with random spread directions (if enabled)."""
    if m < 2:
        return base_dirs
    if not ENABLE_RANDOM_OP_DIRECTIONS or N_RANDOM_OP_DIRECTIONS_PER_COMBO <= 0:
        return base_dirs
    return generate_spread_unit_directions(
        dim=m,
        n_random=N_RANDOM_OP_DIRECTIONS_PER_COMBO,
        rng=rng,
        fixed_dirs=base_dirs,
        n_candidates=RANDOM_OP_DIRECTION_CANDIDATES,
    )

# ----------------------------------------------------------------------
def plan_c_autofill_combos_bruteforce(
    parent: Crystal,
    landau_basis: Dict[str, List[np.ndarray]],
    target_sgs: List[int],
    symprec_identify: float,
    angle_tolerance: float,
) -> List[Dict[str, Any]]:
    """Robust Plan-C that does NOT depend on spgrep_modulation.

    It brute-tests 1-mode and 2-mode combinations with a small set of OP directions,
    and keeps those that *actually* get identified as one of target_sgs.
    """
    target_set = set(int(x) for x in target_sgs)
    all_modes: List[Tuple[str, int]] = []
    for irrep_tag, vecs in landau_basis.items():
        for i in range(len(vecs)):
            all_modes.append((irrep_tag, i))

    # Prefer @R modes if they exist (often needed to break centering)
    r_modes = [m for m in all_modes if "@R" in m[0]]
    g_modes = [m for m in all_modes if "@G" in m[0]]
    modes_priority = r_modes + g_modes if r_modes else all_modes

    found: Dict[Tuple[Tuple[Tuple[str, int], ...], Tuple[float, ...]], Set[int]] = {}

    def test_combo(modes: List[Tuple[str, int]], direction: List[float]):
        d = normalize_direction(direction)
        mode_amplitudes = {(modes[k][0], modes[k][1]): PLAN_C_TEST_RHO * d[k] for k in range(len(modes))}
        trial = make_distorted_crystal(parent, landau_basis, mode_amplitudes, 1.0, 1.0, 1.0)
        sg, _, _ = identify_spacegroup(trial, symprec=symprec_identify, angle_tolerance=angle_tolerance)
        if sg in target_set:
            key_modes = tuple(modes)
            key_dir = tuple(float(x) for x in direction)
            found.setdefault((key_modes, key_dir), set()).add(int(sg))

    # 1-mode
    if PLAN_C_MAX_COMBO_SIZE >= 1:
        for m in modes_priority:
            test_combo([m], [1.0])

    # 2-mode
    if PLAN_C_MAX_COMBO_SIZE >= 2:
        for m1, m2 in combinations(modes_priority, 2):
            for d in PLAN_C_DIR_POOL_2D:
                test_combo([m1, m2], d)

    combos: List[Dict[str, Any]] = []
    for (modes_key, dir_key), sgs in found.items():
        combos.append({
            "target_sgs": set(int(x) for x in sgs),
            "irrep_modes": [(m[0], int(m[1])) for m in modes_key],
            "op_directions": [list(dir_key)],
        })

    # Sort for deterministic output
    combos.sort(key=lambda c: (min(c["target_sgs"]), len(c["irrep_modes"]), str(c["irrep_modes"])))
    return combos


def auto_generate_combos(landau_basis: Dict[str, List[np.ndarray]], max_size: int = 2) -> List[Dict[str, Any]]:
    """Generate generic multi-mode combos (no target SG constraint)."""
    all_modes: List[Tuple[str, int]] = []
    for irrep_tag, vecs in landau_basis.items():
        for i in range(len(vecs)):
            all_modes.append((irrep_tag, i))

    combos: List[Dict[str, Any]] = []
    if max_size <= 1:
        return combos

    # Only size-2 combos for now (keeps scan size manageable)
    for m1, m2 in combinations(all_modes, 2):
        combos.append({
            "target_sgs": set(),  # unknown
            "irrep_modes": [m1, m2],
            "op_directions": [d[:] for d in AUTO_DIR_POOL_2D],
        })

    return combos


def build_combo_specs(
    parent: Crystal,
    landau_basis: Dict[str, List[np.ndarray]],
    symprec_identify: float,
    angle_tolerance: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Return (combo_specs, using_confirmed).

    Critical fix for your crash:
    - If USE_ISOTROPY_CONFIRMED_COMBOS=True but list is empty:
        try Plan-C -> if still empty -> fallback to auto combos (NO RuntimeError).
    """
    global ISOTROPY_CONFIRMED_COMBOS

    if USE_ISOTROPY_CONFIRMED_COMBOS:
        if ISOTROPY_CONFIRMED_COMBOS:
            return ISOTROPY_CONFIRMED_COMBOS, True

        if ENABLE_PLAN_C_AUTOFILL_COMBOS:
            print(f"[INFO] combos 为空，开始 Plan-C 自动枚举... target SGs = {PLAN_C_TARGET_SGS}")
            try:
                combos = plan_c_autofill_combos_bruteforce(
                    parent=parent,
                    landau_basis=landau_basis,
                    target_sgs=PLAN_C_TARGET_SGS,
                    symprec_identify=symprec_identify,
                    angle_tolerance=angle_tolerance,
                )
                ISOTROPY_CONFIRMED_COMBOS = combos
            except Exception as e:
                print(f"[Plan-C][WARN] 自动枚举失败，将回退到 AUTO combos。原因: {e}")
                ISOTROPY_CONFIRMED_COMBOS = []

            print("==================================================================")
            print(f"[Plan-C] total combos found = {len(ISOTROPY_CONFIRMED_COMBOS)}")
            print("==================================================================")
            print("[INFO] Plan-C 生成的 ISOTROPY_CONFIRMED_COMBOS 如下（可直接复制粘贴到脚本配置区）：")
            print("ISOTROPY_CONFIRMED_COMBOS = [")
            for c in ISOTROPY_CONFIRMED_COMBOS:
                print(f"  {c},")
            print("]")
            print()

            if ISOTROPY_CONFIRMED_COMBOS:
                return ISOTROPY_CONFIRMED_COMBOS, True

            print(
                "[WARN] Plan-C 未找到可用 combos，将回退到 AUTO_COMBO 生成策略继续扫描（不再中止）。"
            )
            combos_auto = auto_generate_combos(landau_basis, max_size=AUTO_COMBO_MAX_SIZE)
            return combos_auto, False

        # Plan-C disabled (no targets) -> direct fallback
        print("[INFO] combos 为空且未指定 TARGET_SGS，直接使用 AUTO_COMBO 生成策略扫描。")
        combos_auto = auto_generate_combos(landau_basis, max_size=AUTO_COMBO_MAX_SIZE)
        return combos_auto, False

    # Not using confirmed combos
    combos_auto = auto_generate_combos(landau_basis, max_size=AUTO_COMBO_MAX_SIZE)
    return combos_auto, False


# =========================
# Main scanning
# =========================

def _safe_filename(s: str, max_len: int = 180) -> str:
    """Make a filesystem-safe filename (cross-platform)."""
    s = str(s).strip()
    # Replace path separators and other unsafe chars on common OSes
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = s.replace(" ", "_")
    s = re.sub(r"__+", "_", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s.strip("._")


def maybe_write_trial_structure(
    trial: Crystal,
    out_dir: Path,
    trial_id: int,
    kind: str,
    label: str,
    sg: int,
    sg_sym: str,
    pg: Optional[str],
    hit: bool,
    write_counts: Dict[int, int],
) -> Optional[str]:
    """Write a trial structure according to the output policy.

    Returns:
        The written POSCAR path (as str) if written; otherwise None.
    """
    want_write = bool(WRITE_ALL_TRIAL_STRUCTURES) or (bool(WRITE_HIT_STRUCTURES) and bool(hit))
    if not want_write:
        return None

    if WRITE_LIMIT_PER_SG is not None:
        if write_counts.get(int(sg), 0) >= int(WRITE_LIMIT_PER_SG):
            return None

    write_counts[int(sg)] = write_counts.get(int(sg), 0) + 1

    subdir = "structures_by_sg" if WRITE_ALL_TRIAL_STRUCTURES else "hit_structures_by_sg"
    folder = Path(out_dir) / subdir / f"sg{int(sg):03d}"
    fname = _safe_filename(f"trial{int(trial_id):06d}_{kind}_{label}_sg{int(sg)}")
    path = folder / f"{fname}.vasp"

    comment = (
        f"{PROGRAM_TAG} v{PROGRAM_VERSION} | "
        f"trial={trial_id} | kind={kind} | SG={sg} ({sg_sym}) | PG={pg} | {label}"
    )
    write_poscar(trial, path, comment=comment)
    return str(path)


def scan_landau_space_guided(parent: Crystal, landau_basis: Dict[str, List[np.ndarray]], out_dir: Path) -> List[Dict[str, Any]]:
    """Run the scan. Returns list of trial records."""
    results: List[Dict[str, Any]] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_counts: Dict[int, int] = {}
    rng = np.random.default_rng(RANDOM_OP_DIRECTION_SEED)

    print("[INFO] 开始在 {α_i, ε_a, ε_b, ε_c} 空间中扫描子相（含 Plan-C 自动确认 combos）...")
    print(f"[INFO] 单模振幅网格: {AMP_GRID}")
    print(f"[INFO] 组合径向 rho 网格: {RHO_GRID}")
    print(f"[INFO] Strain a 网格: {STRAIN_A_GRID}")
    print(f"[INFO] Strain b 网格: {STRAIN_B_GRID}")
    print(f"[INFO] Strain c 网格: {STRAIN_C_GRID}")
    print()

    # ============ single-irrep probe ============
    print("================= 单 irrep 分支探测 (probe) =================")
    probe_amps = [0.4, 0.8]
    for irrep_tag, vecs in landau_basis.items():
        for midx in range(len(vecs)):
            hits: Dict[int, int] = {}
            for a in probe_amps:
                trial = make_distorted_crystal(parent, landau_basis, {(irrep_tag, midx): a}, 1.0, 1.0, 1.0)
                sg, sg_sym, pg_trial = identify_spacegroup(trial, symprec=SYMPREC_IDENTIFY, angle_tolerance=ANGLE_TOLERANCE)
                hits[int(sg)] = hits.get(int(sg), 0) + 1
            dominant_sg = max(hits.items(), key=lambda kv: kv[1])[0]
            print(f"  {irrep_tag}[{midx}] -> dominant sg={dominant_sg}, hits={hits}")
    print("============================================================")
    print()

    # ============ scan singles ============
    trial_id = 0
    for irrep_tag, vecs in landau_basis.items():
        for midx in range(len(vecs)):
            for amp, sa, sb, sc in product(AMP_GRID, STRAIN_A_GRID, STRAIN_B_GRID, STRAIN_C_GRID):
                trial_id += 1
                modes = {(irrep_tag, midx): float(amp)}
                label = f"single:{irrep_tag}[{midx}]_amp={amp:+.3f}"
                print(f"[SCAN] trial #{trial_id}: {label}, sa={sa:.2f}, sb={sb:.2f}, sc={sc:.2f}, modes={modes}")
                trial = make_distorted_crystal(parent, landau_basis, modes, sa, sb, sc)
                sg, sg_sym, pg_trial = identify_spacegroup(trial, symprec=SYMPREC_IDENTIFY, angle_tolerance=ANGLE_TOLERANCE)
                print(f"       spglib 识别的 space-group = {sg} ({sg_sym})")

                rec = {
                    "trial": trial_id,
                    "type": "single",
                    "irrep_modes": [(irrep_tag, midx)],
                    "amps": [float(amp)],
                    "sa": float(sa),
                    "sb": float(sb),
                    "sc": float(sc),
                    "sg": int(sg),
                    "sg_sym": str(sg_sym),
                    "pg": normalize_pointgroup_symbol(pg_trial),
                    "label": label,
                }

                hit = int(sg) in set(int(x) for x in PLAN_C_TARGET_SGS)
                poscar_path = maybe_write_trial_structure(
                    trial=trial,
                    out_dir=out_dir,
                    trial_id=trial_id,
                    kind="single",
                    label=label,
                    sg=int(sg),
                    sg_sym=str(sg_sym),
                    pg=normalize_pointgroup_symbol(pg_trial),
                    hit=hit,
                    write_counts=write_counts,
                )
                if poscar_path is not None:
                    rec["poscar"] = poscar_path

                results.append(rec)

    # ============ build combos ============
    combo_specs, using_confirmed = build_combo_specs(
        parent=parent,
        landau_basis=landau_basis,
        symprec_identify=SYMPREC_IDENTIFY,
        angle_tolerance=ANGLE_TOLERANCE,
    )

    if not combo_specs:
        print("[INFO] 没有多模 combos（combo_specs 为空），结束扫描。")
        return results

    print()
    print("================= 多模组合扫描 (combos) =================")
    print(f"[INFO] combo_specs 数量 = {len(combo_specs)} (using_confirmed={using_confirmed})")
    print("=========================================================")
    print()

    # ============ scan combos ============
    for cidx, combo in enumerate(combo_specs, start=1):
        irrep_modes: List[Tuple[str, int]] = [(m[0], int(m[1])) for m in combo["irrep_modes"]]
        op_dirs: List[List[float]] = [list(d) for d in combo.get("op_directions", [])]
        target_sgs: Set[int] = set(int(x) for x in combo.get("target_sgs", set()) if x is not None)

        m = len(irrep_modes)
        if m < 2:
            continue
        if not op_dirs:
            # Default: equal-weight direction
            op_dirs = [([1.0] * m)]

        dirs_to_scan = build_op_directions_for_combo(m, op_dirs, rng)
        if ENABLE_RANDOM_OP_DIRECTIONS and len(dirs_to_scan) > len(op_dirs):
            print(f"[INFO] combo #{cidx}: base_dirs={len(op_dirs)}, random_added={len(dirs_to_scan)-len(op_dirs)}, total_dirs={len(dirs_to_scan)}")
        for dir_idx, d_raw in enumerate(dirs_to_scan, start=1):
            if len(d_raw) != m:
                print(f"[WARN] 跳过一个方向向量（长度不匹配）: modes={irrep_modes}, dir={d_raw}")
                continue
            d = normalize_direction(d_raw)

            for rho, sa, sb, sc in product(RHO_GRID, STRAIN_A_GRID, STRAIN_B_GRID, STRAIN_C_GRID):
                trial_id += 1
                amps = [float(rho) * float(di) for di in d]
                modes = {(irrep_modes[k][0], irrep_modes[k][1]): amps[k] for k in range(m)}
                label = f"combo{cidx:03d}_dir{dir_idx:02d}_rho={rho:.3f}"
                print(f"[SCAN] trial #{trial_id}: {label}, sa={sa:.2f}, sb={sb:.2f}, sc={sc:.2f}, modes={modes}")
                trial = make_distorted_crystal(parent, landau_basis, modes, sa, sb, sc)
                sg, sg_sym, pg_trial = identify_spacegroup(trial, symprec=SYMPREC_IDENTIFY, angle_tolerance=ANGLE_TOLERANCE)
                print(f"       spglib 识别的 space-group = {sg} ({sg_sym})")

                rec = {
                    "trial": trial_id,
                    "type": "combo",
                    "combo_index": cidx,
                    "dir_index": dir_idx,
                    "irrep_modes": irrep_modes,
                    "dir": d_raw,
                    "rho": float(rho),
                    "amps": amps,
                    "sa": float(sa),
                    "sb": float(sb),
                    "sc": float(sc),
                    "sg": int(sg),
                    "sg_sym": str(sg_sym),
                    "pg": normalize_pointgroup_symbol(pg_trial),
                    "label": label,
                    "combo_target_sgs": sorted(target_sgs),
                }

                hit = False
                if target_sgs and int(sg) in target_sgs:
                    hit = True
                if int(sg) in set(int(x) for x in PLAN_C_TARGET_SGS):
                    hit = True

                poscar_path = maybe_write_trial_structure(
                    trial=trial,
                    out_dir=out_dir,
                    trial_id=trial_id,
                    kind="combo",
                    label=label,
                    sg=int(sg),
                    sg_sym=str(sg_sym),
                    pg=normalize_pointgroup_symbol(pg_trial),
                    hit=hit,
                    write_counts=write_counts,
                )
                if poscar_path is not None:
                    rec["poscar"] = poscar_path

                results.append(rec)

    return results


# =========================
# Entry point
# =========================

def main() -> None:
    print(f"[{PROGRAM_TAG}] {PROGRAM_NAME} v{PROGRAM_VERSION}")
    print(f"[{PROGRAM_TAG}] Exfoliating Crystallographic Plane (ECP) companion utility")
    print()

    if not Path(PARENT_POSCAR).is_file():
        raise FileNotFoundError(f"Parent POSCAR not found: {PARENT_POSCAR}")

    parent = read_poscar(PARENT_POSCAR)
    print(f"[INFO] 原始父相: {PARENT_POSCAR}, N = {parent.nsites}")

    sg_parent, sg_sym, pg_raw = identify_spacegroup(parent, symprec=SYMPREC_PARENT, angle_tolerance=ANGLE_TOLERANCE)
    pg = normalize_pointgroup_symbol(pg_raw)
    pg_hint = normalize_pointgroup_symbol(PARENT_POINT_GROUP_HINT)
    pg_use = pg_hint or pg

    print(f"[INFO] spglib 识别的父相 space-group: {sg_parent} ({sg_sym})")
    print(f"[INFO] spglib 识别的父相 point-group: {pg_raw} -> {pg} (用户 hint: {PARENT_POINT_GROUP_HINT} -> {pg_hint})")

    # Prepare output directory (unique run folder recommended)
    base_out = Path(OUTPUT_DIR)
    base_out.mkdir(parents=True, exist_ok=True)

    run_dir = base_out
    if CREATE_RUN_SUBDIR:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_stem = Path(PARENT_POSCAR).stem
        pg_tag = _safe_filename(str(pg_use))
        parts: List[str] = []
        if RUN_TAG:
            parts.append(_safe_filename(RUN_TAG))
        parts.append(_safe_filename(f"{parent_stem}_SG{int(sg_parent):03d}_PG{pg_tag}_{ts}"))
        run_dir = base_out / "_".join([p for p in parts if p])
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 输出目录: {run_dir.resolve()}")

    # Record minimal run info for reproducibility
    try:
        info_lines = [
            f"program: {PROGRAM_NAME} v{PROGRAM_VERSION}",
            f"parent_poscar: {PARENT_POSCAR}",
            f"parent_spacegroup: {sg_parent} ({sg_sym})",
            f"parent_pointgroup_spglib: {pg_raw}",
            f"parent_pointgroup_use: {pg_use}",
            f"symprec_parent: {SYMPREC_PARENT}",
            f"symprec_identify: {SYMPREC_IDENTIFY}",
            f"angle_tolerance: {ANGLE_TOLERANCE}",
            f"enable_parity_sectors: {ENABLE_PARITY_SECTORS}",
        ]
        (run_dir / "run_info.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"[WARN] failed to write run_info.txt: {e}")

    # Strict consistency checks to prevent wrong irrep labels later.
    # --- Layer-group / space-group / point-group consistency checks ---
    # Table20 provides a correspondence between layer-group numbers (1–80) and 3D space-group numbers (1–230).
    # We use it only for sanity checks / logging, to avoid mixing up:
    #   - "layer-group #39"  vs  "space-group #39".
    cand_lgs = SPACEGROUP_NO_TO_LAYERGROUP_NOS.get(int(sg_parent), [])
    if cand_lgs:
        if len(cand_lgs) == 1:
            print(f"[INFO] Table20: space-group #{sg_parent} 对应 layer-group #{cand_lgs[0]}")
        else:
            print(f"[INFO] Table20: space-group #{sg_parent} 对应多个 layer-group: {cand_lgs}")
    else:
        print(f"[WARN] Table20: 未找到 space-group #{sg_parent} 对应的 layer-group（可能不是标准 layer-group 对应的 SG 编号）。")

    if PARENT_SPACE_GROUP_NO_HINT is not None:
        if int(PARENT_SPACE_GROUP_NO_HINT) != int(sg_parent):
            msg = (
                f"[ERROR] 用户指定的 space-group #{PARENT_SPACE_GROUP_NO_HINT} 与 spglib 识别的 "
                f"space-group #{sg_parent} ({sg_sym}) 不一致。\n"
                "        为避免模式分解/irrep 标签错误，已中止。"
            )
            if STRICT_POINTGROUP_MATCH:
                raise RuntimeError(msg)
            print(msg.replace("[ERROR]", "[WARN]"))

    if PARENT_LAYER_GROUP_NO_HINT is not None:
        expected_sg = layergroup_expected_spacegroup_number(PARENT_LAYER_GROUP_NO_HINT)
        if expected_sg is None:
            raise ValueError(f"Invalid PARENT_LAYER_GROUP_NO_HINT={PARENT_LAYER_GROUP_NO_HINT}. Should be 1..80")

        if int(expected_sg) != int(sg_parent):
            msg = (
                f"[ERROR] Layer-group #{PARENT_LAYER_GROUP_NO_HINT} 在 Table20 中对应 space-group #{expected_sg}，"
                f"但当前结构 spglib 识别 space-group #{sg_parent} ({sg_sym}).\n"
                "        注意：layer-group 编号(1–80) 和 space-group 编号(1–230) 不是一回事！\n"
            )
            if cand_lgs:
                msg += f"        对于 space-group #{sg_parent}，Table20 对应 layer-group 可能是: {cand_lgs}.\n"
            msg += "        为避免模式分解/irrep 标签错误，已中止。"
            if STRICT_POINTGROUP_MATCH:
                raise RuntimeError(msg)
            print(msg.replace("[ERROR]", "[WARN]"))

        expected_pg = layergroup_expected_pointgroup(PARENT_LAYER_GROUP_NO_HINT)
        if expected_pg is None or expected_pg == "?":
            msg = (
                f"[ERROR] 无法从 layer-group #{PARENT_LAYER_GROUP_NO_HINT} 推断期望 point-group（Table20->SG #{expected_sg}）。\n"
                "        为避免模式分解/irrep 标签错误，已中止。"
            )
            if STRICT_POINTGROUP_MATCH:
                raise RuntimeError(msg)
            print(msg.replace("[ERROR]", "[WARN]"))
        else:
            if pg_use is not None and normalize_pointgroup_symbol(pg_use) != normalize_pointgroup_symbol(expected_pg):
                msg = (
                    f"[ERROR] Layer-group #{PARENT_LAYER_GROUP_NO_HINT} (Table20->SG #{expected_sg}) 期望 point-group='{expected_pg}', "
                    f"但当前 pg_use='{pg_use}'.\n"
                    "        为避免模式分解/irrep 标签错误，已中止。"
                )
                if STRICT_POINTGROUP_MATCH:
                    raise RuntimeError(msg)
                print(msg.replace("[ERROR]", "[WARN]"))


    if pg_hint is not None and pg is not None and normalize_pointgroup_symbol(pg_hint) != normalize_pointgroup_symbol(pg):
        msg = (
            f"[ERROR] PARENT_POINT_GROUP_HINT='{pg_hint}' 与 spglib 检测点群 '{pg}' 不一致。\n"
            "        为避免 irrep 标签错误，建议修正 hint 或检查父相结构/容差。"
        )
        if STRICT_POINTGROUP_MATCH:
            raise RuntimeError(msg)
        print(msg.replace("[ERROR]", "[WARN]"))

    if pg_use is None:
        raise RuntimeError(
            "无法从 spglib 识别 point-group，且未提供 PARENT_POINT_GROUP_HINT。\n"
            "请设置 PARENT_POINT_GROUP_HINT (例如 'mm2','4mm','-3m','6/mmm' 等)，"
            "以保证 Mulliken irrep 标签映射正确。"
        )

    rotations, translations = get_symmetry_ops(parent, symprec=SYMPREC_PARENT, angle_tolerance=ANGLE_TOLERANCE)

    print(f"[INFO] 开始构造 Landau 模式基（含 @G/@R parity 扇区），并将 IRxx 映射为 Mulliken 标签... point-group={pg_use}")
    landau_basis = build_landau_basis_auto_pointgroup(
        crys=parent,
        rotations=rotations,
        translations=translations,
        symprec_map=SYMPREC_PARENT,
        point_group=pg_use,
        enable_parity=ENABLE_PARITY_SECTORS,
    )

    # Report basis sizes
    if not landau_basis:
        raise RuntimeError("Landau basis is empty. Check symmetry detection and point group assumptions.")

    for k in sorted(landau_basis.keys()):
        print(f"  irrep {k}: {len(landau_basis[k])} 个模式向量")

    # Run scan
    results = scan_landau_space_guided(parent, landau_basis, out_dir=run_dir)

    # Simple summary
    sgs = [r["sg"] for r in results]
    uniq = sorted(set(sgs))
    print()
    print("================= 扫描结束 Summary =================")
    print(f"[INFO] total trials = {len(results)}")
    print(f"[INFO] unique space-groups found = {uniq}")
    print("===================================================")

# Save results
try:
    import json
    out_json = Path(run_dir) / "scan_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[INFO] results saved to: {out_json}")
except Exception as e:
    print(f"[WARN] failed to save JSON results: {e}")

if WRITE_RESULTS_CSV:
    try:
        import csv
        out_csv = Path(run_dir) / "scan_results.csv"
        # Collect all keys for a stable header
        keys: List[str] = []
        for r in results:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"[INFO] results saved to: {out_csv}")
    except Exception as e:
        print(f"[WARN] failed to save CSV results: {e}")


if __name__ == "__main__":
    main()
