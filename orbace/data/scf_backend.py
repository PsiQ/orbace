# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""SCF backend dispatch: regular PySCF (CPU) by default, optional gpu4pyscf.

Default OrbACE workflows pass ``device=None`` or ``"cpu"`` so Hartree–Fock runs
on CPU. Pass ``device="cuda"`` to request gpu4pyscf when CuPy + gpu4pyscf import.

Usage:
    from orbace.data.scf_backend import run_hf

    mf = run_hf(mol, device="cpu")    # PySCF on CPU (typical)
    mf = run_hf(mol, device="cuda")   # gpu4pyscf if available, else CPU
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _cuda_home_for_preload() -> Optional[str]:
    """Toolkit root used to preload libcublas before CuPy (optional)."""
    for key in ("ORBACE_CUDA_HOME", "CUDA_HOME", "CUDA_PATH", "CUDATOOLKIT_HOME"):
        v = os.environ.get(key)
        if v and os.path.isdir(v):
            return os.path.normpath(v)
    return None


def _preload_toolkit_cublas_before_cupy() -> None:
    """dlopen toolkit libcublas(+Lt) with RTLD_GLOBAL so cusolver finds matching symbols.

    When conda/PyTorch `nvidia-*` wheels appear earlier on the library search path than the
    HPC CUDA toolkit, `libcusolver` can load while `libcublas` resolves to an incompatible
    build (`undefined symbol ... libcublas.so.12`). Loading the toolkit's BLAS first fixes
    many cases without changing global shell `LD_LIBRARY_PATH` order.

    Disable with ``ORBACE_SKIP_CUDA_PRELOAD=1``. Requires ``CUDA_HOME`` / ``CUDA_PATH`` /
    ``ORBACE_CUDA_HOME`` (e.g. from ``module load cuda`` or ``export CUDA_HOME=...``).
    """
    if os.environ.get("ORBACE_SKIP_CUDA_PRELOAD", "").lower() in ("1", "true", "yes"):
        return
    if not sys.platform.startswith("linux"):
        return
    root = _cuda_home_for_preload()
    if root is None:
        return
    import ctypes

    mode = getattr(os, "RTLD_NOW", 2)
    mode |= getattr(os, "RTLD_GLOBAL", 0x00100)
    lib_dirs = [
        os.path.join(root, "targets", "x86_64-linux", "lib"),
        os.path.join(root, "lib64"),
    ]
    to_load: List[str] = []
    seen = set()
    for d in lib_dirs:
        if not os.path.isdir(d):
            continue
        for name in (
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcublas.so.11",
            "libcublasLt.so.11",
        ):
            fp = os.path.join(d, name)
            if os.path.isfile(fp) and fp not in seen:
                to_load.append(fp)
                seen.add(fp)
    n_ok = 0
    for fp in to_load:
        try:
            ctypes.CDLL(fp, mode=mode)
            n_ok += 1
        except OSError as err:
            logger.debug("CUDA preload skip %s: %s", fp, err)
    if n_ok:
        logger.debug(
            "Preloaded %d CUDA toolkit BLAS librar(y/ies) from %s (RTLD_GLOBAL) before CuPy",
            n_ok,
            root,
        )


def _is_pip_style_dir(path: str) -> bool:
    """True for typical install roots (pip / conda), not arbitrary PYTHONPATH trees."""
    n = path.replace("\\", "/").lower()
    return "site-packages" in n or "dist-packages" in n


def _strip_shadow_gpu4pyscf_from_sys_path() -> Tuple[List[str], List[str]]:
    """Drop sys.path entries that expose a *local* gpu4pyscf tree (shadows pip wheels).

    Many users have ``export PYTHONPATH=~/gpu4pyscf`` for development; that prepends a
    source checkout that may target a different CUDA than ``gpu4pyscf-cuda12x`` + CuPy,
    causing ``libcusolver.so.11`` errors. Removing those entries lets the pip package win.

    Set ``ORBACE_KEEP_LOCAL_GPU4PYSCF_PATH=1`` to skip stripping (editable / dev installs).
    Returns (removed_paths, kept_paths) for logging.
    """
    if os.environ.get("ORBACE_KEEP_LOCAL_GPU4PYSCF_PATH", "").lower() in (
        "1", "true", "yes",
    ):
        return [], list(sys.path)

    removed: List[str] = []
    kept: List[str] = []
    for p in sys.path:
        try:
            if p == "":
                norm = os.path.normpath(os.path.abspath(os.getcwd()))
            else:
                norm = os.path.normpath(os.path.abspath(p))
            has_g4 = os.path.isdir(os.path.join(norm, "gpu4pyscf"))
        except OSError:
            has_g4 = False
            norm = p or ""
        if has_g4 and norm and not _is_pip_style_dir(norm):
            removed.append(p)
        else:
            kept.append(p)
    return removed, kept

_GPU_CHECKED = False
_GPU_OK = False
_GPU_FAIL_REASON: Optional[str] = None
_GPU_FAIL_HINT: Optional[str] = None


def _hint_for_import_error(exc: BaseException) -> Optional[str]:
    """Human-readable follow-up when probe fails (CUDA / wheel mismatch)."""
    text = f"{type(exc).__name__}: {exc}"
    tl = text.lower()
    if "undefined symbol" in tl and ("cusolver" in tl or "cublas" in tl or "libcublas" in tl):
        return (
            "CUDA libraries are inconsistent (cusolver vs cublas from different roots or search "
            "order). Try, in order:\n"
            "  1) export CUDA_HOME=<CUDA_ROOT>   # same tree as your toolkit, e.g. /opt/apps/cuda/12.9.0\n"
            "     OrbACE will preload toolkit libcublas (RTLD_GLOBAL) before CuPy when CUDA_HOME/CUDA_PATH is set.\n"
            "  2) export LD_LIBRARY_PATH="
            "<CUDA_ROOT>/targets/x86_64-linux/lib:<CUDA_ROOT>/lib64:${LD_LIBRARY_PATH}\n"
            "     Put these **before** conda `nvidia-*` paths; use a fresh shell after `module load cuda`.\n"
            "  3) Or set ORBACE_SKIP_CUDA_PRELOAD=1 if preload causes issues, and rely on (2) only."
        )
    if "libcusolver.so" in tl or "cusolver.so" in tl:
        return (
            "Often the dynamic linker cannot see CUDA toolkit libraries. On clusters, run "
            "`module load cuda/...` or set e.g.\n"
            "  export LD_LIBRARY_PATH=<CUDA_ROOT>/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}\n"
            "then retry. Also test: python -c \"from cupy_backends.cuda.libs import cusolver\".\n"
            "If you use cupy-cuda12x but still see .so.11 after fixing LD_LIBRARY_PATH, try a "
            "clean reinstall: pip uninstall -y cupy cupy-cuda11x cupy-cuda12x && "
            "pip install cupy-cuda12x gpu4pyscf-cuda12x. "
            "Do not put a source gpu4pyscf checkout on PYTHONPATH unless developing it."
        )
    return None


def _check_gpu() -> bool:
    """Lazy one-time check for gpu4pyscf + cupy."""
    global _GPU_CHECKED, _GPU_OK, _GPU_FAIL_REASON, _GPU_FAIL_HINT
    if _GPU_CHECKED:
        return _GPU_OK
    _GPU_CHECKED = True
    _GPU_FAIL_REASON = None
    _GPU_FAIL_HINT = None
    path_backup = sys.path[:]
    try:
        removed, kept = _strip_shadow_gpu4pyscf_from_sys_path()
        if removed:
            sys.path[:] = kept
            logger.debug(
                "Temporarily ignoring %d sys.path entr(y/ies) with a non-wheel gpu4pyscf "
                "(so pip gpu4pyscf-cuda* can load): %s",
                len(removed),
                removed,
            )
        _preload_toolkit_cublas_before_cupy()
        import cupy  # noqa: F401
        import gpu4pyscf  # noqa: F401
        # Smoke-test that to_gpu()/to_cpu() actually work (catches API
        # mismatches like missing Mole.from_cpu in older gpu4pyscf).
        from pyscf import gto, scf as _scf
        _mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1)
        _mf = _scf.ROHF(_mol).to_gpu()
        _mf.to_cpu()
        del _mol, _mf
        _GPU_OK = True
        logger.info("gpu4pyscf backend detected — GPU SCF available")
    except Exception as e:
        _GPU_OK = False
        _GPU_FAIL_REASON = f"{type(e).__name__}: {e}"
        _GPU_FAIL_HINT = _hint_for_import_error(e)
        logger.warning(
            "GPU SCF unavailable (%s). Using PySCF on CPU.",
            _GPU_FAIL_REASON,
        )
        if _GPU_FAIL_HINT:
            logger.warning("%s", _GPU_FAIL_HINT)
        else:
            logger.warning(
                "Install matching pip wheels (e.g. cupy-cuda12x + gpu4pyscf-cuda12x), "
                "and avoid PYTHONPATH to a source gpu4pyscf checkout unless you use "
                "ORBACE_KEEP_LOCAL_GPU4PYSCF_PATH=1.",
            )
    finally:
        sys.path[:] = path_backup
    return _GPU_OK


def gpu_available() -> bool:
    return _check_gpu()


def gpu_unavailable_reason() -> Optional[str]:
    """If GPU SCF is not usable, return error + optional hint; else None."""
    _check_gpu()
    if _GPU_OK:
        return None
    if _GPU_FAIL_HINT:
        return f"{_GPU_FAIL_REASON}\n{_GPU_FAIL_HINT}"
    return _GPU_FAIL_REASON


def _wants_gpu(device: Optional[str]) -> bool:
    """Return True if the caller requested a CUDA device."""
    if device is None:
        return False
    return device.startswith("cuda") or device == "gpu"


def run_hf(mol, device: Optional[str] = None, max_cycle: int = 200, dm0=None):
    """Run restricted (or restricted-open) Hartree–Fock.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Pre-built molecule object.
    device : str | None
        ``"cuda"`` / ``"gpu"`` to request gpu4pyscf, anything else → CPU.
    max_cycle : int
        Maximum SCF iterations.
    dm0 : np.ndarray | None
        Optional initial-guess density matrix. In MD, passing the previous
        step's converged density matrix warm-starts SCF (far fewer iterations),
        which is exact -- the converged solution is unchanged. Shape must match
        the current ``mol`` (same atoms/basis); a mismatch is ignored.

    Returns
    -------
    mf : converged mean-field object (CPU PySCF flavour).
        If gpu4pyscf was used, the result is converted back to CPU
        via ``mf.to_cpu()`` so downstream code (integrals, localization)
        works unchanged.
    """
    # CPU path first: never touch CuPy/gpu4pyscf unless CUDA SCF was explicitly requested.
    if not _wants_gpu(device):
        return _run_hf_cpu(mol, max_cycle, dm0=dm0)
    if not _check_gpu():
        return _run_hf_cpu(mol, max_cycle, dm0=dm0)
    return _run_hf_gpu(mol, max_cycle, dm0=dm0)


def run_dft(mol, xc: str, device: Optional[str] = None, max_cycle: int = 200, dm0=None):
    """Run restricted (or restricted-open) Kohn-Sham DFT.

    Mirrors :func:`run_hf` but for a KS functional ``xc`` (e.g. ``"wb97x"``,
    ``"pbe"``, ``"b3lyp"``). Used for DFT delta-learning baselines whose analytic
    nuclear gradient (``mf.nuc_grad_method()``) gives far better forces than a
    minimal-basis HF baseline. Returns a converged CPU mean-field object.
    """
    if not _wants_gpu(device):
        return _run_dft_cpu(mol, xc, max_cycle, dm0=dm0)
    if not _check_gpu():
        return _run_dft_cpu(mol, xc, max_cycle, dm0=dm0)
    return _run_dft_gpu(mol, xc, max_cycle, dm0=dm0)


def _run_dft_cpu(mol, xc: str, max_cycle: int, dm0=None):
    from pyscf import dft, scf

    if mol.spin > 0:
        mf = dft.ROKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-9
    mf.max_cycle = max_cycle
    mf.kernel(dm0=_valid_dm0(mol, mf, dm0))
    if not mf.converged:
        mf = scf.newton(mf).run()
    return mf


def _run_dft_gpu(mol, xc: str, max_cycle: int, dm0=None):
    from pyscf import dft as cpu_dft

    logger.debug("Running KS-DFT on GPU via gpu4pyscf (xc=%s)", xc)

    if mol.spin > 0:
        mf_cpu = cpu_dft.ROKS(mol)
    else:
        mf_cpu = cpu_dft.RKS(mol)
    mf_cpu.xc = xc

    try:
        mf_gpu = mf_cpu.to_gpu()
        mf_gpu.conv_tol = 1e-9
        mf_gpu.max_cycle = max_cycle
        guess = _valid_dm0(mol, mf_gpu, dm0)
        if guess is not None:
            import cupy as _cp
            guess = _cp.asarray(guess)
        mf_gpu.kernel(dm0=guess)
        if not mf_gpu.converged:
            mf_gpu = mf_gpu.newton().run()
        return mf_gpu.to_cpu()
    except Exception as e:
        global _GPU_OK
        _GPU_OK = False
        logger.warning(f"gpu4pyscf DFT failed ({e}), disabling GPU and falling back to CPU")
        return _run_dft_cpu(mol, xc, max_cycle, dm0=dm0)


def _valid_dm0(mol, mf, dm0):
    """Return dm0 only if its shape is consistent with this mol/mf, else None."""
    if dm0 is None:
        return None
    try:
        import numpy as _np
        nao = mol.nao_nr()
        arr = _np.asarray(dm0)
        # RHF: (nao, nao); ROHF/UHF guess: (2, nao, nao). Accept either if the
        # trailing dims match the current basis size.
        if arr.shape[-1] == nao and arr.shape[-2] == nao:
            return dm0
    except Exception:
        pass
    return None


def _run_hf_cpu(mol, max_cycle: int, dm0=None):
    from pyscf import scf

    if mol.spin > 0:
        mf = scf.ROHF(mol)
    else:
        mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = max_cycle
    mf.kernel(dm0=_valid_dm0(mol, mf, dm0))
    if not mf.converged:
        mf = scf.newton(mf).run()
    return mf


def _run_hf_gpu(mol, max_cycle: int, dm0=None):
    from pyscf import scf as cpu_scf

    logger.debug("Running HF on GPU via gpu4pyscf")

    if mol.spin > 0:
        mf_cpu = cpu_scf.ROHF(mol)
    else:
        mf_cpu = cpu_scf.RHF(mol)

    try:
        mf_gpu = mf_cpu.to_gpu()
        mf_gpu.conv_tol = 1e-10
        mf_gpu.max_cycle = max_cycle
        guess = _valid_dm0(mol, mf_gpu, dm0)
        if guess is not None:
            import cupy as _cp
            guess = _cp.asarray(guess)
        mf_gpu.kernel(dm0=guess)
        if not mf_gpu.converged:
            mf_gpu = mf_gpu.newton().run()
        return mf_gpu.to_cpu()
    except Exception as e:
        global _GPU_OK
        _GPU_OK = False
        logger.warning(f"gpu4pyscf HF failed ({e}), disabling GPU and falling back to CPU")
        return _run_hf_cpu(mol, max_cycle, dm0=dm0)
