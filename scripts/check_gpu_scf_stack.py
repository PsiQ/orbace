#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Report CUDA / CuPy / gpu4pyscf alignment for OrbACE GPU SCF.

Run in the same environment you use for scripts/prepare.py:
  python scripts/check_gpu_scf_stack.py

Exit code 0 always (informational). Suggests pip install lines for mismatches.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path


def _nvidia_smi_driver_cuda() -> tuple[str | None, str | None]:
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None, None
        text = out.stdout
        # Driver Version: 550.54.15
        dm = re.search(r"Driver Version:\s*(\S+)", text)
        # CUDA Version: 12.4
        cm = re.search(r"CUDA Version:\s*(\S+)", text)
        return (dm.group(1) if dm else None, cm.group(1) if cm else None)
    except (OSError, subprocess.TimeoutExpired):
        return None, None


def _torch_cuda() -> str | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return getattr(torch.version, "cuda", None)
    except Exception:
        return None


def _cupy_info() -> dict:
    info: dict = {"import_ok": False}
    try:
        import cupy

        info["import_ok"] = True
        info["cupy_file"] = cupy.__file__
        try:
            ver = cupy.cuda.runtime.runtimeGetVersion()
            major = ver // 1000
            minor = (ver % 1000) // 10
            info["cupy_runtime_version"] = f"{major}.{minor}"
            info["cupy_runtime_major"] = major
        except Exception as e:
            info["cupy_runtime_error"] = str(e)
        try:
            show = cupy.show_config()
            if show:
                info["show_config"] = show
        except Exception:
            pass
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
    return info


def _pip_meta_for_cupy() -> str | None:
    try:
        import importlib.metadata as md
        for name in (
            "cupy-cuda12x",
            "cupy-cuda11x",
            "cupy-cuda13x",
            "cupy",
        ):
            try:
                return f"{name}=={md.version(name)}"
            except md.PackageNotFoundError:
                continue
    except Exception:
        pass
    return None


def _gpu4pyscf_meta() -> str | None:
    try:
        import importlib.metadata as md
        for name in (
            "gpu4pyscf-cuda12x",
            "gpu4pyscf-cuda11x",
            "gpu4pyscf-cuda13x",
            "gpu4pyscf",
        ):
            try:
                return f"{name}=={md.version(name)}"
            except md.PackageNotFoundError:
                continue
    except Exception:
        pass
    return None


def _suggested_wheels(cuda_major: int | None) -> str:
    if cuda_major is None:
        return "Could not infer CUDA major; use nvidia-smi 'CUDA Version' and pick:\n" \
            "  CUDA 11.x -> pip install cupy-cuda11x gpu4pyscf-cuda11x\n" \
            "  CUDA 12.x -> pip install cupy-cuda12x gpu4pyscf-cuda12x\n" \
            "  CUDA 13.x -> pip install cupy-cuda13x gpu4pyscf-cuda13x"
    m = {11: "cupy-cuda11x gpu4pyscf-cuda11x",
         12: "cupy-cuda12x gpu4pyscf-cuda12x",
         13: "cupy-cuda13x gpu4pyscf-cuda13x"}
    if cuda_major in m:
        return f"pip install {m[cuda_major]}"
    return f"CUDA major {cuda_major}: check PyPI for cupy-cuda{cuda_major}x / gpu4pyscf-cuda{cuda_major}x (names may vary)."


def main() -> None:
    print("=== OrbACE GPU SCF stack check ===\n")

    drv, smi_cuda = _nvidia_smi_driver_cuda()
    if drv or smi_cuda:
        print(f"nvidia-smi:  Driver {drv or '?'}  |  reported CUDA {smi_cuda or '?'}")
    else:
        print("nvidia-smi:  (not available or failed)")

    tc = _torch_cuda()
    print(f"PyTorch:     cuda available -> {tc or 'N/A (no GPU or no torch)'}")

    cupy_d = _cupy_info()
    if cupy_d.get("import_ok"):
        print(f"CuPy:        OK  {cupy_d.get('cupy_file', '')}")
        if "cupy_runtime_version" in cupy_d:
            print(f"             CuPy CUDA runtime API version: {cupy_d['cupy_runtime_version']} "
                  f"(major {cupy_d.get('cupy_runtime_major')})")
        if "cupy_runtime_error" in cupy_d:
            print(f"             runtime probe error: {cupy_d['cupy_runtime_error']}")
    else:
        print(f"CuPy:        NOT OK  {cupy_d.get('import_error', '')}")

    pip_cupy = _pip_meta_for_cupy()
    pip_g4 = _gpu4pyscf_meta()
    print(f"pip (CuPy):  {pip_cupy or 'none of cupy-cuda11x/12x/13x/cupy found'}")
    print(f"pip (g4):    {pip_g4 or 'none of gpu4pyscf-cuda11x/12x/13x/gpu4pyscf found'}")

    # Infer target major: prefer CuPy runtime, else parse nvidia-smi CUDA Version
    major: int | None = cupy_d.get("cupy_runtime_major")
    if major is None and smi_cuda:
        try:
            major = int(float(smi_cuda))
        except ValueError:
            major = None
    if major is None and tc:
        try:
            major = int(tc.split(".")[0])
        except (ValueError, IndexError):
            major = None

    print("\n--- Suggested matching wheels (same CUDA major for CuPy and gpu4pyscf) ---")
    print(_suggested_wheels(major))

    if pip_cupy and pip_g4:
        cmaj = None
        for m in (13, 12, 11):
            if f"cuda{m}x" in pip_cupy.replace("_", "-"):
                cmaj = m
                break
        gmaj = None
        for m in (13, 12, 11):
            if f"cuda{m}x" in pip_g4.replace("_", "-"):
                gmaj = m
                break
        if cmaj is not None and gmaj is not None and cmaj != gmaj:
            print(f"\nWARNING: CuPy wheel looks like CUDA {cmaj}x but gpu4pyscf wheel looks like CUDA {gmaj}x — reinstall both to the same line.")

    # Same sys.path hygiene as orbace.data.scf_backend (avoid local repo shadowing pip wheel).
    _strip = None
    try:
        _scf = Path(__file__).resolve().parents[1] / "orbace" / "data" / "scf_backend.py"
        spec = importlib.util.spec_from_file_location("_orbace_scf_backend_strip", _scf)
        if spec and spec.loader:
            _mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_mod)
            _strip = getattr(_mod, "_strip_shadow_gpu4pyscf_from_sys_path", None)
    except Exception:
        pass

    path_bak = sys.path[:]
    try:
        if _strip is not None:
            _removed, _kept = _strip()
            if _removed:
                sys.path[:] = _kept
                print(
                    f"\n(check script) Dropped {len(_removed)} non-site-packages gpu4pyscf "
                    f"path(s) for import test: {_removed}"
                )
        import gpu4pyscf  # noqa: F401
        print("\nimport gpu4pyscf: OK")
    except Exception as e:
        print(f"\nimport gpu4pyscf: FAILED  {type(e).__name__}: {e}")
        es = str(e).lower()
        if "cusolver" in es or "libnvrtc" in es:
            print(
                "\nIf you use cupy-cuda12x from pip, ensure the linker sees the CUDA toolkit "
                "libraries, e.g.:\n"
                "  export LD_LIBRARY_PATH=<CUDA_ROOT>/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}\n"
                "or `module load cuda/...` on your cluster. "
                "Sanity check: python -c \"from cupy_backends.cuda.libs import cusolver\""
            )
    finally:
        sys.path[:] = path_bak

    print("\nPYTHONPATH:", os.environ.get("PYTHONPATH", "(unset)"))
    print("(OrbACE strips non-site-packages gpu4pyscf trees during its probe; local repos on PYTHONPATH can still confuse other tools.)")


if __name__ == "__main__":
    main()
