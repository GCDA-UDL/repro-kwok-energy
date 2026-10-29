"""Provenance (Phase 4, R2).

Captures reproducibility metadata into the fitted ``Coeffs`` object and supplies
the hash primitive used for input-trace / platform-config lineage (R5).

The full provenance block lives on ``Coeffs`` (Phase 4/9). Per-row records
(``ValidationRow``, result CSVs) carry the R2 **lineage subset** —
``coeffs_git_sha`` (= ``git_sha``), input hashes, ``schema_version``,
``created_at`` — exactly reconciled with PLAN.md §2 R2 and Phase 10.

Operations (named types, see PLAN.md §3 and the provenance spec):
- ``git_sha() -> str`` — commit sha (or a documented sentinel off-git).
- ``hash_file(path) -> digest`` — SHA-256 over raw file bytes.
- ``host_metadata() -> {kernel, cpu, gpu, nvml_driver}``.
- ``build_block(...) -> provenance`` — the full block stamped on ``Coeffs``.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

# The provenance block schema version (independent of the trace schema_version).
SCHEMA_VERSION = "1"

# Full block keys (live on Coeffs). The lineage subset (per-row) is the named
# cross-section: git_sha -> coeffs_git_sha + the two input hashes + schema_version
# + created_at.
_FULL_BLOCK_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "git_sha",
        "created_at",
        "solver",
        "transform_kind",
        "host",
        "calibration_trace_sha256",
        "platform_config_sha256",
    }
)

# Off-git sentinel (documented): returned when not under git or git is unavailable.
_GIT_SENTINEL = "not-under-git"


def git_sha() -> str:
    """Return the git commit sha of the working tree, or a documented sentinel.

    Deterministic for a given tree state (R1). Never raises — provenance must
    degrade gracefully off-git rather than break the pipeline.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return _GIT_SENTINEL
    if result.returncode != 0 or not result.stdout.strip():
        return _GIT_SENTINEL
    return result.stdout.strip()


def hash_file(path: str | Path) -> str:
    """Stable SHA-256 digest over a file's raw bytes (R2/R5).

    Identical file -> identical digest; any byte change -> different digest.
    This hashes the FILE (not the parsed canonical form used by ``hash_trace``);
    it answers "did this input file change?" for provenance stamping.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def host_metadata() -> dict[str, str]:
    """Capture runtime host metadata so a bench run is traceable (R3).

    Returns ``{kernel, cpu, gpu, nvml_driver}``. GPU/driver fields degrade to a
    documented sentinel when NVML is unavailable (never raises).
    """
    cpu = platform.processor() or platform.machine() or "unknown"
    gpu, nvml_driver = _gpu_metadata()
    return {
        "kernel": f"{platform.system()} {platform.release()}",
        "cpu": cpu,
        "gpu": gpu,
        "nvml_driver": nvml_driver,
    }


def _gpu_metadata() -> tuple[str, str]:
    """Best-effort GPU model + NVML driver; sentinels if unavailable."""
    try:
        import pynvml  # noqa: F401

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8", "replace")
            pynvml.nvmlShutdown()
            return str(name), str(driver)
        except Exception:  # pragma: no cover - environment dependent
            return "nvml-present-unreadable", "unknown"
    except Exception:  # pragma: no cover - pynvml absent
        return "unavailable", "unavailable"


def build_block(
    *,
    solver: str,
    transform_kind: str,
    calibration_trace_sha256: str,
    platform_config_sha256: str,
    host: Mapping[str, str] | None = None,
    schema_version: str = SCHEMA_VERSION,
    created_at: str | None = None,
) -> dict[str, str | Mapping[str, str]]:
    """Assemble the full provenance block stamped onto ``Coeffs``.

    The block is a superset of the per-row lineage subset; ``git_sha`` becomes
    ``coeffs_git_sha`` on per-row records (reconciled with Phase 10 ValidationRow).
    ``created_at`` defaults to a UTC ISO-8601 timestamp (R1: deterministic given
    the call instant; callers fitting coeffs pass a pinned timestamp for
    byte-stable coeffs.json).
    """
    block: dict[str, str | Mapping[str, str]] = {
        "schema_version": schema_version,
        "git_sha": git_sha(),
        "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "solver": solver,
        "transform_kind": transform_kind,
        "host": host if host is not None else host_metadata(),
        "calibration_trace_sha256": calibration_trace_sha256,
        "platform_config_sha256": platform_config_sha256,
    }
    assert _FULL_BLOCK_KEYS <= set(block), "provenance block missing required keys"
    return block


__all__ = [
    "SCHEMA_VERSION",
    "git_sha",
    "hash_file",
    "host_metadata",
    "build_block",
]
