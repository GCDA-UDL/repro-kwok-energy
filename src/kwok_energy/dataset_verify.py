"""Verify checked-in dataset manifest file hashes.

The verifier treats dataset files as lineage artifacts only. It never reads traces as
model inputs and never calls prediction code, preserving the non-circularity boundary
(I1) while checking R5 data lineage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a dataset manifest cannot be interpreted."""


@dataclass(frozen=True)
class VerificationFailure:
    path: str
    kind: str
    expected: str | None = None
    actual: str | None = None


@dataclass(frozen=True)
class ManifestVerification:
    manifest: Path
    checked: int
    failures: list[VerificationFailure]

    @property
    def ok(self) -> bool:
        return not self.failures


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest for a file."""

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_files(manifest_data: dict[str, Any]) -> dict[str, str]:
    files = manifest_data.get("files")
    if not isinstance(files, dict) or not files:
        raise ManifestError("manifest must contain a non-empty object field 'files'")

    extracted: dict[str, str] = {}
    for rel, digest in files.items():
        if not isinstance(rel, str) or not rel:
            raise ManifestError("manifest file paths must be non-empty strings")
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise ManifestError(f"manifest file path escapes bundle: {rel}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ManifestError(f"manifest sha256 for {rel} must be a 64-character hex string")
        # Validate hex by parsing; keep lowercase for deterministic comparisons.
        int(digest, 16)
        extracted[rel] = digest.lower()
    return dict(sorted(extracted.items()))


def verify_manifest(path: str | Path) -> ManifestVerification:
    """Verify one dataset manifest and return a deterministic result object."""

    manifest = Path(path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a JSON object")
    files = _extract_files(data)

    failures: list[VerificationFailure] = []
    bundle_dir = manifest.parent
    for rel, expected in files.items():
        candidate = bundle_dir / rel
        if not candidate.exists():
            failures.append(VerificationFailure(path=rel, kind="missing", expected=expected))
            continue
        actual = sha256_file(candidate)
        if actual != expected:
            failures.append(
                VerificationFailure(
                    path=rel,
                    kind="sha256_mismatch",
                    expected=expected,
                    actual=actual,
                )
            )

    return ManifestVerification(manifest=manifest, checked=len(files), failures=failures)


def verify_manifests(paths: Sequence[str | Path]) -> list[ManifestVerification]:
    """Verify manifests in stable path order."""

    return [verify_manifest(path) for path in sorted(Path(p) for p in paths)]


__all__ = [
    "ManifestError",
    "ManifestVerification",
    "VerificationFailure",
    "sha256_file",
    "verify_manifest",
    "verify_manifests",
]
