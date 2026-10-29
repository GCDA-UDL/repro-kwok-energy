"""verify-dataset: verify checked-in dataset manifest hashes."""

from __future__ import annotations

import sys
from pathlib import Path

from kwok_energy.dataset_verify import ManifestError, verify_manifests


def _default_manifests() -> list[Path]:
    return sorted(Path("dataset").glob("*/manifest.json"))


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    manifests = [Path(arg) for arg in args] if args else _default_manifests()
    if not manifests:
        sys.stderr.write("verify-dataset: no manifest paths found\n")
        return 2

    try:
        results = verify_manifests(manifests)
    except (OSError, ManifestError, ValueError, TypeError) as exc:
        sys.stderr.write(f"verify-dataset: {exc}\n")
        return 2

    total_checked = 0
    ok = True
    for result in results:
        total_checked += result.checked
        if result.ok:
            print(f"ok {result.manifest}: {result.checked} files")
            continue
        ok = False
        print(f"FAIL {result.manifest}: {len(result.failures)} failures")
        for failure in result.failures:
            if failure.kind == "missing":
                print(f"  missing {failure.path}")
            else:
                print(
                    f"  {failure.kind} {failure.path}: "
                    f"expected={failure.expected} actual={failure.actual}"
                )

    print(f"checked {total_checked} files across {len(results)} manifests")
    return 0 if ok else 1


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
