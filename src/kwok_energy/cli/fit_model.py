"""fit-model: fit per-component energy models from a calibration trace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import calibration as cal
from ..features import retau_rows
from ..trace import read as read_trace


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 3:
        sys.stderr.write(
            "usage: fit-model --platform <id> --input <trace.csv,...> --output <coeffs.json>"
            " [--features v1|v2 | --selection <selection.json> [--variant base|hist]"
            " [--session <session.csv>]]\n"
        )
        return 2

    platform_id = "unknown"
    input_paths: list[str] = []
    output_path: str | None = None
    features = "v1"
    selection: str | None = None
    variant = "hist"
    session: str | None = None
    i = 0
    while i < len(args):
        if args[i] == "--platform" and i + 1 < len(args):
            platform_id = args[i + 1]
            i += 2
        elif args[i] == "--input" and i + 1 < len(args):
            input_paths = [p.strip() for p in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        elif args[i] == "--features" and i + 1 < len(args):
            features = args[i + 1]
            i += 2
        elif args[i] == "--selection" and i + 1 < len(args):
            selection = args[i + 1]
            i += 2
        elif args[i] == "--variant" and i + 1 < len(args):
            variant = args[i + 1]
            i += 2
        elif args[i] == "--session" and i + 1 < len(args):
            session = args[i + 1]
            i += 2
        else:
            i += 1

    if not input_paths or not output_path:
        sys.stderr.write("fit-model: --input and --output are required\n")
        return 2
    if features not in ("v1", "v2"):
        sys.stderr.write("fit-model: --features must be v1 (default) or v2 (adds gpu_util_ewma)\n")
        return 2

    feats = cal.FIT_FEATURES_V2 if features == "v2" else cal.FIT_FEATURES
    alpha = 0.01
    tau_s = None
    if selection:
        # Per-platform configuration chosen by tools/e2e/model_selection.py (LOWO CV on
        # the calibration data); overrides --features.
        if variant not in ("base", "hist"):
            sys.stderr.write("fit-model: --variant must be base or hist\n")
            return 2
        cfg = json.loads(Path(selection).read_text(encoding="utf-8"))[platform_id][variant]
        feats, alpha, tau_s = list(cfg["features"]), float(cfg["alpha"]), cfg.get("tau_s")
    if tau_s is not None and "gpu_util_ewma" in feats:
        # The traces carry the history signal at the capture tau; recompute it at the
        # selected tau on the session timeline, so the coefficients match the tau that
        # the payload declares and that every consumer applies.
        session_path = Path(session) if session else Path(input_paths[0]).parent / "session.csv"
        if not session_path.is_file():
            sys.stderr.write(f"fit-model: tau {tau_s} needs the session timeline; {session_path} "
                             "not found (pass --session)\n")
            return 2
        cal._check_not_heldout(input_paths)
        rows = retau_rows([r for p in input_paths for r in read_trace(p)],
                          read_trace(session_path), float(tau_s))
        coeffs = cal.fit(rows, platform_id=platform_id, features=feats, ridge_alpha=alpha,
                         calibration_trace_sha256=cal.combined_trace_sha256(input_paths))
    else:
        coeffs = cal.fit(input_paths, platform_id=platform_id, features=feats, ridge_alpha=alpha)
    payload = cal.to_payload(coeffs, tau_s)
    payload["provenance"]["ridge_alpha"] = alpha
    payload["provenance"]["calibration_traces"] = len(input_paths)
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8", newline="\n")
    sys.stderr.write(f"fit-model: wrote {len(coeffs.covered_components)} components to {output_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
