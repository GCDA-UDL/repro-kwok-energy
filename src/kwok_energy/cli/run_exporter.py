"""run-exporter: serve replayed signals + predicted power as Prometheus metrics.

Loads a node-profile map + coeffs, builds an ``EnergyExporter``, and serves
``/metrics`` over HTTP. A background ticker advances logical time and runs the
model; the HTTP handler only serializes the cache (I4/I5).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from ..exporter import EnergyExporter, start_server
from ..replay import load_map


def _load_coeffs(path: str | Path) -> dict[str, object]:
    """Load a coeffs JSON file and return a platform_id → Coeffs map."""
    from ..calibration import Coeffs

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    coeffs = Coeffs(
        platform_id=data["platform_id"],
        covered_components=frozenset(data["covered_components"]),
        transform=data["transform"],
        components=data["components"],
        provenance=data.get("provenance", {}),
    )
    return {coeffs.platform_id: coeffs}


def build_exporter(
    *,
    map_path: str | Path,
    coeffs_path: str | Path,
    platform_override: str | None = None,
) -> EnergyExporter:
    """Build an ``EnergyExporter`` from a node map + coeffs file.

    ``platform_override`` (for testing) registers the coeffs under a different
    platform_id so the exporter rejects unknown platforms loudly.
    """

    source = load_map(map_path)
    coeffs_map = _load_coeffs(coeffs_path)
    if platform_override is not None:
        # Re-key the coeffs map under the override id (test seam for unknown-platform)
        coeffs_map = {platform_override: next(iter(coeffs_map.values()))}
    return EnergyExporter(source=source, coeffs_map=coeffs_map)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    map_path: str | None = None
    coeffs_path: str | None = None
    port = 9100
    interval = 1.0
    speed = 1.0

    i = 0
    while i < len(args):
        if args[i] == "--map" and i + 1 < len(args):
            map_path = args[i + 1]
            i += 2
        elif args[i] == "--coeffs" and i + 1 < len(args):
            coeffs_path = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--interval" and i + 1 < len(args):
            interval = float(args[i + 1])
            i += 2
        elif args[i] == "--speed" and i + 1 < len(args):
            speed = float(args[i + 1])
            i += 2
        else:
            i += 1

    if not map_path or not coeffs_path:
        sys.stderr.write(
            "usage: run-exporter --map <node_profile_map.json> --coeffs <coeffs.json> "
            "[--port 9100] [--interval 1.0] [--speed 1.0]\n"
        )
        return 2

    exp = build_exporter(map_path=map_path, coeffs_path=coeffs_path)
    server = start_server(exp, port=port)

    # Background ticker: advance logical time and run the model (I4/I5)
    stop = threading.Event()

    def ticker() -> None:
        logical_t = 0.0
        while not stop.wait(interval):
            logical_t += interval * speed
            exp.tick(logical_t)

    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()

    sys.stderr.write(
        f"run-exporter: serving /metrics on :{port} "
        f"(interval={interval}s, speed={speed}, nodes={len(exp._source.nodes())})\n"
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        server.shutdown()
        sys.stderr.write("run-exporter: stopped\n")
    return 0


__all__ = ["build_exporter", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
