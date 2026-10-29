"""gen-cluster: generate KWOK/replay/Prometheus identity artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from ..gen_cluster import generate, write_artifacts


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    config_path: str | None = None
    output_dir: str | None = None

    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            i += 1

    if not config_path or not output_dir:
        sys.stderr.write("usage: gen-cluster --config <config.yaml|json> --output-dir <dir>\n")
        return 2

    cfg_file = Path(config_path)
    text = cfg_file.read_text(encoding="utf-8")
    config = json.loads(text) if cfg_file.suffix == ".json" else yaml.safe_load(text)
    paths = write_artifacts(generate(config, config_path=cfg_file), output_dir)
    sys.stderr.write(
        "gen-cluster: wrote "
        f"{paths.node_profile_map}, {paths.kwok_fake_nodes}, "
        f"{paths.kwok_fake_pods}, {paths.prometheus_targets}\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
