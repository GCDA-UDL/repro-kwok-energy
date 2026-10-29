"""Archive an experiment run as a citable, open-access dataset bundle.

Copies the raw per-interval traces + clean results + fitted coeffs into a tracked,
self-describing folder `dataset/<platform>_<YYYYMMDD>/`, with a `manifest.json`
(hardware, software versions, git sha, per-file sha256). `dataset/` is version-
controlled (unlike `data/`), so each run is a versioned, checksummed dataset ready
for a Zenodo/open-access data publication.

    python tools/e2e/archive_dataset.py --platform a4000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform as plat
import shutil
import socket
import subprocess
import sys
from datetime import date, timezone, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kwok_energy import provenance  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, timeout=5).strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive a run as an open-access dataset bundle.")
    ap.add_argument("--platform", required=True)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--capture", type=Path, default=None,
                    help="capture_v2 output dir (protocol v2); hardware is taken from its "
                         "session_manifest.json, not from the archiving host")
    ap.add_argument("--analysis", type=Path, default=Path("data/q1/v2"),
                    help="q1_analysis/model_selection output dir (with --capture)")
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="additional result dir copied into results/ (with --capture)")
    ap.add_argument("--date", default=None, help="bundle date YYYYMMDD (default: today)")
    a = ap.parse_args()
    if a.capture is not None:
        return archive_v2(a)

    data = Path(a.datadir)
    day = date.today().isoformat().replace("-", "")
    dst = Path(a.out) / f"{a.platform}_{day}"
    raw, res, co = dst / "raw", dst / "results", dst / "coeffs"
    for d in (raw, res, co):
        d.mkdir(parents=True, exist_ok=True)

    # copy artifacts
    for f in sorted((data / "e2e" / a.platform).glob("*.csv")):
        shutil.copy2(f, raw / f.name)
    for f in sorted((data / "results").glob(f"{a.platform}_*.csv")):
        shutil.copy2(f, res / f.name)
    coeffs_src = data / "coeffs" / f"{a.platform}.json"
    if coeffs_src.exists():
        shutil.copy2(coeffs_src, co / coeffs_src.name)

    # checksums of everything copied
    files = sorted(p for p in dst.rglob("*") if p.is_file() and p.name != "manifest.json")
    checksums = {str(p.relative_to(dst)).replace("\\", "/"): _sha256(p) for p in files}

    hw = provenance.host_metadata()

    def _val(x: object) -> object | None:
        return x if x and x != "unavailable" else None

    manifest = {
        "dataset": "KWOK hardware-grounded energy — calibration + held-out traces",
        "platform_id": a.platform,
        "date": day,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "git_sha": provenance.git_sha(),
        "schema_version": "1",
        "capture": {"reps": a.reps, "duration_s": a.duration},
        "hardware": {
            "cpu": _val(hw.get("cpu")) or _cmd(["bash", "-lc",
                   "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs"]),
            "gpu": _val(hw.get("gpu")) or _cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
            "kernel": _val(hw.get("kernel")) or plat.release(),
            "nvml_driver": _val(hw.get("nvml_driver"))
            or _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "mem": _cmd(["bash", "-lc", "free -h | awk '/Mem:/{print $2}'"]),
        },
        "software": {
            "python": plat.python_version(),
            "platform": plat.platform(),
        },
        "license": "CC-BY-4.0",
        "funding": "MICIU AEI/10.13039/501100011033 PID2023-146193OB-I00",
        "n_files": len(checksums),
        "files": checksums,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")

    print(f"archived {len(checksums)} files -> {dst}")
    print(f"  raw traces: {len(list(raw.glob('*.csv')))} | results: {len(list(res.glob('*.csv')))} "
          f"| coeffs: {len(list(co.glob('*.json')))}")
    print(f"Commit it:  git add {dst} && git commit -m \"dataset({a.platform}): {day}\" && git push")
    return 0


def _filter_csv(src: Path, dst: Path, keep) -> int:
    """Copy the rows of ``src`` for which ``keep(row)`` holds; returns the row count."""
    import csv
    with src.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        rows = [r for r in rd if keep(r)]
        fields = rd.fieldnames or []
    if rows:
        with dst.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def archive_v2(a: argparse.Namespace) -> int:
    """Protocol-v2 bundle: traces + session timeline + this platform's analysis rows."""
    cap: Path = a.capture
    sm = json.loads((cap / "session_manifest.json").read_text(encoding="utf-8"))
    p = a.platform
    if sm["platform_id"] != p:
        raise SystemExit(f"{cap}: platform_id {sm['platform_id']!r} != --platform {p!r}")
    day = a.date or date.today().isoformat().replace("-", "")
    dst = Path(a.out) / f"{p}_{day}"
    raw, res, co = dst / "raw", dst / "results", dst / "coeffs"
    for d in (raw, res, co):
        d.mkdir(parents=True, exist_ok=True)

    for f in sorted(cap.glob("*_r*.csv")) + [cap / "session.csv", cap / "session_manifest.json"]:
        shutil.copy2(f, raw / f.name)
    an: Path = a.analysis
    if (an / "per_workload.csv").is_file():
        _filter_csv(an / "per_workload.csv", res / "per_workload.csv",
                    lambda r: r["test_platform"] == p)
    if (an / "per_run.csv").is_file():
        _filter_csv(an / "per_run.csv", res / "per_run.csv", lambda r: r["test_platform"] == p)
    for name in ("summary.csv", "summary_components.csv"):
        if (an / name).is_file():
            _filter_csv(an / name, res / name, lambda r: r["test_platform"] == p)
    if (an / "model_selection.csv").is_file():
        _filter_csv(an / "model_selection.csv", res / "model_selection.csv",
                    lambda r: r["platform"] == p)
    if (an / "node_level.csv").is_file():
        _filter_csv(an / "node_level.csv", res / "node_level.csv", lambda r: r["platform"] == p)
    if (an / "selection.json").is_file():
        sel = json.loads((an / "selection.json").read_text(encoding="utf-8"))
        if p in sel:
            (res / "selection.json").write_text(json.dumps({p: sel[p]}, indent=2) + "\n",
                                                encoding="utf-8")
    for extra in a.extra:
        for f in sorted(extra.glob("*.csv")):
            shutil.copy2(f, res / f.name)
    # Prefer the evaluated model written by q1_analysis.py (the coefficients behind the
    # reported numbers); fall back to a fit-model output.
    coeffs_src = an / f"coeffs_{p}.json"
    if not coeffs_src.exists():
        coeffs_src = Path(a.datadir) / "coeffs" / f"{p}_v2.json"
    if coeffs_src.exists():
        shutil.copy2(coeffs_src, co / f"{p}.json")

    files = sorted(f for f in dst.rglob("*") if f.is_file() and f.name != "manifest.json")
    checksums = {str(f.relative_to(dst)).replace("\\", "/"): _sha256(f) for f in files}
    host = sm["host"]
    manifest = {
        "dataset": "KWOK hardware-grounded energy — calibration + held-out traces (protocol v2)",
        "platform_id": p,
        "date": day,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": host.get("hostname"),
        "git_sha": host.get("git_sha"),
        "git_sha_archive": provenance.git_sha(),
        "schema_version": sm.get("schema_version"),
        "protocol": sm.get("protocol"),
        "capture": {**sm.get("args", {}), "session_start_utc": sm["background_start"]["time_utc"],
                    "wall_s": sm.get("wall_s"), "status": sm.get("status")},
        "hardware": {
            "cpu": host.get("cpu_model"), "ncpu": host.get("ncpu"), "gpus": host.get("gpus"),
            "gpu_count": host.get("gpu_count"), "kernel": host.get("kernel"),
            "rapl_sockets": host.get("rapl_sockets"), "rapl_dram": host.get("rapl_dram"),
            "platform_meter": host.get("platform_meter"), "mem_proxy": host.get("mem_proxy"),
        },
        "software": {"python": host.get("python"), "torch": host.get("torch"),
                     "ollama_image": host.get("ollama_image"),
                     "ollama_version": host.get("ollama_version")},
        "license": "CC-BY-4.0",
        "funding": "MICIU AEI/10.13039/501100011033 PID2023-146193OB-I00",
        "n_files": len(checksums),
        "files": checksums,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    print(f"archived {len(checksums)} files -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
