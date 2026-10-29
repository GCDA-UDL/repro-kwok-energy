"""Memory-bandwidth proxy sampler for the calibration capture (Linux, perf).

Populates the model's `mem_bandwidth_proxy` signal so the DRAM term can track
memory-bound workloads (the previous capture stubbed it to 0, which is why held-out
DRAM error was high under memory load). Measures over an interval with `perf stat`:

  1. uncore IMC CAS read+write counters (true DRAM traffic) -> method "perf_imc"
  2. LLC-load-misses as a correlated proxy                  -> method "perf_llc"
  3. otherwise just sleeps the interval                     -> method "unavailable" (0)

The absolute scale is irrelevant — calibration fits a coefficient to whatever the
proxy is, as long as it is CONSISTENT between calibration and held-out (same sampler).
`sample(dt)` BLOCKS ~dt (the perf measurement window), so it replaces the loop's sleep.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time

_IMC_EVENTS = "uncore_imc/cas_count_read/,uncore_imc/cas_count_write/"
_LLC_EVENT = "LLC-load-misses"
_NUM = re.compile(r"^\s*([0-9][0-9.,]*)")


class MemBandwidthSampler:
    """Measures a memory-bandwidth proxy over an interval via perf; graceful fallback."""

    def __init__(self) -> None:
        self._perf = shutil.which("perf")
        self._mode = "imc" if self._perf else "none"
        self.method = "unavailable"

    def _perf_sum(self, events: str, dt: float) -> float | None:
        try:
            p = subprocess.run(
                ["perf", "stat", "-a", "-x", ",", "-e", events, "--", "sleep", str(dt)],
                capture_output=True, text=True, timeout=dt + 10,
            )
        except Exception:
            return None
        total = 0.0
        found = False
        for line in p.stderr.splitlines():  # perf writes counts to stderr
            first = line.split(",", 1)[0].strip()
            if first in ("", "<not", "<not supported>", "<not counted>"):
                continue
            m = _NUM.match(first.replace(",", ""))
            if m:
                try:
                    total += float(m.group(1))
                    found = True
                except ValueError:
                    pass
        return total if found else None

    def sample(self, dt: float) -> tuple[float, str]:
        """Return (proxy_rate, method) measured over ~dt seconds. Blocks ~dt."""
        if self._mode == "imc":
            v = self._perf_sum(_IMC_EVENTS, dt)
            if v and v > 0:
                self.method = "perf_imc"
                return v / dt, "perf_imc"
            self._mode = "llc"  # IMC unsupported here -> downgrade once
        if self._mode == "llc":
            v = self._perf_sum(_LLC_EVENT, dt)
            if v and v > 0:
                self.method = "perf_llc"
                return v / dt, "perf_llc"
            self._mode = "none"  # neither works -> stop trying perf
        time.sleep(dt)
        self.method = "unavailable"
        return 0.0, "unavailable"


class ContinuousIMCSampler:
    """Memory-traffic proxy from one long-lived ``perf stat -I`` process.

    ``MemBandwidthSampler`` starts a ``perf stat`` per interval; on some hosts perf
    needs ~0.9 s just to start, which stretches a 1 s sampling interval to ~2 s.
    Here a single ``perf stat -a -I <ms>`` keeps counting and prints one line per
    event every interval; a reader thread keeps the latest complete interval's
    IMC read+write traffic (MiB/s). ``sample(dt)`` sleeps ``dt`` and returns it, so
    the value lags the sampler's interval by at most one perf interval.
    """

    def __init__(self, interval_ms: int = 1000) -> None:
        import threading

        self.method = "unavailable"
        self._latest: float | None = None
        self._lock = threading.Lock()
        self._proc = None
        if not shutil.which("perf"):
            return
        self._proc = subprocess.Popen(
            ["perf", "stat", "-a", "-x", ",", "-I", str(interval_ms), "-e", _IMC_EVENTS],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        self._period = interval_ms / 1000.0
        threading.Thread(target=self._read, daemon=True).start()
        deadline = time.time() + 3 * self._period + 2.0
        while time.time() < deadline and self._latest is None and self._proc.poll() is None:
            time.sleep(0.05)
        if self._latest is not None:
            self.method = "perf_imc_continuous"
        else:
            self.close()

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        stamp, total = None, 0.0
        for line in self._proc.stderr:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                ts, val = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if stamp is not None and ts != stamp:
                with self._lock:
                    self._latest = total / self._period
                total = 0.0
            stamp = ts
            total += val

    def sample(self, dt: float) -> tuple[float, str]:
        time.sleep(dt)
        with self._lock:
            v = self._latest
        if v is None or self.method == "unavailable":
            return 0.0, "unavailable"
        return v, self.method

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self.method = "unavailable"


__all__ = ["MemBandwidthSampler", "ContinuousIMCSampler"]
