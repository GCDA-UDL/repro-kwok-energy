"""RAPL energy measurement (Phase 12, Component B; vendor-agnostic-measurement).

Reads CPU package/DRAM energy counters, vendor-agnostically:
  * powercap RAPL  ``/sys/class/powercap/{intel-rapl,amd-rapl}:*/energy_uj``
  * amd_energy     ``/sys/class/hwmon/*/energy*_input`` (AMD fallback, package only)

Computes wrap-corrected deltas using each domain's ``max_energy_range_uj``.
Package is matched by zone-name ``package``/``package-N`` (real Intel/AMD name is
``package-0``); DRAM by ``dram``.

Guarantees: I2 (joules canonical); graceful skip on missing perms/devices;
no monotonic-power assumption under throttling.
"""

from __future__ import annotations

from pathlib import Path

RAPL_ROOT = Path("/sys/class/powercap")
HWMON_ROOT = Path("/sys/class/hwmon")

# powercap directory prefixes that expose RAPL zones (Intel + AMD).
_RAPL_ZONE_PREFIXES = ("intel-rapl", "amd-rapl")

# Max energy range (μJ) for common RAPL domains. Used for wrap correction only as a
# fallback when a domain does not expose ``max_energy_range_uj``.
DEFAULT_PKG_MAX_UJ = 263637745
DEFAULT_DRAM_MAX_UJ = 16302046487


def _read_text(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def read_energy_uj(domain_path: str | Path) -> int:
    """Read the current energy counter (μJ) from a RAPL domain path."""
    p = Path(domain_path) / "energy_uj"
    return int(p.read_text().strip())


def read_max_energy_range_uj(domain_path: str | Path) -> int | None:
    """Read the wrap modulus (μJ) for a RAPL domain from sysfs.

    The real wrap point is per-domain and SKU-specific; use it instead of a
    hardcoded default so wrap correction is correct on any host.
    """
    p = Path(domain_path) / "max_energy_range_uj"
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def delta(prev: int, cur: int, max_range: int) -> int:
    """Compute the wrap-corrected energy difference (μJ).

    If ``cur >= prev``: simple difference. If ``cur < prev``: assume the counter
    wrapped around ``max_range``: ``cur + max_range - prev``.
    """
    if cur >= prev:
        return cur - prev
    return cur + max_range - prev


def uj_to_j(energy_uj: float | int) -> float:
    """Convert microjoules to joules (exact)."""
    return float(energy_uj) / 1e6


def _is_rapl_zone_dir(name: str) -> bool:
    return name.startswith(_RAPL_ZONE_PREFIXES)


def discover_domains(root: Path | None = None) -> list[dict[str, str | Path]]:
    """Discover all RAPL domains under the powercap root (Intel + AMD).

    Returns ``{name, path}`` per zone. Top-level zones are ``intel-rapl:0`` /
    ``amd-rapl:0`` (name ``package-0``); sub-zones ``:0:0``, ``:0:1`` (``dram``,
    ``core``, ``uncore``).
    """
    powercap = root or RAPL_ROOT
    domains: list[dict[str, str | Path]] = []
    if not powercap.is_dir():
        return domains
    for d in sorted(powercap.iterdir()):
        if not _is_rapl_zone_dir(d.name):
            continue
        name = _read_text(d / "name")
        if name is None:
            continue
        domains.append({"name": name, "path": d})
        for sub in sorted(d.iterdir()):
            if not _is_rapl_zone_dir(sub.name):
                continue
            sub_name = _read_text(sub / "name")
            if sub_name is not None:
                domains.append({"name": sub_name, "path": sub})
    return domains


def find_domain(name: str, root: Path | None = None) -> Path | None:
    """Find a RAPL domain by name (e.g. 'package', 'dram', 'core').

    Matches the exact zone name or a per-index variant: ``find_domain('package')``
    matches the real Intel/AMD zone name ``package-0`` (and ``package-1`` on the
    next socket). Exact names like ``dram`` still match exactly.
    """
    for d in discover_domains(root):
        dn = str(d["name"])
        if dn == name or dn.startswith(name + "-"):
            return d["path"]  # type: ignore[return-value]
    return None


def discover_amd_energy_package(hwmon_root: Path | None = None) -> Path | None:
    """Find an ``amd_energy`` hwmon package-energy input file (μJ), or None.

    AMD hosts without a powercap RAPL zone expose per-socket/core energy via the
    ``amd_energy`` hwmon driver. We use the socket-labelled channel as the package
    energy (AMD RAPL has no DRAM domain). Returns the ``energyN_input`` Path.
    """
    root = hwmon_root or HWMON_ROOT
    if not root.is_dir():
        return None
    for h in sorted(root.glob("hwmon*")):
        if _read_text(h / "name") != "amd_energy":
            continue
        inputs = sorted(h.glob("energy*_input"))
        for ein in inputs:
            idx = ein.name[len("energy"):-len("_input")]
            label = (_read_text(h / f"energy{idx}_label") or "").lower()
            if "socket" in label:
                return ein
        if inputs:
            return inputs[0]
    return None


class RAPLReader:
    """Reads CPU package/DRAM energy (Intel/AMD), graceful skip on missing devices."""

    def __init__(self, root: Path | None = None, hwmon_root: Path | None = None) -> None:
        self._root = root or RAPL_ROOT
        self._hwmon_root = hwmon_root or HWMON_ROOT
        self._pkg_energy: Path | None = None
        self._dram_energy: Path | None = None
        self._pkg_max_uj: int | None = None
        self._dram_max_uj: int | None = None
        self._source = "none"
        self._discover()

    @staticmethod
    def _is_package(name: str) -> bool:
        """Intel/AMD name the package zone ``package-0`` (per socket), not ``package``."""
        return name == "package" or name.startswith("package-")

    def _discover(self) -> None:
        # 1) powercap RAPL (Intel + AMD)
        for d in discover_domains(self._root):
            name = str(d["name"])
            path = Path(str(d["path"]))
            if self._is_package(name) and self._pkg_energy is None:
                self._pkg_energy = path / "energy_uj"
                self._pkg_max_uj = read_max_energy_range_uj(path)
                self._source = "powercap"
            elif name == "dram" and self._dram_energy is None:
                self._dram_energy = path / "energy_uj"
                self._dram_max_uj = read_max_energy_range_uj(path)
        # 2) amd_energy hwmon fallback (package only; AMD RAPL has no DRAM domain)
        if self._pkg_energy is None:
            amd = discover_amd_energy_package(self._hwmon_root)
            if amd is not None:
                self._pkg_energy = amd
                self._source = "amd_energy"

    @property
    def available(self) -> bool:
        return self._pkg_energy is not None

    @property
    def dram_available(self) -> bool:
        return self._dram_energy is not None

    @property
    def source(self) -> str:
        """How package energy is read: ``powercap`` | ``amd_energy`` | ``none``."""
        return self._source

    @property
    def pkg_max_uj(self) -> int | None:
        """Per-domain wrap modulus for the package counter (None if absent)."""
        return self._pkg_max_uj

    @property
    def dram_max_uj(self) -> int | None:
        """Per-domain wrap modulus for the DRAM counter (None if absent)."""
        return self._dram_max_uj

    def _read_uj(self, energy_file: Path | None) -> int | None:
        if energy_file is None:
            return None
        try:
            return int(energy_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def read_package_uj(self) -> int | None:
        return self._read_uj(self._pkg_energy)

    def read_dram_uj(self) -> int | None:
        return self._read_uj(self._dram_energy)


class MultiSocketRAPL:
    """Every RAPL ``package-N`` and ``dram`` zone of the host (schema v1.1).

    :class:`RAPLReader` keeps the first package and the first DRAM zone, i.e. socket
    0 only; on a multi-socket server that misses the other sockets. This reader
    keeps all of them (deduplicated by real path: sysfs lists sub-zones such as
    ``intel-rapl:0:0`` both nested and at the top level) and sums per-zone,
    wrap-corrected deltas.
    """

    def __init__(self, root: Path | None = None) -> None:
        zones: dict[Path, str] = {}
        for d in discover_domains(root):
            zones.setdefault(Path(str(d["path"])).resolve(), str(d["name"]))
        ordered = sorted(zones.items(), key=lambda kv: str(kv[0]))
        pkg = [(p, read_max_energy_range_uj(p)) for p, n in ordered if RAPLReader._is_package(n)]
        dram = [(p, read_max_energy_range_uj(p)) for p, n in ordered if n == "dram"]
        # The wrap modulus is SKU-specific (~2.6e11 uJ on these Xeons); the module
        # defaults are far smaller, so a wrap computed with them would be garbage.
        self.wrap_known = all(m is not None for _, m in pkg + dram)
        self.package_zones = [(p / "energy_uj", m or DEFAULT_PKG_MAX_UJ) for p, m in pkg]
        self.dram_zones = [(p / "energy_uj", m or DEFAULT_DRAM_MAX_UJ) for p, m in dram]

    @property
    def sockets(self) -> int:
        return len(self.package_zones)

    @staticmethod
    def _read(zones: list[tuple[Path, int]]) -> list[int | None]:
        out: list[int | None] = []
        for f, _ in zones:
            try:
                out.append(int(f.read_text().strip()))
            except (OSError, ValueError):
                out.append(None)
        return out

    def read(self) -> tuple[list[int | None], list[int | None]]:
        """Raw counters (uJ) of every package zone and every DRAM zone."""
        return self._read(self.package_zones), self._read(self.dram_zones)

    @staticmethod
    def joules(prev: list[int | None], cur: list[int | None],
               zones: list[tuple[Path, int]]) -> float | None:
        """Summed wrap-corrected energy (J) between two reads; None if any zone failed."""
        if not zones or any(v is None for v in prev) or any(v is None for v in cur):
            return None
        total = 0
        for a, b, (_, m) in zip(prev, cur, zones, strict=True):
            if b < a and m < a:  # wrap with a modulus below the counter: impossible
                return None
            total += delta(a, b, m)  # type: ignore[arg-type]
        return total / 1e6


__all__ = [
    "MultiSocketRAPL",
    "RAPLReader",
    "read_energy_uj",
    "read_max_energy_range_uj",
    "delta",
    "uj_to_j",
    "discover_domains",
    "find_domain",
    "discover_amd_energy_package",
    "DEFAULT_PKG_MAX_UJ",
    "DEFAULT_DRAM_MAX_UJ",
]
