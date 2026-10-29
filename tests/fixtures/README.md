# tests/fixtures/README.md
#
# Adversarial and reference data fixtures for kwok_energy tests.
#
# Each file here is loaded via the `fixture_text` / `fixture_path` conftest
# fixtures (see tests/conftest.py). Fixtures encode known traps as data:
# normal, missing_dram, loop_boundary, irregular_dt, irregular_scrape,
# restart_counter, speed_factor, near_zero_idle, component_cancellation,
# io_enabled, target_leakage (see PLAN.md Phase 6).
#
# `data/` (real measurement output) is gitignored and NOT committed; only these
# synthetic fixtures live in version control.
