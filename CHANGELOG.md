# Changelog

All notable changes to this project are documented here. Newest releases first.

## [v0.3.1] - 2026-06-18

> Package version `0.3.1`. Still not production-ready yet.

### Changed
- `ImageEncodingMap` now encodes every image up front instead of lazily on first access, which drops the in-memory buffer machinery and simplifies the class.
- `ImageEncodingMap.save_to_disk()` / `load_from_disk()` now use the safetensors format instead of HDF5. Files default to the `.safetensors` extension.
- `skip_errors` moved from `save_to_disk()` to the `ImageEncodingMap` constructor, since encoding now happens at construction time.

### Removed
- Dropped `h5py` as a dependency; added `safetensors`.
- Removed `ImageEncodingMap.clear_buffer()` (there's no buffer to clear anymore).

### Breaking
- ⚠️ Unreadable or missing images now raise (`FileNotFoundError` / `ValueError`) when the map is built, not on first access. Use `skip_errors=True` to drop them with a warning instead.
- ⚠️ Encoding maps saved with `0.3.0` (HDF5) can't be loaded by `0.3.1`; re-save them as safetensors.

## [v0.3.0] - 2026-06-17

> Package version `0.3.0`. Still not production-ready yet.

### Added
- New encoding map feature for the encoders (#36).
- PyPI publishing step in the CI workflow, so releases ship automatically (#37).

### Changed
- Batches now fetch dynamically from PyPI instead of being hardcoded (#38).
- Moved the notebooks into a tidier layout (#35).

### Fixed
- Dropped the deprecated `project.license` TOML table in `pyproject.toml` (#39).

## [v0.2.0] - 2026-06-16

> ⚠️ Heads up: still not production-ready. Treat this as an early preview.

### Added
- Clustering models with a fresh public API (#19), plus docs to match (#21).
- Public types `ImageInput` and `MatLike`, and you can now pass torch images straight in (#23).
- Unit tests across the board (#24), including behavioral tests that check VLAD and Fisher Vector encoders return the same vector before and after serialization (#32).
- Early sketch of a Siamese neural net (#26).

### Changed
- Migrated tooling to `uv` (#5).
- Added ruff, pre-commit hooks, and a CI pipeline that runs on every PR (#9).
- Integrated mypy and cleaned up the type errors (#8, #13).
- Now compatible with Python 3.10 through 3.12 (#14, #15).
- Refreshed the outdated getting-started notebook (#30).

### Fixed
- `VLADEncoder` now raises if no descriptor is extracted, instead of failing silently (#11).
- Use `flatten()` instead of `squeeze()` for setid arrays, so single-element arrays behave (#29).

## [v0.1.3-alpha] - 2025-01-24

- Initial alpha release.
