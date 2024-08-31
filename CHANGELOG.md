# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added `ditail args` to image infotext 

### Changed
- Changed the logic for enabling/disabling ditial in `sd-webui-ditial.scripts.ditail_ext.process` to desync the UI and backend values.
- Changed `script_callbacks.remove_current_script_callbacks()` to `script_callbacks.remove_callbacks_for_function(callback_func)`

### Fixed
- Fixed runtime error `RuntimeError: Input type (float) and bias type (struct c10::Half) should be the same` when running on half precision.
- Disentangled UI values and backend values to avoid multi-user conflicts.

### Removed
- Removed placeholder image

## [0.1.0] - 2024-08-29
### Added
- Initial release of the Ditail extension for Automatic 1111 Webui, supports SD1.5 checkpoints and DDIM inversion and sampling.
- AGPL-3.0 License