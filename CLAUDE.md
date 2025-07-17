# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stereo vision project built with Python, using PyTorch and NumPy for computer vision tasks. The project is in early development stage with a modular architecture planned for stereo vision processing.

## Development Environment

- **Python Version**: >=3.10
- **Package Manager**: uv (configured with uv.lock)
- **Dependencies**: numpy>=2.2.6, torch>=2.7.1, torchvision>=0.22.1

## Common Commands

### Running the Application

```bash
uv run hello.py
```

### Package Management

```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package_name>

# Update dependencies
uv lock --upgrade
```

## Project Structure

The project follows a modular architecture with the following planned components:

- `stereo_vision/` - Main package directory
  - `feature_extractor/` - Feature extraction algorithms
  - `feature_matcher/` - Feature matching algorithms
  - `two_view_geometory/` - Two-view geometry computations (note: directory name has typo)
- `scripts/` - Utility scripts
- `hello.py` - Main entry point (currently a placeholder)

## Architecture Notes

- The project is structured as a Python package with separate modules for different stereo vision processing stages
- Currently only contains a basic "Hello World" entry point
- Directory structure suggests a pipeline approach: feature extraction → feature matching → geometric processing
- Uses modern Python packaging with pyproject.toml configuration
