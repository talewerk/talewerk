"""Talewerk: turn a simple idea into a finished video."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("talewerk")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
