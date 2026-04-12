"""Core internals for ``ProjectManager``.

This package contains the decomposition of the former ``project.py`` monolith
into:
- small shared helper modules (for chunk conversion and constants), and
- domain-focused mixins that are composed back into ``ProjectManager``.

The package is intentionally implementation-oriented and mirrors existing
runtime behavior rather than defining a separate public API surface.
"""
