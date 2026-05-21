# SPDX-License-Identifier: Apache-2.0
"""Performance benchmark harness (RFC-0016 §3.4).

This package is **not** part of the public ``geno_lewm`` surface. Each
entry point under ``bench/`` is a standalone script that times a hot
path and writes a structured JSON result to
``bench/results/<machine>/<benchmark>.json``.

The shared library lives in :mod:`bench._harness`; per-target scripts
(``bench/inference.py``, ``bench/training.py``, ``bench/planning.py``,
``bench/profile.py``) own their own workload.

The harness is intentionally stdlib-only so it can run on a fresh
CPU-only laptop without the optional ML extras installed.
"""
