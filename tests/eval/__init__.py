# SPDX-License-Identifier: Apache-2.0
"""End-to-end evaluation smoke tests (testing contract).

Tests in this package exercise the hosted fixture-backed eval smoke gate.
They generate public score/label artifacts, run the ``geno-lewm-eval``
and ``geno-lewm-eval-all`` CLI boundaries, enforce metric thresholds,
and record why real checkpoint/dataset evaluation remains outside the
hosted smoke path.
"""
