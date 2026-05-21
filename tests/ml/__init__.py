# SPDX-License-Identifier: Apache-2.0
"""ML-specific smoke tests (RFC-0015 §3.1).

Tests in this package catch collapse / instability failures that have a
shape distinct from either unit or integration failures. The full
suite lands with the trainer (#44 / #89): identity-at-init,
loss-decreases, no-NaN-inf, receipt-determinism. Until the encoder
(#32) and predictor (#41) modules exist, this package is intentionally
empty — pytest collects zero tests from it.
"""
