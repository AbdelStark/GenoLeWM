# SPDX-License-Identifier: Apache-2.0
"""ML-specific smoke tests (testing contract).

Tests in this package are hosted ML smoke gates that catch collapse,
instability, fixture-training determinism, and optional-runtime predictor
failures with a shape distinct from unit or integration failures.

The suite stays public and fixture-backed: it does not require private
model or data files, and optional torch checks skip explicitly when the
runtime is unavailable.
"""
