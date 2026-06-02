# SPDX-License-Identifier: Apache-2.0
"""Cross-module integration tests (RFC-0015 §3.1).

Tests in this package exercise multiple subsystems together (e.g., the
verify CLI driving the provenance + action layers end-to-end). They
are slower than unit tests but still complete in under 30 s on a
laptop. Layer-spanning tests that need real ML modules land with the
trainer (#44); for now the package mostly empty.
"""
