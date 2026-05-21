# SPDX-License-Identifier: Apache-2.0
"""End-to-end evaluation smoke tests (RFC-0015 §3.1).

Tests in this package run a 1 k-variant smoke eval (RFC-0007 §3.8) to
catch regressions in the scoring pipeline before they hit the nightly
full eval. The harness lands with the ClinVar loader (#50) and the
eval runner (#53 / #56); until those modules exist the package stays
empty.
"""
