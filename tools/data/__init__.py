# SPDX-License-Identifier: Apache-2.0
"""Upstream data-acquisition tools for the first-experiment dataset snapshot.

These live under ``tools/`` rather than the network-confined ``geno_lewm``
package (runtime contract): fetching ClinVar/gnomAD VCFs and materializing the
pinned Carbon corpus windows is operator-initiated acquisition, kept out of the
fail-closed runtime.
"""
