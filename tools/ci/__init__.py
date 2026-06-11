# SPDX-License-Identifier: Apache-2.0
"""CI helpers (testing contract / performance budget).

Scripts in this package are invoked from the CI workflow; they are not
part of the public ``geno_lewm`` surface. They include the changed-files
coverage gate, benchmark regression checks, and the generated-fixture
eval smoke gate.
"""
