# Carbon runtime-hash CPU probe

This probe is a cheap, nonpublishing check of the exact Carbon model volume before another H200
cache attempt. It mounts `HuggingFaceBio/Carbon-500M` at revision
`5d31d59b3c845b288a13aedb1358934196852eec` read-only at `/carbon`, clones one exact public
GenoLeWM commit, and evaluates the real
`geno_lewm.encoder._identity.encoder_runtime_hash` function over the complete mount. The streamed
digest includes the model weights as well as runtime-critical Carbon and GenoLeWM implementation
files.

The probe does not upload an artifact and receives no in-job secret. It proves only that the
mounted bytes produce the explicitly expected runtime hash under the selected GenoLeWM source
commit. It does not prove model quality, cache correctness, H200 behavior, corpus coverage, or
completion within 24 hours.

## Review the launch contract

Run from the clean canonical checkout that will be submitted. The launcher intentionally has no
built-in expected runtime hash. Read the value from the versioned identity file in that exact
checkout and pass the full digest explicitly:

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)"
EXPECTED_RUNTIME_HASH="$(
  uv run --no-project python -c \
    'import json; print(json.load(open("configs/data_v03/carbon-500m-l2-runtime-identity.json"))["runtime_hash"])'
)"

uv run --script tools/research/v03_carbon_runtime_hash_probe_launch.py \
  --source-commit "$SOURCE_COMMIT" \
  --expected-runtime-hash "$EXPECTED_RUNTIME_HASH" \
  --run-attempt 1 \
  --dry-run
```

The credential-free dry run must show all of these bindings:

- the digest-pinned `uv` image and `cpu-basic` flavor;
- the submitted 30-minute timeout;
- the exact public source commit and explicit expected hash;
- the exact Carbon repository, revision, `/carbon` mount, `read_only: true`, and `path: null`;
- `purpose=geno-lewm-v03-carbon-runtime-hash-probe`; and
- no secret names or values.

Hugging Face JobInfo does not echo the submitted timeout. The launcher therefore records the
30-minute value in the dry-run and in-job terminal bindings without claiming a server-side timeout
round trip.

## Submit and verify

Submission requires `HF_TOKEN` only on the host so the API can create the Job. The token is not
passed into the container:

```bash
uv run --script tools/research/v03_carbon_runtime_hash_probe_launch.py \
  --source-commit "$SOURCE_COMMIT" \
  --expected-runtime-hash "$EXPECTED_RUNTIME_HASH" \
  --run-attempt 1
```

After submission, the launcher accepts the JobInfo only if its image, command, empty argument
vector, environment, `cpu-basic` flavor, purpose label, empty secret-name set, namespace, absent
Space image, and complete revision-bearing model-volume tuple all match. Any visible drift triggers
a cancellation attempt and a failed launch.

The in-job command creates a fresh workspace, clones the public canonical repository, checks out
the exact commit in detached mode, requires a clean tree, and installs only the frozen base project
dependencies with `uv sync --frozen --no-dev`. It then runs the repository's probe module against
the real `/carbon` mount. Success is exactly one terminal line beginning with:

```text
GENO_LEWM_V03_CARBON_RUNTIME_HASH_PROBE_OK
```

That line binds the source commit, Carbon repository/revision/mount, observed runtime hash,
container image, flavor, namespace, purpose label, submitted timeout, and run attempt. A missing
runtime file or any digest mismatch exits nonzero before that marker and before any publication.
