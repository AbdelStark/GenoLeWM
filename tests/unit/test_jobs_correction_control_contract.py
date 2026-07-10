"""Static contract tests for the clean-machine correction-control job."""

from __future__ import annotations

from pathlib import Path

PROOF_RUN = Path("tools/jobs/proof_run.sh")


def test_proof_job_uses_only_correction_control_config_and_immutable_sources() -> None:
    script = PROOF_RUN.read_text(encoding="utf-8")

    assert "configs/correction_control/train-carbon-500m-snv-l2-smoke-v1.yaml" in script
    assert "configs/correction_control/dataset-snapshot-snv-l2-smoke-v1.json" in script
    assert "configs/first_experiment" not in script
    assert "configs/serious_completion" not in script
    assert "archive_2.0/2026/clinvar_20260415.vcf.gz" in script
    assert "vcf_GRCh38/clinvar.vcf.gz" not in script
    assert "?generation=1713312296186865" in script
    assert 'CLINVAR_MD5="e63b5c3a046010c098cc70e81bebaa8d"' in script
    assert 'GNOMAD_MD5="dcf191563e69054a71bd4dc77862799a"' in script
    assert "35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db" in script
    assert "FASTA22_URL" not in script


def test_proof_job_validates_before_writes_and_uploads_only_after_postflight() -> None:
    script = PROOF_RUN.read_text(encoding="utf-8")

    preflight = script.index("python -m tools.research.correction_control_preflight")
    no_clobber = script.index("REMOTE_AUDIT_URL=")
    first_work_write = script.index("mkdir -p")
    state_audit = script.index("python -m tools.research.state_contract_audit")
    training = script.index("geno-lewm-train --carbon-train")
    postflight = script.index("python -m tools.research.correction_control_postflight")
    export = script.index("geno-lewm-export")
    model_upload = script.index(
        'hf upload "$UPLOAD_REPO" "$WORK/model" "$RUN_NAME/model" --repo-type model'
    )
    dataset_upload = script.index(
        'hf upload "$UPLOAD_REPO" "$WORK/dataset" "$RUN_NAME/dataset" --repo-type model'
    )
    completed_run_upload = script.index(
        'hf upload "$UPLOAD_REPO" "$WORK/run" "$RUN_NAME/run" --repo-type model'
    )

    assert (
        preflight
        < no_clobber
        < first_work_write
        < state_audit
        < training
        < postflight
        < export
        < model_upload
        < dataset_upload
        < completed_run_upload
    )
    assert '--window-bp "$WINDOW_BP"' in script
    assert "--no-trust-remote-code" in script
    assert 'STEPS="${STEPS:-50}"' in script
    assert 'RUN_ATTEMPT="${RUN_ATTEMPT:-1}"' in script
    assert 'MAX_WINDOWS="${MAX_WINDOWS:-512}"' in script
    assert 'CLINVAR_LINES="${CLINVAR_LINES:-60000}"' in script
    assert 'TUPLE_THROUGHPUT_SAMPLES="${TUPLE_THROUGHPUT_SAMPLES:-400}"' in script
    assert 'MIN_CUDA_VRAM_GB="120"' in script
    assert 'if "H200" not in properties.name:' in script
    assert 'cp "$WORK/dataset/dataset_input_check_report.json"' in script
    assert 'if [ "$job_contract_rc" -ne 0 ]; then' in script
    assert "printf '%s\\n' \"$JOB_CONTRACT_REPORT\" >&2" in script

    postflight_invocation = script[postflight:completed_run_upload]
    for evidence_flag in (
        "--job-contract-preflight-json",
        "--source-identity-report-json",
        "--dataset-manifest-json",
        "--dataset-snapshot-report-json",
        "--training-preflight-report-json",
        "--tuple-throughput-report-json",
    ):
        assert evidence_flag in postflight_invocation

    training_invocation = script[training:postflight]
    for override_flag in ("--set", "--seed", "--run-id", "--deterministic"):
        assert override_flag not in training_invocation
