"""Unit tests for the optional PyTorch action encoder."""

from __future__ import annotations

import importlib.util

import pytest

from geno_lewm.action import EditType, RelEdit
from geno_lewm.errors import InputError, RuntimeSetupError, UnsupportedEditError


def _edit(
    rel_pos: int = 4,
    edit_type: EditType = EditType.SNV,
    ref_bases: str = "A",
    alt_bases: str = "T",
) -> RelEdit:
    return RelEdit(
        rel_pos=rel_pos,
        edit_type=edit_type,
        ref_bases=ref_bases,
        alt_bases=alt_bases,
    )


def test_action_encoder_reports_missing_torch_runtime() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")
    from geno_lewm.action import ActionEncoder

    with pytest.raises(RuntimeSetupError):
        ActionEncoder()


def test_action_encoder_default_output_shape_and_parameter_budget() -> None:
    pytest.importorskip("torch")
    from geno_lewm.action import ActionEncoder

    encoder = ActionEncoder()
    output = encoder(
        [[_edit(), _edit(rel_pos=8, edit_type=EditType.INS, ref_bases="A", alt_bases="AC")]]
    )
    params = sum(param.numel() for param in encoder.parameters())

    assert output.shape == (1, 2, 512)
    assert encoder.d_action == 512
    assert 2_400_000 <= params <= 2_800_000


def test_action_encoder_ablation_switches_and_batch_padding() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.action import ActionEncoder

    encoder = ActionEncoder(d_action=32, d_pos=16, d_type=8, d_seq=32, max_window_bp=128)
    output = encoder(
        [
            [
                _edit(rel_pos=0),
                _edit(rel_pos=12, edit_type=EditType.MNV, ref_bases="AC", alt_bases="GT"),
            ],
            [_edit(rel_pos=6, edit_type=EditType.DEL, ref_bases="AC", alt_bases="A")],
        ]
    )

    assert output.shape == (2, 2, 32)
    assert encoder.d_action == 32
    torch.testing.assert_close(output[1, 1], encoder.padding_embedding)


def test_action_encoder_rejects_invalid_dimensions() -> None:
    pytest.importorskip("torch")
    from geno_lewm.action import ActionEncoder

    with pytest.raises(InputError):
        ActionEncoder(d_pos=15)
    with pytest.raises(InputError):
        ActionEncoder(d_seq=30)


def test_action_encoder_rejects_out_of_window_position() -> None:
    pytest.importorskip("torch")
    from geno_lewm.action import ActionEncoder

    encoder = ActionEncoder(d_action=32, d_pos=16, d_type=8, d_seq=32, max_window_bp=16)

    with pytest.raises(InputError):
        encoder([_edit(rel_pos=16)])


def test_action_encoder_rejects_structural_variant_lengths() -> None:
    pytest.importorskip("torch")
    from geno_lewm.action import ActionEncoder

    encoder = ActionEncoder(d_action=32, d_pos=16, d_type=8, d_seq=32)

    with pytest.raises(UnsupportedEditError):
        encoder([_edit(ref_bases="A" * 17)])
