import torch

from edgefault_bench.hardware import multiply_accumulate_count, serialized_state_size
from edgefault_bench.models import build_model


def test_compact_model_has_lower_parameter_storage_and_macs():
    sample = torch.zeros(1, 1, 4096)
    standard = build_model("standard_cnn_1d", num_classes=4).eval()
    compact = build_model("compact_depthwise_cnn_1d", num_classes=4).eval()

    assert sum(p.numel() for p in compact.parameters()) < sum(
        p.numel() for p in standard.parameters()
    )
    assert serialized_state_size(compact) < serialized_state_size(standard)
    assert multiply_accumulate_count(compact, sample) < multiply_accumulate_count(
        standard, sample
    )
