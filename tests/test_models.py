import torch

from edgefault_bench.models import (
    CompactDepthwiseCNN1D,
    StandardCNN1D,
    coral_loss,
    trainable_parameter_count,
)


def test_models_return_logits_and_embeddings() -> None:
    signals = torch.randn(3, 1, 4096)
    for model in (StandardCNN1D(), CompactDepthwiseCNN1D()):
        logits, embedding = model(signals, return_embedding=True)
        assert logits.shape == (3, 4)
        assert embedding.shape[0] == 3


def test_models_accept_mehran_triaxial_input_and_two_classes() -> None:
    signals = torch.randn(2, 3, 4096)
    for model in (
        StandardCNN1D(num_classes=2, in_channels=3),
        CompactDepthwiseCNN1D(num_classes=2, in_channels=3),
    ):
        logits = model(signals)
        assert logits.shape == (2, 2)


def test_compact_model_is_at_most_quarter_standard_parameters() -> None:
    standard = trainable_parameter_count(StandardCNN1D())
    compact = trainable_parameter_count(CompactDepthwiseCNN1D())
    assert compact <= standard * 0.25


def test_coral_loss_is_zero_for_identical_domains() -> None:
    base = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    embeddings = torch.cat((base, base), dim=0)
    domains = torch.tensor([0, 0, 0, 1, 1, 1])
    assert coral_loss(embeddings, domains).item() == 0.0


def test_coral_loss_is_differentiable_with_single_domain() -> None:
    embeddings = torch.randn(4, 8, requires_grad=True)
    loss = coral_loss(embeddings, torch.zeros(4, dtype=torch.long))
    loss.backward()
    assert embeddings.grad is not None
