import torch

from edgefault_bench.robustness import add_awgn, stratified_fraction_indices


def test_stratified_fraction_is_deterministic_and_preserves_strata():
    indices = torch.arange(16)
    targets = torch.tensor([0] * 8 + [1] * 8)
    domains = torch.tensor(([0] * 4 + [1] * 4) * 2)

    first = stratified_fraction_indices(
        indices, targets, domains, fraction=0.5, seed=17
    )
    second = stratified_fraction_indices(
        indices, targets, domains, fraction=0.5, seed=17
    )

    assert torch.equal(first, second)
    assert len(first) == 8
    assert set(zip(targets[first].tolist(), domains[first].tolist(), strict=True)) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_targeted_fraction_keeps_non_target_class():
    indices = torch.arange(12)
    targets = torch.tensor([0] * 6 + [1] * 6)
    domains = torch.tensor([0, 0, 0, 1, 1, 1] * 2)

    selected = stratified_fraction_indices(
        indices, targets, domains, fraction=0.5, seed=29, target_class=1
    )

    assert set(range(6)).issubset(selected.tolist())
    assert int((targets[selected] == 1).sum()) == 2


def test_awgn_is_deterministic_and_restandardized():
    signals = torch.linspace(-1.0, 1.0, 128).reshape(2, 1, 64)
    first = add_awgn(signals, snr_db=10.0, seed=43)
    second = add_awgn(signals, snr_db=10.0, seed=43)

    assert torch.equal(first, second)
    assert torch.allclose(first.mean(dim=-1), torch.zeros(2, 1), atol=1e-6)
    assert torch.allclose(
        first.std(dim=-1, correction=0), torch.ones(2, 1), atol=1e-6
    )
