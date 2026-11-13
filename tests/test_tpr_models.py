"""Check TPR composition, sentence-only decoding, and weighted training losses."""

import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from ac_tpr.tpr.decomposition import TensorProductModel, tpr_loss
from ac_tpr.tpr.training import TrainingConfig, fit_model
from ac_tpr.tpr.unpacking import (
    BOS_ID, EOS_ID, PAD_ID, SentenceUnpacker, UnpackingConfig, unpacking_loss,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_unpacker():
    return SentenceUnpacker(2, 6, UnpackingConfig(
        hidden_size=16, n_layers=1, n_heads=2, ffn_size=32, dropout=0.0, max_length=6,
    ))


def test_tpr_binding_order_bias_cancellation_and_swap_inversion():
    torch.manual_seed(7)
    model = TensorProductModel(6, 7, filler_dim=3, role_dim=2)
    fillers = torch.tensor([[0, 1, 4, 5], [2, 3, 4, 5]])
    roles = torch.arange(4).expand(2, -1)
    permutation = [3, 1, 0, 2]
    torch.testing.assert_close(model(fillers, roles), model(fillers[:, permutation], roles[:, permutation]))

    swapped_roles = roles[:, [1, 0, 2, 3]]
    delta = model(fillers, swapped_roles) - model(fillers, roles)
    assert delta.norm() > 0
    old_bias = model.output.bias.detach().clone()
    with torch.no_grad():
        model.output.bias.add_(13.0)
    torch.testing.assert_close(model(fillers, swapped_roles) - model(fillers, roles), delta, atol=2e-6, rtol=1e-5)
    with torch.no_grad():
        model.output.bias.copy_(old_bias)
    partial_delta = model(fillers[:, :2], swapped_roles[:, :2]) - model(fillers[:, :2], roles[:, :2])
    torch.testing.assert_close(partial_delta, delta)
    activation, sigma = torch.randn(2, 7), torch.rand(7) + 0.1
    edited = activation + sigma * delta
    reversed_edit = edited + sigma * (model(fillers, roles) - model(fillers, swapped_roles))
    torch.testing.assert_close(reversed_edit, activation)


def test_unpacker_causal_mask_and_memory_conditioning():
    torch.manual_seed(1)
    model = small_unpacker().eval()
    z = torch.eye(2)
    original = torch.tensor([[BOS_ID, 3, 4, 5], [BOS_ID, 3, 4, 5]])
    changed = original.clone()
    changed[:, 2:] = torch.tensor([5, 3])
    with torch.no_grad():
        logits = model(z, original)
        altered = model(z, changed)
    torch.testing.assert_close(logits[:, :2], altered[:, :2])
    assert not torch.allclose(logits[0, 0], logits[1, 0])


def test_unpacker_learns_free_generation_and_state_dict_roundtrip():
    torch.manual_seed(3)
    model = small_unpacker()
    z = torch.eye(2)
    decoder_inputs = torch.tensor([[BOS_ID, 3, 4], [BOS_ID, 4, 3]])
    targets = torch.tensor([[3, 4, EOS_ID], [4, 3, EOS_ID]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(100):
        loss, count = unpacking_loss(model, (z, decoder_inputs, targets))
        optimizer.zero_grad(set_to_none=True)
        (loss / count).backward()
        optimizer.step()
    torch.testing.assert_close(model.generate(z), targets)
    assert model.training
    restored = small_unpacker().eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.generate(z.flip(0)), targets.flip(0))


def test_generation_stops_per_row_and_pads_finished_rows(monkeypatch):
    model = small_unpacker()

    def logits(z, inputs):
        output = torch.zeros(2, inputs.shape[1], 6)
        output[0, -1, EOS_ID] = 1
        output[1, -1, EOS_ID if inputs.shape[1] == 3 else 3] = 1
        return output

    monkeypatch.setattr(model, "forward", logits)
    torch.testing.assert_close(model.generate(torch.eye(2)), torch.tensor([
        [EOS_ID, PAD_ID, PAD_ID], [3, 3, EOS_ID],
    ]))
    with pytest.raises(ValueError, match="Generation length"):
        model.generate(torch.eye(2), max_length=7)


def test_losses_use_nonpadding_tokens_and_activation_coordinates():
    torch.manual_seed(5)
    unpacker = small_unpacker().eval()
    z = torch.eye(2)
    inputs = torch.tensor([[BOS_ID, 3, PAD_ID], [BOS_ID, 4, 5]])
    targets = torch.tensor([[3, EOS_ID, PAD_ID], [4, 5, EOS_ID]])
    total, count = unpacking_loss(unpacker, (z, inputs, targets))
    first, n_first = unpacking_loss(unpacker, (z[:1], inputs[:1, :2], targets[:1, :2]))
    second, n_second = unpacking_loss(unpacker, (z[1:], inputs[1:], targets[1:]))
    assert count == n_first + n_second == 5
    torch.testing.assert_close(total, first + second)
    model = TensorProductModel(6, 7, filler_dim=3, role_dim=2)
    fillers = torch.tensor([[0, 1, 4, 5], [2, 3, 4, 5]])
    roles = torch.arange(4).expand(2, -1)
    target = model(fillers, roles).detach() + 2
    loss, count = tpr_loss(model, (fillers, roles, target))
    assert count == 14
    torch.testing.assert_close(loss, torch.tensor(56.0))


def test_fit_tpr_recovers_unseen_role_combinations_and_saves_best(tmp_path):
    torch.manual_seed(4)
    pairs = [(agent, patient) for agent in range(4) for patient in range(4) if agent != patient]
    fillers = torch.tensor([[agent, patient, 4, 5] for agent, patient in pairs])
    roles = torch.arange(4).expand(len(pairs), -1)
    targets = torch.tensor([[agent / 3, patient / 3, (agent - patient) / 3] for agent, patient in pairs])
    validation = torch.tensor([pair in {(0, 3), (3, 0)} for pair in pairs])
    training_set = TensorDataset(fillers[~validation], roles[~validation], targets[~validation])
    validation_set = TensorDataset(fillers[validation], roles[validation], targets[validation])
    model = TensorProductModel(6, 3, filler_dim=4, role_dim=4)
    history = fit_model(model, training_set, validation_set, tpr_loss, tmp_path, TrainingConfig(
        batch_size=10, learning_rate=0.03, epochs=250, eval_every=10, patience=30, max_lr_reductions=1,
    ), "cpu")
    assert history["best_validation_loss"] < 1e-4
    assert not model.training
    torch.testing.assert_close(model(fillers, roles), targets, atol=0.025, rtol=0)
    saved = torch.load(tmp_path / "best.pt", weights_only=True)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, saved[name])
    assert json.loads((tmp_path / "history.json").read_text()) == history


def test_fit_weights_partial_validation_batch_and_stops_at_plateau(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    train_set = TensorDataset(torch.zeros(2, 1))
    valid_set = TensorDataset(torch.tensor([[1.0], [1.0], [4.0]]))

    def loss_fn(model, batch):
        values, = batch
        return model.weight.sum() * 0 + values.sum(), values.numel()

    history = fit_model(model, train_set, valid_set, loss_fn, tmp_path, TrainingConfig(
        batch_size=2, epochs=10, eval_every=1, patience=1, max_lr_reductions=1,
    ), "cpu")
    assert history["best_validation_loss"] == 2.0
    assert history["steps"] == 3
    assert history["learning_rate_reductions"] == 1
    assert [entry["learning_rate"] for entry in history["evaluations"]] == [1e-3, 1e-3, 5e-4]
