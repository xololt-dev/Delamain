import pytest
import torch

from alternative_models.Delamain import Delamain
from alternative_models.Delamain_2 import Delamain_2
from alternative_models.Delamain_2_1 import Delamain_2_1
from alternative_models.Delamain_2_5 import Delamain_2_5, Delamain_2_5_PPO
from tests.helpers import DEVICES


# --- Models to parametrize ---
# Each entry: (model_class_name, input_fixture_name)
MODELS_DQN = [
    ("Delamain", "sample_state_raw"),
    ("Delamain_2", "sample_state_d2"),
    ("Delamain_2_1", "sample_state_d21"),
    ("Delamain_2_5", "sample_state"),
]

MODEL_CLASSES = {
    "Delamain": Delamain,
    "Delamain_2": Delamain_2,
    "Delamain_2_1": Delamain_2_1,
    "Delamain_2_5": Delamain_2_5,
}


@pytest.fixture
def model_class(request):
    return MODEL_CLASSES[request.param]


@pytest.fixture
def dqn_input(
    request, sample_state_raw, sample_state_d2, sample_state_d21, sample_state
):
    fixture_map = {
        "sample_state_raw": sample_state_raw,
        "sample_state_d2": sample_state_d2,
        "sample_state_d21": sample_state_d21,
        "sample_state": sample_state,
    }
    return fixture_map[request.param]


class TestDelamainDQNModels:
    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    @pytest.mark.parametrize(
        "model_class, dqn_input",
        [(name, fixture) for name, fixture in MODELS_DQN],
        indirect=["model_class", "dqn_input"],
        ids=[m[0] for m in MODELS_DQN],
    )
    def test_forward_shape(self, model_class, dqn_input, device):
        model = model_class().to(device)
        model.eval()
        with torch.no_grad():
            output = model(dqn_input.to(device))
        assert output.shape == (2, 5)

    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    @pytest.mark.parametrize(
        "model_class, dqn_input",
        [(name, fixture) for name, fixture in MODELS_DQN],
        indirect=["model_class", "dqn_input"],
        ids=[m[0] for m in MODELS_DQN],
    )
    def test_output_is_float32(self, model_class, dqn_input, device):
        model = model_class().to(device)
        model.eval()
        with torch.no_grad():
            output = model(dqn_input.to(device))
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    @pytest.mark.parametrize(
        "model_class, dqn_input",
        [(name, fixture) for name, fixture in MODELS_DQN],
        indirect=["model_class", "dqn_input"],
        ids=[m[0] for m in MODELS_DQN],
    )
    def test_gradient_flow(self, model_class, dqn_input, device):
        model = model_class().to(device)
        model.train()
        input_tensor = dqn_input.float().to(device)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    @pytest.mark.parametrize(
        "model_class, dqn_input",
        [(name, fixture) for name, fixture in MODELS_DQN],
        indirect=["model_class", "dqn_input"],
        ids=[m[0] for m in MODELS_DQN],
    )
    def test_get_params_positive(self, model_class, dqn_input):
        model = model_class()
        params = model.get_params()
        assert isinstance(params, int)
        assert params > 0

    def test_delamain_prev_frame_not_needed(self):
        assert Delamain().is_prev_frame_needed() is False

    def test_delamain_2_prev_frame_needed(self):
        assert Delamain_2().is_prev_frame_needed() is True

    def test_delamain_2_1_prev_frame_needed(self):
        assert Delamain_2_1().is_prev_frame_needed() is True

    def test_delamain_2_5_prev_frame_needed(self):
        assert Delamain_2_5().is_prev_frame_needed() is True

    def test_delamain_2_5_prev_frames_needed(self):
        assert Delamain_2_5().prev_frames_needed() == 4


class TestDelamainPPO:
    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    def test_forward_returns_tuple(self, device):
        model = Delamain_2_5_PPO().to(device)
        model.eval()
        state = torch.randint(0, 256, (2, 96, 96, 12), dtype=torch.uint8, device=device)
        with torch.no_grad():
            actor_out, critic_out = model(state)
        assert actor_out.shape == (2, 5)
        assert critic_out.shape == (2, 1)

    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    def test_forward_dtypes(self, device):
        model = Delamain_2_5_PPO().to(device)
        model.eval()
        state = torch.randint(0, 256, (2, 96, 96, 12), dtype=torch.uint8, device=device)
        with torch.no_grad():
            actor_out, critic_out = model(state)
        assert actor_out.dtype == torch.float32
        assert critic_out.dtype == torch.float32

    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    def test_gradient_flow(self, device):
        model = Delamain_2_5_PPO().to(device)
        model.train()
        state = torch.randint(
            0, 256, (2, 96, 96, 12), dtype=torch.float32, device=device
        )
        actor_out, critic_out = model(state)
        loss = actor_out.sum() + critic_out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_get_params_positive(self):
        params = Delamain_2_5_PPO().get_params()
        assert isinstance(params, int)
        assert params > 0

    def test_prev_frame_needed(self):
        assert Delamain_2_5_PPO().is_prev_frame_needed() is True

    def test_prev_frames_needed(self):
        assert Delamain_2_5_PPO().prev_frames_needed() == 4
