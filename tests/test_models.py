import pytest
import torch
from functools import partial

from alternative_models.Delamain import Delamain
from alternative_models.Delamain_2 import Delamain_2
from alternative_models.Delamain_2_1 import Delamain_2_1
from alternative_models.Delamain_2_5 import Delamain_2_5, Delamain_2_5_PPO
from alternative_models.Delamain_2_6 import Delamain_2_6, Delamain_2_6_PPO
from tests.helpers import DEVICES


# --- Models to parametrize ---
# Each entry: (model_class_name, input_fixture_name)
MODELS_DQN = [
    ("Delamain", "sample_state_raw"),
    ("Delamain_2", "sample_state_d2"),
    ("Delamain_2_1", "sample_state_d21"),
    ("Delamain_2_5", "sample_state"),
]

MODELS_DQN_84 = [
    ("Delamain", "sample_state_raw_84"),
    ("Delamain_2", "sample_state_d2_84"),
    ("Delamain_2_1", "sample_state_d21_84"),
    ("Delamain_2_5", "sample_state_84"),
]

MODEL_CLASSES = {
    "Delamain": Delamain,
    "Delamain_2": Delamain_2,
    "Delamain_2_1": Delamain_2_1,
    "Delamain_2_5": Delamain_2_5,
}

INPUT_SIZES = [96, 84]


@pytest.fixture
def model_class(request):
    return MODEL_CLASSES[request.param]


@pytest.fixture
def dqn_input(
    request, sample_state_raw, sample_state_d2, sample_state_d21, sample_state,
    sample_state_raw_84, sample_state_d2_84, sample_state_d21_84, sample_state_84
):
    fixture_map = {
        "sample_state_raw": sample_state_raw,
        "sample_state_d2": sample_state_d2,
        "sample_state_d21": sample_state_d21,
        "sample_state": sample_state,
        "sample_state_raw_84": sample_state_raw_84,
        "sample_state_d2_84": sample_state_d2_84,
        "sample_state_d21_84": sample_state_d21_84,
        "sample_state_84": sample_state_84,
    }
    return fixture_map[request.param]


def _get_input_size_for_fixture(fixture_name):
    return 84 if "_84" in fixture_name else 96


class TestDelamainDQNModels:
    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    @pytest.mark.parametrize(
        "model_class, dqn_input",
        [(name, fixture) for name, fixture in MODELS_DQN + MODELS_DQN_84],
        indirect=["model_class", "dqn_input"],
        ids=[f"{m[0]}_{_get_input_size_for_fixture(m[1])}" for m in MODELS_DQN + MODELS_DQN_84],
    )
    def test_forward_shape(self, model_class, dqn_input, device):
        input_size = dqn_input.shape[1]
        model = model_class(input_size=input_size).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dqn_input.to(device))
        assert output.shape == (2, 5)

    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    @pytest.mark.parametrize(
        "model_class, dqn_input",
        [(name, fixture) for name, fixture in MODELS_DQN + MODELS_DQN_84],
        indirect=["model_class", "dqn_input"],
        ids=[f"{m[0]}_{_get_input_size_for_fixture(m[1])}" for m in MODELS_DQN + MODELS_DQN_84],
    )
    def test_output_is_float32(self, model_class, dqn_input, device):
        input_size = dqn_input.shape[1]
        model = model_class(input_size=input_size).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dqn_input.to(device))
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    @pytest.mark.parametrize(
        "model_class, dqn_input",
        [(name, fixture) for name, fixture in MODELS_DQN + MODELS_DQN_84],
        indirect=["model_class", "dqn_input"],
        ids=[f"{m[0]}_{_get_input_size_for_fixture(m[1])}" for m in MODELS_DQN + MODELS_DQN_84],
    )
    def test_gradient_flow(self, model_class, dqn_input, device):
        input_size = dqn_input.shape[1]
        model = model_class(input_size=input_size).to(device)
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
        [(name, fixture) for name, fixture in MODELS_DQN + MODELS_DQN_84],
        indirect=["model_class", "dqn_input"],
        ids=[f"{m[0]}_{_get_input_size_for_fixture(m[1])}" for m in MODELS_DQN + MODELS_DQN_84],
    )
    def test_get_params_positive(self, model_class, dqn_input):
        input_size = dqn_input.shape[1]
        model = model_class(input_size=input_size)
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


class TestDelamainDQNModelsInputSize:
    """Test that models produce correct FC input sizes for different input resolutions."""

    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_delamain_fc_size(self, input_size):
        model = Delamain(input_size=input_size)
        # Verify by running a forward pass
        x = torch.randint(0, 256, (1, input_size, input_size, 3), dtype=torch.uint8)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_delamain_2_fc_size(self, input_size):
        model = Delamain_2(input_size=input_size)
        x = torch.randint(0, 256, (1, input_size, input_size, 6), dtype=torch.uint8)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_delamain_2_1_fc_size(self, input_size):
        model = Delamain_2_1(input_size=input_size)
        x = torch.randint(0, 256, (1, input_size, input_size, 12), dtype=torch.uint8)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_delamain_2_5_fc_size(self, input_size):
        model = Delamain_2_5(input_size=input_size)
        x = torch.randint(0, 256, (1, input_size, input_size, 12), dtype=torch.uint8)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_delamain_2_6_fc_size(self, input_size):
        model = Delamain_2_6(in_channels=12, input_size=input_size)
        x = torch.randint(0, 256, (1, input_size, input_size, 12), dtype=torch.uint8)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)


class TestDelamainPPO:
    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_forward_returns_tuple(self, device, input_size):
        model = Delamain_2_5_PPO(input_size=input_size).to(device)
        model.eval()
        state = torch.randint(0, 256, (2, input_size, input_size, 12), dtype=torch.uint8, device=device)
        with torch.no_grad():
            actor_out, critic_out = model(state)
        assert actor_out.shape == (2, 5)
        assert critic_out.shape == (2, 1)

    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_forward_dtypes(self, device, input_size):
        model = Delamain_2_5_PPO(input_size=input_size).to(device)
        model.eval()
        state = torch.randint(0, 256, (2, input_size, input_size, 12), dtype=torch.uint8, device=device)
        with torch.no_grad():
            actor_out, critic_out = model(state)
        assert actor_out.dtype == torch.float32
        assert critic_out.dtype == torch.float32

    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_gradient_flow(self, device, input_size):
        model = Delamain_2_5_PPO(input_size=input_size).to(device)
        model.train()
        state = torch.randint(
            0, 256, (2, input_size, input_size, 12), dtype=torch.float32, device=device
        )
        actor_out, critic_out = model(state)
        loss = actor_out.sum() + critic_out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_get_params_positive(self, input_size):
        params = Delamain_2_5_PPO(input_size=input_size).get_params()
        assert isinstance(params, int)
        assert params > 0

    def test_prev_frame_needed(self):
        assert Delamain_2_5_PPO().is_prev_frame_needed() is True

    def test_prev_frames_needed(self):
        assert Delamain_2_5_PPO().prev_frames_needed() == 4


class TestDelamain26PPO:
    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_forward_returns_tuple(self, input_size):
        model = Delamain_2_6_PPO(in_channels=12, input_size=input_size)
        model.eval()
        state = torch.randint(0, 256, (2, input_size, input_size, 12), dtype=torch.uint8)
        with torch.no_grad():
            actor_out, critic_out = model(state)
        assert actor_out.shape == (2, 5)
        assert critic_out.shape == (2, 1)

    @pytest.mark.parametrize("input_size", INPUT_SIZES, ids=lambda s: f"size_{s}")
    def test_gradient_flow(self, input_size):
        model = Delamain_2_6_PPO(in_channels=12, input_size=input_size)
        model.train()
        state = torch.randint(
            0, 256, (2, input_size, input_size, 12), dtype=torch.float32
        )
        actor_out, critic_out = model(state)
        loss = actor_out.sum() + critic_out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
