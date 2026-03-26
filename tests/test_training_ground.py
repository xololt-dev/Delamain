import pytest
from functools import partial

from TrainingGround import TrainingGround
from environment.Algorithms import Algorithms
from environment.AgentDQN import AgentDQN
from environment.AgentDDQN import AgentDDQN
from environment.AgentPPO import AgentPPO
from alternative_models.Delamain import Delamain
from alternative_models.Delamain_2 import Delamain_2
from alternative_models.Delamain_2_1 import Delamain_2_1
from alternative_models.Delamain_2_5 import Delamain_2_5, Delamain_2_5_PPO
from alternative_models.Delamain_2_6 import Delamain_2_6, Delamain_2_6_PPO


@pytest.fixture
def tg():
    """Create a TrainingGround with default init, then patch attributes for testing."""
    # We need to bypass __init__ since it reads YAML and creates env
    tg = TrainingGround.__new__(TrainingGround)
    tg.algorithm = Algorithms.DQN
    tg._skip_frames = 4
    tg._optical_flow = False
    return tg


class TestParseClassName:
    def test_delamain(self, tg):
        tg.algorithm = Algorithms.DQN
        result = tg.parse_class_name("Delamain")
        assert isinstance(result, partial)
        assert result.func is Delamain

    def test_delamain_2(self, tg):
        result = tg.parse_class_name("Delamain_2")
        assert isinstance(result, partial)
        assert result.func is Delamain_2

    def test_delamain_2_1(self, tg):
        result = tg.parse_class_name("Delamain_2_1")
        assert isinstance(result, partial)
        assert result.func is Delamain_2_1

    def test_delamain_2_5_dqn(self, tg):
        tg.algorithm = Algorithms.DQN
        result = tg.parse_class_name("Delamain_2_5")
        assert isinstance(result, partial)
        assert result.func is Delamain_2_5

    def test_delamain_2_5_ddqn(self, tg):
        tg.algorithm = Algorithms.DDQN
        result = tg.parse_class_name("Delamain_2_5")
        assert isinstance(result, partial)
        assert result.func is Delamain_2_5

    def test_delamain_2_5_ppo(self, tg):
        tg.algorithm = Algorithms.PPO
        result = tg.parse_class_name("Delamain_2_5")
        assert isinstance(result, partial)
        assert result.func is Delamain_2_5_PPO

    def test_delamain_2_6_dqn(self, tg):
        tg.algorithm = Algorithms.DQN
        result = tg.parse_class_name("Delamain_2_6")
        assert isinstance(result, partial)
        assert result.func is Delamain_2_6

    def test_delamain_2_6_ddqn(self, tg):
        tg.algorithm = Algorithms.DDQN
        result = tg.parse_class_name("Delamain_2_6")
        assert isinstance(result, partial)
        assert result.func is Delamain_2_6

    def test_delamain_2_6_ppo(self, tg):
        tg.algorithm = Algorithms.PPO
        result = tg.parse_class_name("Delamain_2_6")
        assert isinstance(result, partial)
        assert result.func is Delamain_2_6_PPO

    def test_unknown_returns_delamain(self, tg):
        result = tg.parse_class_name("NonExistent")
        assert isinstance(result, partial)
        assert result.func is Delamain

    def test_none_returns_delamain(self, tg):
        result = tg.parse_class_name(None)
        assert isinstance(result, partial)
        assert result.func is Delamain

    def test_default_input_size(self, tg):
        result = tg.parse_class_name("Delamain")
        assert result.keywords.get("input_size") == 96

    def test_custom_input_size(self, tg):
        result = tg.parse_class_name("Delamain", input_size=84)
        assert result.keywords.get("input_size") == 84


class TestParseAlgorithm:
    def test_dqn(self, tg):
        assert tg.parse_algorithm(Algorithms.DQN) is AgentDQN

    def test_ddqn(self, tg):
        assert tg.parse_algorithm(Algorithms.DDQN) is AgentDDQN

    def test_ppo(self, tg):
        assert tg.parse_algorithm(Algorithms.PPO) is AgentPPO

    def test_unknown_returns_agent(self, tg):
        assert tg.parse_algorithm(None) is AgentDQN


class TestInitReporting:
    def test_defaults(self):
        tg = TrainingGround.__new__(TrainingGround)
        section = {}
        tg.init_reporting(section)
        assert tg.when2learn == 4
        assert tg.when2sync == 5000
        assert tg.when2save == 50000
        assert tg.when2report == 5000
        assert tg.when2eval == 50000
        assert tg.when2log == 10
        assert tg.report_type == "text"

    def test_custom_values(self):
        tg = TrainingGround.__new__(TrainingGround)
        section = {
            "when2learn": 8,
            "when2sync": 1000,
            "when2save": 10000,
            "when2report": 500,
            "when2eval": 2000,
            "when2log": 5,
            "report_type": "plot",
        }
        tg.init_reporting(section)
        assert tg.when2learn == 8
        assert tg.when2sync == 1000
        assert tg.when2save == 10000
        assert tg.when2report == 500
        assert tg.when2eval == 2000
        assert tg.when2log == 5
        assert tg.report_type == "plot"


class TestFineTune:
    def test_raises_exception(self):
        tg = TrainingGround.__new__(TrainingGround)
        with pytest.raises(Exception, match="fine_tune not supported"):
            tg.fine_tune()
