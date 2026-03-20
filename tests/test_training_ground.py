import pytest

from TrainingGround import TrainingGround
from enviroment.Algorithms import Algorithms
from enviroment.Agent import Agent
from enviroment.AgentDDQN import AgentDDQN
from enviroment.AgentPPO import AgentPPO
from alternative_models.Delamain import Delamain
from alternative_models.Delamain_2 import Delamain_2
from alternative_models.Delamain_2_1 import Delamain_2_1
from alternative_models.Delamain_2_5 import Delamain_2_5, Delamain_2_5_PPO


@pytest.fixture
def tg():
    """Create a TrainingGround with default init, then patch attributes for testing."""
    # We need to bypass __init__ since it reads YAML and creates env
    tg = TrainingGround.__new__(TrainingGround)
    tg.algorithm = Algorithms.DQN
    return tg


class TestParseClassName:
    def test_delamain(self, tg):
        tg.algorithm = Algorithms.DQN
        assert tg.parse_class_name("Delamain") is Delamain

    def test_delamain_2(self, tg):
        assert tg.parse_class_name("Delamain_2") is Delamain_2

    def test_delamain_2_1(self, tg):
        assert tg.parse_class_name("Delamain_2_1") is Delamain_2_1

    def test_delamain_2_5_dqn(self, tg):
        tg.algorithm = Algorithms.DQN
        assert tg.parse_class_name("Delamain_2_5") is Delamain_2_5

    def test_delamain_2_5_ddqn(self, tg):
        tg.algorithm = Algorithms.DDQN
        assert tg.parse_class_name("Delamain_2_5") is Delamain_2_5

    def test_delamain_2_5_ppo(self, tg):
        tg.algorithm = Algorithms.PPO
        assert tg.parse_class_name("Delamain_2_5") is Delamain_2_5_PPO

    def test_unknown_returns_delamain(self, tg):
        assert tg.parse_class_name("NonExistent") is Delamain

    def test_none_returns_delamain(self, tg):
        assert tg.parse_class_name(None) is Delamain


class TestParseAlgorithm:
    def test_dqn(self, tg):
        assert tg.parse_algorithm(Algorithms.DQN) is Agent

    def test_ddqn(self, tg):
        assert tg.parse_algorithm(Algorithms.DDQN) is AgentDDQN

    def test_ppo(self, tg):
        assert tg.parse_algorithm(Algorithms.PPO) is AgentPPO

    def test_unknown_returns_agent(self, tg):
        assert tg.parse_algorithm(None) is Agent


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
