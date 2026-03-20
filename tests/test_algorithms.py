from enviroment.Algorithms import Algorithms


class TestAlgorithms:
    def test_has_three_members(self):
        assert len(Algorithms) == 3

    def test_dqn_value(self):
        assert Algorithms.DQN.value == "DQN"

    def test_ddqn_value(self):
        assert Algorithms.DDQN.value == "DDQN"

    def test_ppo_value(self):
        assert Algorithms.PPO.value == "PPO"

    def test_lookup_by_name(self):
        assert Algorithms["DQN"] == Algorithms.DQN
        assert Algorithms["DDQN"] == Algorithms.DDQN
        assert Algorithms["PPO"] == Algorithms.PPO
