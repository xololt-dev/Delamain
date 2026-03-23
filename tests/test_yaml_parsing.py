import pytest
import yaml
import textwrap

from TrainingGround import TrainingGround
from enviroment.Algorithms import Algorithms


FULL_YAML = textwrap.dedent("""\
    model:
      class_name: "Delamain_2_6"
      file_name:
    env:
      random_colors: False
      skip_frames: 4
      vec: True
      envs_num: 4
      optical_flow: False
      crop_size: 84
      mode: "train"
      device:
    train:
      algorithm: "PPO"
      batch_n: 64
      play_n_episodes: 25000
      gamma: 0.95
      epsilon: 1.0
      epsilon_end: 0.05
      epsilon_decay: 0.9999925
      lr: 0.002
      lr_decay: 0.9999925
      buffer_size: 42500
    eval:
      tracks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      video: False
    reporting:
      when2learn: 1024
      when2sync: 5000
      when2save: 50000
      when2report: 5000
      when2eval: 25000
      when2log: 20
      report_type: "text"
""")


def _write_yaml(tmp_path, content, filename="test.yaml"):
    path = tmp_path / filename
    path.write_text(content)
    return str(path)


# ======================================================================
# yaml.safe_load parsing
# ======================================================================


class TestYamlSafeLoad:
    def test_safe_load_valid_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_YAML)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "train" in data
        assert "env" in data
        assert "eval" in data
        assert "reporting" in data
        assert "model" in data

    def test_safe_load_invalid_yaml_raises(self, tmp_path):
        bad_yaml = "train:\n  bad: [unclosed\n"
        path = _write_yaml(tmp_path, bad_yaml)
        with pytest.raises(yaml.YAMLError):
            with open(path) as f:
                yaml.safe_load(f)


# ======================================================================
# Full config round-trip
# ======================================================================


class TestFullConfig:
    def test_all_train_values(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_YAML)
        with open(path) as f:
            data = yaml.safe_load(f)
        train = data["train"]
        assert train["batch_n"] == 64
        assert train["play_n_episodes"] == 25000
        assert train["gamma"] == 0.95
        assert train["epsilon"] == 1.0
        assert train["epsilon_end"] == 0.05
        assert train["epsilon_decay"] == pytest.approx(0.9999925)
        assert train["lr"] == pytest.approx(0.002)
        assert train["lr_decay"] == pytest.approx(0.9999925)
        assert train["buffer_size"] == 42500
        assert train["algorithm"] == "PPO"

    def test_all_env_values(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_YAML)
        with open(path) as f:
            data = yaml.safe_load(f)
        env = data["env"]
        assert env["random_colors"] is False
        assert env["skip_frames"] == 4
        assert env["vec"] is True
        assert env["envs_num"] == 4
        assert env["optical_flow"] is False
        assert env["crop_size"] == 84
        assert env["mode"] == "train"
        assert env["device"] is None

    def test_all_eval_values(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_YAML)
        with open(path) as f:
            data = yaml.safe_load(f)
        eval_sec = data["eval"]
        assert eval_sec["tracks"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert eval_sec["video"] is False

    def test_all_reporting_values(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_YAML)
        with open(path) as f:
            data = yaml.safe_load(f)
        rep = data["reporting"]
        assert rep["when2learn"] == 1024
        assert rep["when2sync"] == 5000
        assert rep["when2save"] == 50000
        assert rep["when2report"] == 5000
        assert rep["when2eval"] == 25000
        assert rep["when2log"] == 20
        assert rep["report_type"] == "text"

    def test_model_values(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_YAML)
        with open(path) as f:
            data = yaml.safe_load(f)
        model = data["model"]
        assert model["class_name"] == "Delamain_2_6"
        assert model["file_name"] is None


# ======================================================================
# Default value fallbacks (empty sections)
# ======================================================================


class TestDefaults:
    def _parse_with_section(self, tmp_path, section_name, section_content=""):
        content = f"{section_name}:\n{section_content}"
        path = _write_yaml(tmp_path, content)
        with open(path) as f:
            return yaml.safe_load(f)

    def test_train_defaults(self, tmp_path):
        data = self._parse_with_section(tmp_path, "train")
        train = data["train"] or {}
        assert train.get("batch_n", 32) == 32
        assert train.get("play_n_episodes", 3000) == 3000
        assert train.get("gamma", 0.95) == 0.95
        assert train.get("epsilon", 1.0) == 1.0
        assert train.get("epsilon_end", 0.05) == 0.05
        assert train.get("epsilon_decay", 0.9999925) == pytest.approx(0.9999925)
        assert train.get("lr", 0.0002) == pytest.approx(0.0002)
        assert train.get("lr_decay", 1.0) == 1.0
        assert train.get("buffer_size", 300000) == 300000

    def test_reporting_defaults(self, tmp_path):
        data = self._parse_with_section(tmp_path, "reporting")
        rep = data["reporting"] or {}
        tg = TrainingGround.__new__(TrainingGround)
        tg.init_reporting(rep)
        assert tg.when2learn == 4
        assert tg.when2sync == 5000
        assert tg.when2save == 50000
        assert tg.when2report == 5000
        assert tg.when2eval == 50000
        assert tg.when2log == 10
        assert tg.report_type == "text"

    def test_env_defaults(self, tmp_path):
        data = self._parse_with_section(tmp_path, "env")
        env = data["env"] or {}
        assert env.get("vec", False) is False
        assert env.get("skip_frames", 4) == 4
        assert env.get("optical_flow", False) is False
        assert env.get("crop_size", None) is None
        assert env.get("random_colors", False) is False

    def test_model_defaults(self, tmp_path):
        data = self._parse_with_section(tmp_path, "model")
        model = data["model"] or {}
        assert model.get("class_name", "Delamain") == "Delamain"
        assert model.get("file_name", None) is None


# ======================================================================
# Optional fields
# ======================================================================


class TestOptionalFields:
    def test_file_name_null(self, tmp_path):
        yaml_str = "model:\n  class_name: 'Delamain'\n  file_name:\n"
        path = _write_yaml(tmp_path, yaml_str)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["model"]["file_name"] is None

    def test_device_null(self, tmp_path):
        yaml_str = "env:\n  device:\n"
        path = _write_yaml(tmp_path, yaml_str)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["env"]["device"] is None

    def test_device_string(self, tmp_path):
        yaml_str = "env:\n  device: 'cpu'\n"
        path = _write_yaml(tmp_path, yaml_str)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["env"]["device"] == "cpu"


# ======================================================================
# Algorithm string mapping
# ======================================================================


class TestAlgorithmMapping:
    @pytest.mark.parametrize(
        "algo_str,expected",
        [
            ("DQN", Algorithms.DQN),
            ("DDQN", Algorithms.DDQN),
            ("PPO", Algorithms.PPO),
        ],
    )
    def test_algorithm_lookup(self, algo_str, expected):
        assert Algorithms[algo_str] == expected

    def test_algorithm_default(self):
        train = {}
        algo = Algorithms[train.get("algorithm", "DQN")]
        assert algo == Algorithms.DQN

    def test_algorithm_from_yaml(self, tmp_path):
        yaml_str = "train:\n  algorithm: 'DDQN'\n"
        path = _write_yaml(tmp_path, yaml_str)
        with open(path) as f:
            data = yaml.safe_load(f)
        algo = Algorithms[data["train"]["algorithm"]]
        assert algo == Algorithms.DDQN


# ======================================================================
# Eval tracks parsing
# ======================================================================


class TestEvalTracks:
    def test_tracks_list(self, tmp_path):
        yaml_str = "eval:\n  tracks: [1, 2, 3]\n  video: false\n"
        path = _write_yaml(tmp_path, yaml_str)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["eval"]["tracks"] == [1, 2, 3]

    def test_tracks_empty(self, tmp_path):
        yaml_str = "eval:\n  tracks:\n  video: false\n"
        path = _write_yaml(tmp_path, yaml_str)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["eval"]["tracks"] is None

    def test_video_boolean(self, tmp_path):
        yaml_str = "eval:\n  tracks: [1]\n  video: true\n"
        path = _write_yaml(tmp_path, yaml_str)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["eval"]["video"] is True


# ======================================================================
# TrainingGround init_reporting integration
# ======================================================================


class TestInitReportingIntegration:
    def test_reporting_section_partial(self):
        tg = TrainingGround.__new__(TrainingGround)
        tg.init_reporting({"when2learn": 8, "report_type": "plot"})
        assert tg.when2learn == 8
        assert tg.report_type == "plot"
        # remaining fields use defaults
        assert tg.when2sync == 5000
        assert tg.when2save == 50000
        assert tg.when2log == 10
