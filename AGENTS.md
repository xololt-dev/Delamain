# Delamain

## Info

Python project made using:

- pytorch
- opencv
- gymnasium [box2d]

Goal: have agents learn to travel CarRacing-v3 environment.

Superparameters loaded from training_params.yaml file

- its description is under PARAMS.md

If you need to run scripts, use created venv/

## Project structure

alternative_models/ - inside there are architecture descriptions of models
environment/ - inside wrappers and on and off-policy agent classes
rocm-pytorch/ - should ignore, only for human usage
tests/ - tests
training/ - saved checkpoints as .pt files
venv/ - virtual environment
main.py - main file
training_params.yaml - experiment parameters
TrainingGround.py - main class that handles the whole environment-agent interaction

### Tests

More info under tests/TESTS.md

pytest.ini - includes checkpoint marker for opt-in smoke tests
