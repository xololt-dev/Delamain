# Useful test commands

## All wrapper tests

python -m pytest tests/test_wrappers.py -v

## Visual snapshots (greyscale, hsl)

python -m pytest tests/test_wrappers.py::TestVisualSnapshots -v -s

## Antialias visual snapshots (gaussian, edge-aware)

python -m pytest tests/test_wrappers.py::TestAntialiasVisualSnapshots -v -s

## Antialias functional tests only

python -m pytest tests/test_wrappers.py::TestGaussianAntialiasObservation tests/test_wrappers.py::TestGaussianAntialiasObservationVec tests/test_wrappers.py::TestEdgeAntialiasObservation tests/test_wrappers.py::TestEdgeAntialiasObservationVec -v

## Optical flow visual snapshots (greyscale, hsl, sequence)

python -m pytest tests/test_wrappers.py::TestOpticalFlowVisualSnapshots -v -s

## YAML parsing tests

python -m pytest tests/test_yaml_parsing.py -v

## Agent save/load tests

python -m pytest tests/test_agents.py::TestAgentSaveLoad tests/test_agents.py::TestPPOSaveLoad -v

## Checkpoint smoke tests (validate existing .pt files on disk)

python -m pytest -m checkpoint -v

## Full test suite (excludes checkpoint smoke tests)

python -m pytest tests/ -v

## Full test suite including checkpoint smoke tests

python -m pytest tests/ -v -m ""

## Full test suite (excludes checkpoint smoke tests) with coverage

python -m pytest tests/ -v --cov=. --cov-report=term --cov-report=html --cov-config=.coveragerc
