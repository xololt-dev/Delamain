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

## Full test suite
python -m pytest tests/ -v
