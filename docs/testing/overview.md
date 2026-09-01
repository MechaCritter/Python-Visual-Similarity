# Trigger slow tests

Add the label `run-slow` to the PR to trigger the slow tests.

## Neural Network tests

Some tests use stub backbones (see `test_bce_siamese_network.py` for an example) because
the real backbones are large and need to be downloaded in the first place.

There are separate tests that use the real backbones and marked as ``slow``. You don't need
to execute these if you don't have enough resources on your computer.
