# arc42: dataset

Software architecture of `pyvisim.datasets`. This document is for developers and
is not part of the published documentation.

## Architecture decisions

### The train and test splits are swapped

`OxfordFlowerDataset` maps the original test ids to `train` and the original
train ids to `test`, which turns the shipped 1020/1020/6149 split into a
6149/1020/1020 one. Training therefore gets the larger pool, and that is what
the clustering models of the classic embedders benefit from.

Because of this, numbers measured on this dataset are not comparable to papers
that use the original split, so the swap is also stated on the public page.
