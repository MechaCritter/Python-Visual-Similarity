# OxfordFlowerDataset

File: [`datasets/datasets.py`](../../pyvisim/datasets/datasets.py)

A PyTorch `Dataset` for the [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
dataset (8189 images across 102 categories). Indexing yields a
`(image, label, image_path)` tuple, where `image` is an RGB NumPy array.

```python
from pyvisim.datasets import OxfordFlowerDataset

dataset = OxfordFlowerDataset(purpose="train")
image, label, path = dataset[0]
```

## The swapped train/test split

The constructor's `purpose` accepts `"train"`, `"validation"`, `"test"`, or a list to
combine splits (for example `["train", "validation"]`).

Note the deliberate swap: the original dataset ships 1020 training and 6149 test images.
This class maps the original **test** ids to `train` and the original **train** ids to
`test`, so training has the larger pool. This is more useful for fitting the clustering
models, which benefit from more data. Keep this in mind if you compare results against
papers that use the original split.

## TODO

- Implement `transform` method
