### Overview

The dataset can be downloaded from  [The Website of the University of Oxford's Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). 
In this implementation, following files will be downloaded as you initiate the dataset: \
    - `Dataset images`: 8189 images of 102 flower categories.\
    - `The Image Labels`: a MATLAB file named `imagelabels.mat` containing the labels for the images.\
    - `The Data Splits`: a MATLAB file named `setid.mat` containing the set ids indicating if the images are for training, testing, or validation.

Iterating through this dataset yields the image array, the image label, and the image full
path, in this order:

```python
from pyvisim.datasets import OxfordFlowerDataset
import os

dataset = OxfordFlowerDataset()
for image, label, path in dataset:
    print("Image shape:", image.shape)
    print("Image label:", label)
    print("Image path:", os.path.basename(path))
```

### Note on splits
According to the annotations of the `imagelabels.mat` file, the original dataset has 1020 training images, 1020 validation images, 
and **6149 testing images**. I am not sure if this is intentional, but I swapped the training and testing splits to have more images for training
clustering models.
