### Preprocessing steps

1. Download the dataset: Visit the [Oxford Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) page, and download the files:
    - `1. Dataset images`: 8189 images of 102 flower categories.
    - `4. The Image Labels`: this downloads a MATLAB file named `imagelabels.mat` containing the labels for the images.
    - `5. The Data Splits`: this downloads a MATLAB file named `setid.mat` containing the set ids indicating if the images are for training, testing, or validation.

2. Organize the data: Once downloaded, the files should be organized in a specific manner to ensure that they are read correctly when used. The structure should look like this:
``` 
   oxford_flower_dataset/
   ├── images
   │   ├── image_00001.jpg
   │   ├── image_00002.jpg
   │   └── ...
   │   ├── image_08189.jpg
   ├── imagelabels.mat
   └── setid.mat
```

1. Using the Dataset: Here's an example of how to pass the essential parameters to initiate an instance of the DataSet class:

``` python
dataset = OxfordFlowerDataset(
    image_dir="path_to_oxford_flower_dataset/images",
    image_labels_file="path_to_oxford_flower_dataset/imagelabels.mat",
    set_id_file="path_to_oxford_flower_dataset/setid.mat",
    purpose='train')
```

Iterating through this dataset yields the image matrix, the image label, and the image full
path, in this order:

```python
for image, label, path in dataset:
    print("Image shape:", image.shape)
    print("Image label:", label)
    print("Image path:", os.path.basename(path))
```

## Note on splits
According to the annotations of the `imagelabels.mat` file, the original dataset has 1020 training images, 1020 validation images, 
and **6149 testing images**. I am not sure if this is intentional, but I swapped the training and testing splits to have more images for training
clustering models.

## Important Notes on Deep Features

If you are extracting deep features using `DeepConvFeature`, a transformer can be passed to the feature extractor itself. 
Therefore, do not use the transformer on the dataset itself, as this will transform the image twice.
Another important note is that, as per the current implementation, feature extractors only accept **numpy images with data range 0-255**. 
Further work is needed to allow users more flexibility in the input data format.

Also, the `DeepConvFeature` extractor has **only been tested using the VGG16 model**. Further work is required to validate and test
the functionality with other models to ensure compatibility and performance.