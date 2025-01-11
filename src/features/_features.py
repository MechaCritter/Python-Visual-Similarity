"""
This script defines various feature extractors used for feature-based image encoders 
such as VLAD (Vector of Locally Aggregated Descriptors) or Fisher Vectors. It includes 
implementations for handcrafted features (e.g., SIFT, RootSIFT), user-defined 
custom feature extraction functions, and deep convolutional feature extraction 
with optional spatial encoding.
"""

from functools import wraps
from typing import Callable

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

from src._base_classes import FeatureExtractorBase
from src.config import setup_logging
from src.utils import check_is_image

setup_logging()


def _check_output_shape(func) -> Callable:
    """
    Ensures the feature extractor output is a 2D NumPy array of shape
    (num_vectors, self.output_dim).
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> np.ndarray:
        feat_vecs = func(self, *args, **kwargs)
        if feat_vecs is None:
            print("No feature vectors found. Returning empty array.")
            return np.zeros((0, self.output_dim), dtype=np.float32)

        if not isinstance(feat_vecs, np.ndarray):
            raise ValueError(f"Expected output to be a NumPy array, got {type(feat_vecs)} instead.")

        if feat_vecs.ndim != 2:
            raise ValueError(f"Feature extractor output must be 2D. Got shape {feat_vecs.shape}.")

        if feat_vecs.shape[1] != self.output_dim:
            raise ValueError(f"Expected feat_vecs.shape[1] == {self.output_dim}, "
                             f"but got {feat_vecs.shape[1]}.")

        return feat_vecs

    return wrapper


class SIFT(FeatureExtractorBase):
    """
    Scale-Invariant Feature Transform (SIFT) feature extractor.

    References:
    ===========
    [1] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
    """
    def __init__(self):
        super().__init__()
        self._output_dim = 128

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @_check_output_shape
    @check_is_image(arg_positions=(1,))
    def __call__(self, image: np.ndarray, /) -> np.ndarray:
        """
        Extracts SIFT features from an image.
        :param image:
        :return:
        """
        sift = cv2.SIFT.create()
        _, descriptors = sift.detectAndCompute(image, None)
        return descriptors


class RootSIFT(FeatureExtractorBase):
    """
    Scale-Invariant Feature Transform with Hellinger kernel (RootSIFT) normalizer.

    References:
    ===========
    [1] Arandjelovic, R., & Zisserman, A. (2012). Three things everyone should know to improve object retrieval.
    """
    def __init__(self):
        super().__init__()
        self._output_dim = 128

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @_check_output_shape
    @check_is_image(arg_positions=(1,))
    def __call__(self, image: np.ndarray,/) -> np.ndarray:
        """
        Extracts RootSIFT features from an image.
        :param image:
        :return:
        """
        sift = cv2.SIFT.create()
        _, descriptors = sift.detectAndCompute(image, None)
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors)
        return descriptors


class Lambda(FeatureExtractorBase):
    """
    Lambda feature extractor that allows passing any user-defined
    function to extract features from images.

    The function must accept a single argument (image as NumPy array),
    and output fixed-size feature vectors from each image.
    """
    def __init__(self, func: Callable, output_dim: int):
        """
        Initializes the Lambda feature extractor.
        :param func:
        :param output_dim:
        """
        super().__init__()
        if not callable(func):
            raise ValueError(f"Argument func must be a callable object, got {type(func)} instead")
        self._output_dim = output_dim
        self.func = func

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @_check_output_shape
    @check_is_image(arg_positions=(1,))
    def __call__(self, image: np.ndarray, /) -> np.ndarray:
        return self.func(image)


class DeepConvFeature(FeatureExtractorBase):
    """
    Extracts convolutional feature maps from a chosen conv layer of a torchvision model.
    It flattens the feature maps into feature descriptors. Optionally appends
    normalized (x, y) coordinates to each spatial location.

    The concepts here were inspired by by the work on `VLAD-DCNN` features for face verification, as
    presented in [1], where VLAD encodings were computed from deep convolutional features and input into
    a metric learning algorithm in order to distinguish between different people.

    :param model: A PyTorch model instance from torchvision.model_files. Default is VGG16. In the paper [1], a VGG-Face model trained on Idmb-Wiki
    dataset was used with VLAD encoding for younger faces verification.
    :param target_submodule: Optional submodule name to hook into. If None, the whole model is used.
    :param layer_index: Which conv layer to hook (int). Use `list_conv_layers(...)`
                       to see the ordering or use -1 for the last conv layer.
    :param spatial_encoding: If True, appends (x/W, y/H) to each descriptor.
    :param device: 'cpu' or 'cuda'. Where to run the model.
    :param transform: Optional torchvision.transforms.Compose. Default includes `to_tensor`, `resize(224, 224)`,
                        and normalization with ImageNet stats.

    References:
    ===========
    [1] Liangliang Wang and Deepu Rajan, "An Image Similarity Descriptor for Classification Tasks," J. Vis. Commun. Image R., vol. 71, pp. 102847, 2020.
    [2] Weixia Zhang, Jia Yan, Wenxuan Shi, Tianpeng Feng, and Dexiang Deng, "Refining Deep Convolutional Features for Improving Fine-Grained Image 
    Recognition," EURASIP Journal on Image and Video Processing, 2017.
    """
    def __init__(
        self,
        model: torch.nn.Module = vgg16(weights=VGG16_Weights.DEFAULT),
        target_submodule: str = None,
        layer_index: int = -1,
        spatial_encoding: bool = True,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        transform: transforms.Compose = None
    ):
        super().__init__()
        self._model = None
        self.layer_index = layer_index
        self.spatial_encoding = spatial_encoding
        self.device = device
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((224, 224))])

        self.model: torch.nn.Module  = model # Trigger setter
        self._modules: torch.nn.Module = self._get_submodule(target_submodule)
        self._conv_layers = self.list_conv_layers()
        if not self._conv_layers:
            raise ValueError(f"No convolutional layers found in model {self.model._get_name()}.")

        self.buffer = None
        try:
            _, self.selected_layer_name, self.selected_layer_module = self._conv_layers[self.layer_index]
            self._logger.info(f"Selected layer: {self.selected_layer_name}, {self.selected_layer_module}")
        except IndexError:
            info = "" if target_submodule is None else f" in submodule {self._modules._get_name()}"
            raise IndexError(f"Model {self.model._get_name()} has only {len(self._conv_layers)} convolutional layers {info}"
                             f". Got layer_index={self.layer_index}.")
        self._output_dim = self.selected_layer_module.out_channels + 2 if self.spatial_encoding else self.selected_layer_module.out_channels
        self._register_hook()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Currently, only torch.nn.Module is supported. Got {type(model)} instead.")
        self._model = model

    def _get_submodule(self, submodule_name: str = None) -> torch.nn.Module:
        """
        Retrieves a submodule from a PyTorch model by name.

        :return: The submodule instance.
        """
        if submodule_name is None:
            return self._model
        if not hasattr(self._model, submodule_name):
            raise AttributeError(f"Model {self.model._get_name()} has no submodule named {submodule_name}.")
        return getattr(self._model, submodule_name)

    def list_conv_layers(self) -> list[tuple[int, str, torch.nn.Module]]:
        """
        Utility function to collect convolutional layers (and sub-modules)
        from the model / chosen submodule.

        :return: List of (layer_index, layer_module) for each convolutional layer.
        """
        conv_layers = []
        idx = 0
        for name, module in self._modules.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((idx, name, module))
                idx += 1
        return conv_layers

    def _register_hook(self):
        """
        Registers a forward hook on the selected convolutional layer
        to capture its output (feature map).
        """
        def hook_fn(module, input, output):
            self.buffer = output.detach() # output shape: [batch_size, channels, height, width]
        self.hook = self.selected_layer_module.register_forward_hook(hook_fn)

    @_check_output_shape
    @check_is_image(arg_positions=(1,))
    def __call__(self, image: np.ndarray, /) -> np.ndarray:
        """
        #TODO: first, check if image is tensor and has range [0,1]. If numpy and has range [0,255], normalize and convert to tensor. If numpy and has range [0,1], convert to tensor. Else, raise error.
        Processes a single image through the chosen conv layer and
        returns flattened feature descriptors.

        :param image: Input image as a NumPy array (H x W x C, BGR or RGB).
        :return: N x D NumPy array, where N = (H_conv x W_conv) and
                 D = number_of_channels (+ 2 if spatial coords are appended).
        """
        image = self.transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        self.model.to(self.device)
        _ = self.model(image)  # we only care about the hook's output
        if self.buffer is None:
            raise RuntimeError("Forward hook did not capture any features.")

        # Convert the captured feature map to NumPy
        feature_map = self.buffer.cpu().numpy()  # shape: (1, C, Hf, Wf)
        feature_map = feature_map[0]  # Remove batch dimension

        C, Hf, Wf = feature_map.shape
        feature_map = feature_map.reshape(C, -1).T  # shape: (Hf*Wf, C)

        if self.spatial_encoding:
            coords = []
            for y in range(Hf):
                for x in range(Wf):
                    coords.append([x / Wf, y / Hf])  # (x/Wf, y/Hf)
            coords = np.array(coords, dtype=np.float32)  # shape: (Hf*Wf, 2)
            # Concatenate
            feature_map = np.hstack([feature_map, coords]) # shape: (Hf*Wf, C+2)

        return feature_map