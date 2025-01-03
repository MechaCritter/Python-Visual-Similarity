"""
Defines an abstract base class for feature extractors, and a concrete
'CustomFeatureExtractor' that allows passing any user function + parameters.
"""

import abc
import logging
from typing import Any

import cv2
import numpy as np
import torch

from src.config import setup_logging

setup_logging()


class FeatureExtractorBase(abc.ABC):
    """
    Abstract interface for extracting features from images.

    A feature extractor transforms an image (NumPy array) into a
    set of descriptors (NumPy array).
    """
    _logger = logging.getLogger("Feature_Extractor")
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts features from an image.

        :param image: Input image (NumPy array).
        :return: Feature descriptors (NumPy array).
        """
        pass


class SIFT(FeatureExtractorBase):
    def __init__(self):
        super().__init__()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts SIFT features from an image.
        :param image:
        :return:
        """
        sift = cv2.SIFT.create()
        _, descriptors = sift.detectAndCompute(image, None)
        return descriptors

class RootSIFT(FeatureExtractorBase):
    def __init__(self):
        super().__init__()

    def __call__(self, image: np.ndarray) -> np.ndarray:
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
    def __init__(self, func: Any):
        super().__init__()
        if not callable(func):
            raise ValueError(f"Argument func must be a callable object, got {type(func)} instead")
        
        # self.signature = inspect.signature(func)
        # self.name = func.__name__ if hasattr(func, "__name__") else str(func)
        # functools.wraps(func)(self)
        self.func = func

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.func(img)

class DeepConvFeatureExtractor(FeatureExtractorBase):
    """
    Extracts convolutional feature maps from a chosen conv layer of a torchvision model.
    It flattens the feature maps into feature descriptors. Optionally appends
    normalized (x, y) coordinates to each spatial location.

    :param model_name: Name of the torchvision model (string), e.g. 'vgg16'
    :param layer_index: Which conv layer to hook (int). Use `list_conv_layers(...)`
                       to see the ordering or use -1 for the last conv layer.
    :param append_spatial_coords: If True, appends (x/W, y/H) to each descriptor.
    :param device: 'cpu' or 'cuda'. Where to run the model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_index: int = -1,
        append_spatial_coords: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        self.model = model
        self.layer_index = layer_index
        self.append_spatial_coords = append_spatial_coords
        self.device = device

        self._logger.info(f"Device used: {self.device}")

        self.conv_layers = self.list_conv_layers(self.model)
        if not self.conv_layers:
            raise ValueError(f"No convolutional layers found in model {self.model._get_name()}.")

        self.buffer = None  # will store the activation
        try:
            _, self.selected_layer_name, self.selected_layer_module = self.conv_layers[self.layer_index]
            self._logger.info(f"Selected layer: {self.selected_layer_name}, {self.selected_layer_module}")
        except IndexError:
            raise IndexError(f"Model {self.model._get_name()} has only {len(self.conv_layers)} convolutional layers. Got layer_index={self.layer_index}.")
        self._register_hook()

    def list_conv_layers(self, model: torch.nn.Module) -> list[tuple[int, str, torch.nn.Module]]:
        """
        Utility function to collect convolutional layers (and sub-modules)
        from the given PyTorch model in order of appearance.

        :param model: A PyTorch model (e.g., torchvision.models.vgg16).
        :return: List of (layer_index, layer_module) for each convolutional layer.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Currently, only torch.nn.Module is supported. Got {type(model)} instead.")
        conv_layers = []
        idx = 0
        for name, module in model.named_modules():
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
            # output shape: [batch_size, channels, height, width]
            self.buffer = output.detach()

        self.hook = self.selected_layer_module.register_forward_hook(hook_fn)

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        """
        #TODO: first, check if image is tensor and has range [0,1]. If numpy and has range [0,255], normalize and convert to tensor. If numpy and has range [0,1], convert to tensor. Else, raise error.
        Processes a single image through the chosen conv layer and
        returns flattened feature descriptors, optionally appending
        normalized spatial coordinates.

        :param image: Input image as a NumPy array (H x W x C, BGR or RGB).
        :return: N x D NumPy array, where N = (H_conv x W_conv) and
                 D = number_of_channels (+ 2 if spatial coords are appended).
        """
        # 1. Convert to torch tensor, add batch dimension, move to device
        #    We'll assume image is in [H,W,C], scale [0,255], in RGB or BGR
        #    The user is responsible for pre-processing if needed (mean/std).
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        self.model.eval()
        self.model.to(self.device)
        _ = self.model(image.unsqueeze(0))  # we only care about the hook's output
        if self.buffer is None:
            raise RuntimeError("Forward hook did not capture any features.")

        # 3. Convert the captured feature map to NumPy
        feature_map = self.buffer.cpu().numpy()  # shape: (1, C, Hf, Wf)
        feature_map = feature_map[0]  # Remove batch dimension

        C, Hf, Wf = feature_map.shape
        feature_map = feature_map.reshape(C, -1).T  # shape: (Hf*Wf, C)

        if self.append_spatial_coords:
            coords = []
            for y in range(Hf):
                for x in range(Wf):
                    coords.append([x / Wf, y / Hf])  # (x/Wf, y/Hf)
            coords = np.array(coords, dtype=np.float32)  # shape: (Hf*Wf, 2)
            # Concatenate
            feature_map = np.hstack([feature_map, coords]) # shape: (Hf*Wf, C+2)

        return feature_map