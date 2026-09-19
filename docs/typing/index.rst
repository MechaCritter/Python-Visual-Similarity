Typing
======

``pyvisim.typing`` is the contains public types for annotation as well as
normalization helpers that are used across this library.

Types
-----

``MatLike``
~~~~~~~~~~~

.. code-block:: python

   from pyvisim.typing import MatLike

Anything that can be treated as a numerical image array:

- a NumPy ``ndarray``, the library's internal representation
- a PyTorch ``Tensor``, converted to NumPy automatically before the feature
  extractor sees it
- anything :func:`numpy.asarray` can turn into a numeric array: nested lists of
  numbers, objects with ``__array__``, and so on

``ImageInput``
~~~~~~~~~~~~~~

.. code-block:: python

   from pyvisim.typing import ImageInput

The widest input type accepted wherever the library takes image data, for
example ``embed``, ``learn``, and ``similarity_score``. It covers a single
image, a single batched array/tensor, and an iterable of individual images such as a
generator over a large dataset. Both NumPy arrays and PyTorch tensors are accepted.

``Embedder``
~~~~~~~~~~~~

.. code-block:: python

   from pyvisim.typing import Embedder

A type that represents all image embedders in this library. Example: ``ClipEmbedder``.

``EmbeddingStore``
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyvisim.typing import EmbeddingStore

A structural type for a gallery of embedded images. Example: ``InMemoryEmbeddingStore``.

``SearchIndex``
~~~~~~~~~~~~~~~

.. code-block:: python

   from pyvisim.typing import SearchIndex

A structural type for the index a store searches through. Example: ``HnswIndex``.

Keyword arguments for image data
--------------------------------

``dims`` and ``value_range`` are important parameters that describe how the images 
are read by this library. These are used by methods like ``embed``, ``learn``, 
and ``similarity_score``.


The ``dims`` string
~~~~~~~~~~~~~~~~~~~

``dims`` tells the library how to read your array's axes, one character per
dimension in the exact order the axes appear:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Character
     - Axis
   * - ``"H"``
     - height (rows)
   * - ``"W"``
     - width (columns)
   * - ``"C"``
     - channels (e.g. RGB)
   * - ``"B"``
     - batch size

``"H"`` and ``"W"`` are mandatory, ``"C"`` and ``"B"`` are optional. The
default is ``"HWC"``, which is the standard NumPy/OpenCV single-image layout.
Common layouts:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - ``dims``
     - Shape meaning
     - Typical source
   * - ``"HWC"``
     - height x width x channels
     - NumPy / OpenCV (**default**)
   * - ``"CHW"``
     - channels x height x width
     - PyTorch single image (``torch.Tensor``)
   * - ``"BHWC"``
     - batch x height x width x channels
     - NumPy batch
   * - ``"BCHW"``
     - batch x channels x height x width
     - PyTorch batched ``Tensor``
   * - ``"HWCB"``
     - height x width x channels x batch
     - some data loaders
   * - ``"HW"``
     - height x width only (grayscale)
     - grayscale images

When ``"B"`` is present the batch is automatically split so every image is
processed individually. You do not need to loop yourself.

``dims`` is **case-insensitive**: ``"hwc"``, ``"HWC"``, ``"Hwc"`` all work the
same way.

Example:

.. code-block:: python

   import numpy as np

   from pyvisim.neural_networks import ClipEmbedder

   rng = np.random.default_rng(0)
   image1 = rng.integers(64, 256, size=(64, 80, 3), dtype=np.uint8)
   image2 = rng.integers(64, 256, size=(64, 80, 3), dtype=np.uint8)

   embedder = ClipEmbedder("ViT-B-32", pretrained="openai", device="cpu")

   # baseline: NumPy (H, W, C) layout
   score_hwc = embedder.similarity_score(image1, image2, dims="HWC")

   # same pixels, PyTorch (C, H, W) layout, declared via dims
   score_chw = embedder.similarity_score(
       image1.transpose(2, 0, 1), image2.transpose(2, 0, 1), dims="CHW"
   )

   # The results should be the same
   print(score_hwc, score_chw)


The ``value_range`` tuple
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   value_range: tuple[float, float] = (0.0, 255.0)  # default

Tells the library what numerical range your input values live in. Pixels are
rescaled into ``[0, 255]`` before feature extraction. Common cases:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Your image
     - ``value_range`` to pass
   * - ``uint8`` NumPy array, values 0-255
     - nothing, this is the default
   * - float tensor from ``torchvision.transforms.ToTensor()``, values 0-1
     - ``(0.0, 1.0)``
   * - float image, values -1 to 1 (e.g. some augmentation pipelines)
     - ``(-1.0, 1.0)``

If your image is already ``uint8`` in ``(0, 255)``, the rescaling step is a
no-op.

Example:

.. code-block:: python

   import numpy as np

   from pyvisim.neural_networks import ClipEmbedder

   rng = np.random.default_rng(0)
   image1 = rng.integers(64, 256, size=(64, 80, 3), dtype=np.uint8)
   image2 = rng.integers(64, 256, size=(64, 80, 3), dtype=np.uint8)

   embedder = ClipEmbedder("ViT-B-32", pretrained="openai", device="cpu")

   # baseline: uint8 pixels in the default [0, 255] range
   score_uint8 = embedder.similarity_score(image1, image2, value_range=(0.0, 255.0))

   # same pixels rescaled to [-1, 1], declared via value_range
   image1_signed = image1.astype(np.float64) / 127.5 - 1.0
   image2_signed = image2.astype(np.float64) / 127.5 - 1.0
   score_signed = embedder.similarity_score(
       image1_signed, image2_signed, value_range=(-1.0, 1.0)
   )
   
   # The results should be the same
   print(score_uint8, score_signed)