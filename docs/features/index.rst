Features
========

A feature extractor maps one image to a ``(N, D)`` array of local descriptors.
Embedders consume these descriptors and aggregate them into a fixed-size
vector:

.. code-block:: text

   image -> feature extractor -> local descriptors -> embedder -> embedding

The extractors implemented in ``pyvisim`` are split into
:doc:`handcrafted features <handcrafted/index>` and :doc:`deep learning based
features <deep_learning/index>`.

Reconstructing feature extractors
---------------------------------

Every extractor describes itself as a JSON-safe configuration, and
:meth:`~pyvisim.base.FeatureExtractorBase.from_dict` rebuilds the extractor a
description names.

.. code-block:: python

   from pyvisim.base import FeatureExtractorBase
   from pyvisim.features import RootSIFT

   extractor = RootSIFT(n_hist=2, n_ori=4)

   # Serialize extractor
   serialized = extractor.to_dict()

   # Reload extractor
   reloaded = FeatureExtractorBase.from_dict(serialized)

An extractor can also be saved to a ``.safetensors`` file and loaded with the
``load_from_disk`` method of its class.

.. code-block:: python

   from pyvisim.features import RootSIFT

   path = RootSIFT(n_hist=2, n_ori=4).save_to_disk("root_sift.safetensors")
   reloaded = RootSIFT.load_from_disk(path)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   handcrafted/index
   deep_learning/index

Serialization
-------------

Every extractor except :class:`~pyvisim.features.Lambda` can be written to and
read from a JSON-safe dictionary or a ``.safetensors`` file with the methods
below.

.. automethod:: pyvisim.base.FeatureExtractorBase.to_dict

.. automethod:: pyvisim.base.FeatureExtractorBase.from_dict

.. automethod:: pyvisim.base.FeatureExtractorBase.save_to_disk

.. automethod:: pyvisim.base.FeatureExtractorBase.load_from_disk
