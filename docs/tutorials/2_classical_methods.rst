Classical methods
=================

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Tutorial
     - Description
     - Link
   * - Introduction: VLAD and Fisher Vectors
     - Train classical embedders ``VLAD`` and ``Fisher Vector`` on ``RootSIFT``
       features, previously state-of-the-art methods for image retrieval.
     - :doc:`VLAD and Fisher Vectors <notebooks/vlad_and_fisher_vector_introduction>`
   * - VLAD and Fisher Vector with Deep CNN features
     - Train classical embedders ``VLAD`` and ``Fisher Vector`` on deep features extracted
       from activation maps of a pretrained CNN.
     - :doc:`VLAD and Fisher Vectors with deep Features <notebooks/vlad_and_fisher_with_resnet18_deep_features>`
   * - Embedder pipeline
     - Combine a ``VLAD`` and a ``Fisher Vector`` embedder on deep CNN features
       into a ``Pipeline`` and compare images with the concatenated vectors.
     - :doc:`Pipeline <notebooks/pipeline>`
   * - Custom RootSIFT feature extractor
     - Implement ``RootSIFT`` on top of the ``SIFT`` implementation of
       ``OpenCV`` and inherit ``FeatureExtractorBase`` instead of the built-in
       ``RootSIFT`` module, and train a ``VLAD`` embedder with it.
     - :doc:`Custom Feature Extractor with RootSIFT <notebooks/custom_feature_extractor_with_rootsift>`
   * - Custom `ORB` feature extractor
     - Write your own ``ORB`` feature extractor by inheriting from
       ``FeatureExtractorBase``, train a ``VLAD`` embedder with it and compare
       two images.
     - :doc:`Custom Feature Extractor with ORB <notebooks/custom_feature_extractor_with_orb>`

.. toctree::
   :hidden:

   notebooks/vlad_and_fisher_vector_introduction
   notebooks/vlad_and_fisher_with_resnet18_deep_features
   notebooks/pipeline
   notebooks/custom_feature_extractor_with_rootsift
   notebooks/custom_feature_extractor_with_orb
