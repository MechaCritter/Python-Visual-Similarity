Classical methods
=================

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Tutorial
     - Description
     - Notebook
   * - VLAD and Fisher Vector
     - Learn how ``VLAD`` and ``Fisher Vector``, previously state-of-the-art methods for
       image retrieval, work.
     - :doc:`VLAD and Fisher Vectors <notebooks/vlad_and_fisher_with_resnet18_deep_features>`
   * - Embedder pipeline
     - Combine a ``VLAD`` and a ``Fisher Vector`` embedder on deep features into a
       ``Pipeline`` and compare images with the concatenated vectors.
     - :doc:`Pipeline <notebooks/pipeline>`
   * - Custom feature extractor
     - Write your own feature extractor by inheriting from
       ``FeatureExtractorBase``, train a ``VLAD`` embedder with it and compare
       two images.
     - :doc:`Custom Feature Extractor <notebooks/custom_feature_extractor_with_orb>`

.. toctree::
   :hidden:

   notebooks/vlad_and_fisher_with_resnet18_deep_features
   notebooks/pipeline
   notebooks/custom_feature_extractor_with_orb
