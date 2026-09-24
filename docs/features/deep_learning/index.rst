Deep learning based features
============================

Includes extractors that take their descriptors from the feature maps of a
neural network. They live in ``pyvisim.neural_networks.features`` and need the
``nn`` extra.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Object
     - ``output_dim``
     - Notes
   * - :doc:`DeepConvFeature <deep_conv_feature/deep_conv_feature>`
     - layer channels
     - Neural Network feature maps

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   deep_conv_feature/deep_conv_feature
