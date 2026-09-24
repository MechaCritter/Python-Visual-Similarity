DeepConvFeature
===============

``DeepConvFeature`` flattens the feature maps of one convolutional layer of a
:doc:`backbone </neural_networks/backbones/backbones>` into local
descriptors. The :doc:`VLAD </classic/vlad/vlad>` and :doc:`Fisher Vector
</classic/fisher_vector/fisher_vector>` embedders aggregate them the same way
they aggregate ``SIFT`` descriptors.

API reference
-------------

.. autoclass:: pyvisim.neural_networks.features.DeepConvFeature
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:
