BCESiameseNetwork
=================

Both images are passed through the same shared-weight ``backbone`` and
projection ``head``; each branch output is squashed with a sigmoid into a
feature vector ``h in (0, 1)^D`` (the paper's final fully-connected layer
uses sigmoid units). The two branches are then combined by their
component-wise L1 distance, and a single learned linear layer maps that
distance vector to the probability of the pair showing the same class:

.. math::

   p(x_1, x_2)
   = \sigma\Bigl(\sum_{j} \alpha_j \, \bigl| h_{1,j} - h_{2,j} \bigr|
   + b\Bigr)

where the weights :math:`\alpha_j` learn the importance of each feature
dimension, so unlike :class:`ContrastiveSiameseNetwork` the comparison
metric itself is trained. The network is a binary classifier over pairs and
is trained with binary cross-entropy on labels ``1`` (same class) / ``0``
(different class); :meth:`~pyvisim.neural_networks.BCESiameseNetwork.forward`
returns raw logits so it composes with
:class:`torch.nn.BCEWithLogitsLoss` in a numerically stable way.

Following diagram visualizes this:

.. code-block:: text

                     ┌──────────┐    ┌────────────────┐    ┌─────────┐
   Input Image A ───►│ Backbone │───►│ Embedding Head │───►│ Sigmoid │───► Features A ──┐
                     └──────────┘    └────────────────┘    └─────────┘                  │
                          ╎                  ╎                  ╎                       │    ┌─────────┐    ┌───────────────┐
                          ╎                  ╎                  ╎                       ├───►│ |A - B| │───►│ Scoring Layer │───► P(same class)
                          ╎                  ╎                  ╎                       │    └─────────┘    └───────────────┘
                     ┌──────────┐    ┌────────────────┐    ┌─────────┐                  │
   Input Image B ───►│ Backbone │───►│ Embedding Head │───►│ Sigmoid │───► Features B ──┘
                     └──────────┘    └────────────────┘    └─────────┘
                          (Shared Weights)

Example: training a BCE Siamese Network
---------------------------------------

See :doc:`this tutorial </tutorials/notebooks/siamese_network>`.

API reference
-------------

.. autoclass:: pyvisim.neural_networks.BCESiameseNetwork
   :members:
   :inherited-members: Module
   :show-inheritance:
