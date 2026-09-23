TripletNeuralNetwork
====================

A triplet network is a shared-weight embedding network trained on
triplets of anchor, positive (same class) and negative (different class)
images: the anchor is pulled towards the positive and pushed away from
the negative by at least a margin. The three classic "branches" of the
architecture are realized implicitly by weight sharing: every image is
passed through the same ``backbone`` and projection ``head``, and the
embeddings are L2-normalized so that cosine similarity reduces to a dot
product. Following diagram visualizes this:

.. code-block:: text

   Anchor   ───┐
               │    ┌──────────┐    ┌────────────────┐    ┌──────────────┐
   Positive ───┼───►│ Backbone │───►│ Embedding Head │───►│ L2 Normalize │───► Embeddings
               │    └──────────┘    └────────────────┘    └──────────────┘
   Negative ───┘    (Shared Weights)

                 Embedding A + Embedding P + Embedding N
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ Triplet Loss (training) / fixed metric, e.g. cosine (inference) │
   └─────────────────────────────────────────────────────────────────┘

`Triplet loss` is used to train this network, which has the formula:

.. math::

   L(a, p, n) = \max\bigl(0, \, d(a, p) - d(a, n) + m\bigr)

Training with online mining
---------------------------

See :doc:`this tutorial </tutorials/notebooks/triplet_network>`.

Mining strategies
-----------------

Below mining strategies are implemented in the ``TripletLoss``:

.. list-table::
   :header-rows: 1
   :widths: 15 45 25 15

   * - ``mining``
     - What it picks
     - Averaged over
     - Memory
   * - ``"semi_hard"`` (default)
     - For every positive pair, the closest negative that is still farther than
       the positive. Falls back to the anchor's farthest negative when there is
       none.
     - all positive pairs
     - O(B³)
   * - ``"batch_hard"``
     - Per anchor, its farthest positive and its closest negative (Hermans et
       al., 2017).
     - anchors with both
     - O(B²)
   * - ``"batch_all"``
     - Every valid triplet, but only the ones that violate the margin count
       towards the mean. Averaging over all of them would let the trivially
       satisfied majority wash out the signal.
     - violating triplets
     - O(B³)

Saving and loading
------------------

Save the model to disk:

.. code-block:: python

   path = model.save_to_disk("triplet_resnet18")   # -> triplet_resnet18.embedder
   model = TripletNeuralNetwork.load_from_disk(path)

Load the model from disk. Note that the ``transform`` has to be passed in again
because they are not JSON-serializable:

.. code-block:: python

   model = TripletNeuralNetwork.load_from_disk(path, transform=my_transform)

References
----------

1. **Deep Metric Learning Using Triplet Network** (Hoffer & Ailon, 2014)
   https://arxiv.org/abs/1412.6622

2. **FaceNet: A Unified Embedding for Face Recognition and Clustering**
   (Schroff, Kalenichenko, & Philbin, 2015)
   https://doi.org/10.1109/CVPR.2015.7298682

3. **In Defense of the Triplet Loss for Person Re-Identification** (Hermans,
   Beyer, & Leibe, 2017) https://arxiv.org/abs/1703.07737

API reference
-------------

.. autoclass:: pyvisim.neural_networks.TripletNeuralNetwork
   :members:
   :inherited-members: Module
   :show-inheritance:

.. autoclass:: pyvisim.neural_networks.losses.TripletLoss
   :members:
   :show-inheritance:
