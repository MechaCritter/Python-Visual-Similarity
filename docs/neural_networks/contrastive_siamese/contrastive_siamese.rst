ContrastiveSiameseNetwork
=========================

This network "learns" the similarity metric directly. Two images are passed
through the same shared-weight ``backbone`` and projection ``head`` to
produce embeddings, which are L2-normalized so that cosine similarity
reduces to a dot product. The network is trained so that similar images map
to nearby embeddings and dissimilar images map far apart. Following diagram
visualizes this:

.. code-block:: text

                     ┌──────────┐    ┌────────────────┐    ┌──────────────┐
   Input Image A ───►│ Backbone │───►│ Embedding Head │───►│ L2 Normalize │───► Embedding A
                     └──────────┘    └────────────────┘    └──────────────┘
                          ╎                  ╎                    ╎
                          ╎ Shared Weights   ╎                    ╎
                          ╎                  ╎                    ╎
                     ┌──────────┐    ┌────────────────┐    ┌──────────────┐
   Input Image B ───►│ Backbone │───►│ Embedding Head │───►│ L2 Normalize │───► Embedding B
                     └──────────┘    └────────────────┘    └──────────────┘

                          Embedding A + Embedding B
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Contrastive Loss (training) / fixed metric, e.g. cosine (inference) │
   └─────────────────────────────────────────────────────────────────────┘

`Contrastive loss` is used to train this network, which has the formula:

.. math::

   L = \frac{1}{2N} \sum_{i=1}^{N} \Bigl( y_i \, D_i^2 + (1 - y_i) \, \max(0, m - D_i)^2 \Bigr)

Example: training a Contrastive Siamese Network
-----------------------------------------------

See :doc:`this tutorial </tutorials/notebooks/siamese_network>`.

API reference
-------------

.. autoclass:: pyvisim.neural_networks.ContrastiveSiameseNetwork
   :members:
   :inherited-members: Module
   :show-inheritance:
