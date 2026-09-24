Image Store
===========

An image store holds a gallery of images together with their embeddings.
Calling ``build_store()`` embeds every gallery image with the given embedder
and places the embeddings in a search index, which allows fast nearest neighbor
search algorithms instead of brute-force search.

Once the store is built, one can take a query image and retrieve the top-k most
similar images to it from the gallery.

To refine the query itself before retrieval, one can enable **alpha-weighted
query expansion** (αQE). See :ref:`query-expansion` for more information.

To refine the final candidates after retrieval, one can use a **reranker**. See
:doc:`Reranking Documentation <../reranking/reranking>` for more information.

Example
-------

.. tip::
   See :doc:`this tutorial </tutorials/notebooks/image_search>` for a
   more detailed walkthrough.

.. code-block:: python

   import numpy as np
   from PIL import Image
   from pyvisim.neural_networks import ClipEmbedder
   from pyvisim.image_store import InMemoryImageEmbeddingStore

   embedder = ClipEmbedder("ViT-B/32")
   store = InMemoryImageEmbeddingStore(
       image_paths=["image1.jpg", "image2.jpg", "image3.jpg"],
       embedder=embedder,
       search_index="hnsw",
       space="cosine",
       index_params={"graph_degree": 32, "search_candidates": 100},
   )
   # Embed the gallery and build the index
   store.build_store()

   # Save the store to disk and load it back later
   store.save_to_disk("gallery.safetensors")

   # Retrieve top-k most similar images to a query image
   query_image = np.array(Image.open("query.jpg"))
   results = store.retrieve_top_k_similar(query_image, k=5)[0]

   for candidate in results:
       print(candidate.path, candidate.score)

   # Refine the query with its own neighborhood first (alpha query expansion)
   results = store.retrieve_top_k_similar(
       query_image, k=5, query_expansion=True, expansion_alpha=3.0
   )[0]

API reference
-------------

.. autoclass:: pyvisim.image_store.InMemoryImageEmbeddingStore
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: pyvisim.image_store.Candidate
   :members:

Building the store
------------------

Constructing an ``InMemoryImageEmbeddingStore`` only records the gallery paths.
``build_store()`` embeds the images and builds the index, so call it right
after the constructor:

.. code-block:: python

   store = InMemoryImageEmbeddingStore(image_paths=paths, embedder=embedder)
   store.build_store()

An unbuilt store holds no vectors. ``search``, ``retrieve_top_k_similar``,
``embeddings``, ``embeddings_of``, ``index``, ``dim``, ``to_dict`` and
``save_to_disk`` raise ``RuntimeError`` until the build has run. ``paths``,
``len(store)`` and ``in`` answer from the recorded paths and work either way,
and ``is_built`` tells whether the build has run. A second call to
``build_store()`` does nothing.

To have the constructor build the store itself, pass ``lazy_build=False``:

.. code-block:: python

   store = InMemoryImageEmbeddingStore(
       image_paths=paths, embedder=embedder, lazy_build=False
   )

A store that adopted an ``ExternalSearchIndex`` and a store read back by
``load_from_disk`` already hold their embeddings, so both come back built.

Retrieval
---------

After having built the image store, ``retrieve_top_k_similar`` embeds the query
images with the store's embedder and ranks the gallery through the index. One
ranked list of ``Candidate`` matches comes back per query image. Each candidate
carries the ``path`` of the gallery image, the ``score`` it was ranked by, which
is a distance for the built-in indexes, so lower means more similar, and
``array``, the matched image itself as an RGB ``uint8`` array. The image is read
from ``path`` the first time ``array`` is accessed and kept from then on, which
saves memory. ``clear_buffer()`` drops the kept image again.

.. code-block:: python

   candidates = store.retrieve_top_k_similar(query_image, k=5)[0]
   for candidate in candidates:
       print(candidate.path, candidate.score)

   best_match = candidates[0].array   # (H, W, 3) uint8, read from disk now
   candidates[0].clear_buffer()       # forget it again

.. _query-expansion:

Query expansion
~~~~~~~~~~~~~~~

``retrieve_top_k_similar`` can refine every query with the **alpha-weighted
query expansion** (αQE) of Radenović et al. [1] before the final search, which
has been shown to **substantially improve** ``mean Average Precision`` (mAP).

When enabled, the query is searched once, the embeddings of its
``expansion_neighbors`` best matches are read back from the index, and the
query is replaced by the L2-normalised weighted average of itself and those
matches, where each match weighs its cosine similarity to the query raised to
``expansion_alpha``, and a match whose similarity is not positive weighs
nothing. The results are thereby pulled towards the whole neighborhood the
query belongs to instead of the single point that was embedded. With
``expansion_alpha=0`` every match that resembles the query weighs the same,
which is the classic average query expansion (AQE).

.. code-block:: python

   candidates = store.retrieve_top_k_similar(
       query_image,
       k=5,
       query_expansion=True,
       expansion_alpha=3.0,
       expansion_neighbors=50,
   )[0]

The expansion is off by default: it costs one extra index search per query plus
the decoding of ``expansion_neighbors`` gallery vectors. The defaults
``expansion_alpha=3`` and ``expansion_neighbors=50`` are the values used in
the original paper. It also defines the expansion on L2-normalised embeddings
ranked by cosine similarity.

.. important::

   The weights are always computed on normalised copies of the vectors. Hence,
   make sure that the embeddings are L2-normalised before building the index!

Serialization
-------------

To save the store to disk:

.. code-block:: python

   store.save_to_disk("gallery.safetensors")

References
----------

[1] F. Radenović, G. Tolias, and O. Chum, "Fine-tuning CNN Image Retrieval with
No Human Annotation," IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 41, no. 7, pp. 1655-1668, 2019.