Handcrafted features
====================

Includes extractors that compute descriptors with a fixed algorithm, and
``Lambda`` for a custom function.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Object
     - ``output_dim``
     - Notes
   * - :doc:`SIFT <sift/sift>`
     - 128
     - SIFT descriptors
   * - :doc:`RootSIFT <rootsift/rootsift>`
     - 128
     - SIFT with Hellinger normalization (default extractor)
   * - :doc:`Lambda <lambda/lambda>`
     - user-defined
     - wraps any custom function

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   sift/sift
   rootsift/rootsift
   lambda/lambda
