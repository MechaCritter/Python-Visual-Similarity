# arc42: eval

Software architecture of `pyvisim.eval`. This document is for developers and is
not part of the published documentation.

## Building block view

The module is a flat set of scoring functions with no state and no classes.

`top_k_map` and `top_k_accuracy` are written against the `EmbeddingStore`
protocol, so they do not depend on the concrete `InMemoryImageEmbeddingStore`.
See [the typing arc42](../typing/arc42.md) for that contract.
