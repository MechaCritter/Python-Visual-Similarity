# Distance

File: [`pyvisim/distance.py`](../pyvisim/distance.py)

Implementations of the pairwise metrics
used to compare image embeddings.

The functions take two 2-D matrices, `x` of shape `(N, D)` and `y` of shape
`(M, D)`, and return the full `(N, M)` pairwise result as float64:

- `cosine_similarity(x, y)`: higher means more similar. All-zero rows get a
  similarity of 0.
- `euclidean_distances(x, y)`: lower means more similar.
- `manhattan_distances(x, y, working_memory_bytes=...)`: lower means more
  similar. The optional keyword caps the size of the internal broadcast
  temporary (default 256 MiB).

```python
import numpy as np
from pyvisim.distance import cosine_similarity, euclidean_distances

x = np.random.rand(4, 128)   # 4 embeddings
y = np.random.rand(6, 128)   # 6 embeddings

cosine_similarity(x, y).shape      # (4, 6)
euclidean_distances(x, y).shape    # (4, 6)
```

## Formula

1. **Cosine similarity**:
$$
\text{cosine\_similarity}(x, y) = \frac{x \cdot y^T}{||x||_2 ||y||_2}
$$

2. **Euclidean distance**:
$$
\text{euclidean\_distances}(x, y) = ||x - y||_2
$$

3. **Manhattan distance**:
$$
\text{manhattan\_distances}(x, y) = ||x - y||_1
$$
