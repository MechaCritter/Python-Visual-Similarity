import numpy as np
import numpy.typing as npt

def ssd_matrix(
    flat1: npt.NDArray[np.uint8],
    flat2: npt.NDArray[np.uint8],
    num_threads: int,
) -> npt.NDArray[np.int64]: ...
