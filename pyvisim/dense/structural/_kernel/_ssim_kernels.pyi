import numpy as np
import numpy.typing as npt

def ssim_plane_sums(
    planes1: npt.NDArray[np.float32],
    planes2: npt.NDArray[np.float32],
    kernel: npt.NDArray[np.float32],
    c1: float,
    c2: float,
    num_threads: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
