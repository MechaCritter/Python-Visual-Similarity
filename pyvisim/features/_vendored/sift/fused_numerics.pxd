cimport numpy as cnp

ctypedef fused np_floats:
    cnp.float32_t
    cnp.float64_t
