import numpy as np
import arch_lib
from GEMM.arch import top_level_gemm



m,k,n = (400,400,400)
# # m,k,n = (37800.0, 37800.0, 37800.0)
# m,k,n = 90*200*64,90*200*64,90*200*64
top_level_gemm(m,k,n, arch_lib.leaf_imec, debug=True, general_tiling=True)

