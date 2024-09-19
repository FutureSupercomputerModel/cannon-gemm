import numpy as np
import arch_lib
from GEMM.arch import top_level_gemm


gemm_sizes = [(90*200,90*200,90*200),
              (90*200*2,90*200*2,90*200*2),
              (90*200*4,90*200*4,90*200*4),
              (90*200*8,90*200*8,90*200*8),
              (90*200*16,90*200*16,90*200*16),
              (90*200*32,90*200*32,90*200*32),
              (90*200*64,90*200*64,90*200*64),
              ]

gemm_sizes = [
    (37800.0, 37800.0, 37800.0),
    (28350, 28350, 28350),
    (18900, 18900, 18900),
    (14175, 14175, 14175),
    (9450, 9450, 9450),
    (4725, 4725, 4725),
    (3600, 3600, 3600),
    (2400, 2400, 2400),
    (1800, 1800, 1800),
    (900, 900, 900)
]
# m,k,n = (1800*2,1800*2,1800*2)
# m,k,n = (37800.0, 37800.0, 37800.0)
for m,k,n in gemm_sizes:
    top_level_gemm(m,k,n, arch_lib.hier_arch_blade_1leaf, debug=True, general_tiling=True)

