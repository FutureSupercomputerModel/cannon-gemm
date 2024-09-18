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

m,k,n = (1800*2,1800*2,1800*2)
# m,k,n = (37800.0, 37800.0, 37800.0)
top_level_gemm(m,k,n, arch_lib.hier_arch_blade, debug=True, general_tiling=True)

