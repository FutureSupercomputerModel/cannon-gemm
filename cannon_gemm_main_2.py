import numpy as np
import arch_leaf_specified
from arch_leaf_specified import top_level_gemm
import math


gemm_sizes = [(90*200,90*200,90*200),
              (90*200*2,90*200*2,90*200*2),
              (90*200*4,90*200*4,90*200*4),
              (90*200*8,90*200*8,90*200*8),
              (90*200*16,90*200*16,90*200*16),
              (90*200*32,90*200*32,90*200*32),
              (90*200*64,90*200*64,90*200*64),
              ]

m,k,n = gemm_sizes[0]
top_level_gemm(m,k,n, arch_leaf_specified.example_arch)

