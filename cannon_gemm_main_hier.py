import numpy as np
from GEMM.arch import Arch
from GEMM.arch import top_level_gemm
from GEMM.leaf import Leaf
import math

hier_arch_leaf = Leaf(pe_arr_dim=200.0, pe_freq=1.0, buffer_size=20.0*1024*1024, buffer_bw=64.0, nJ_per_mac=1.0)
hier_arch_0 = Arch(buffer_size=8*1024*1024*1024, buffer_bw=64.0*90.0*2, alpha=1.0, mesh_bw=64.0*1.0, mesh_dim=10.0, nJ_per_Byte=1.0, child_arch=hier_arch_leaf)
hier_arch_top = Arch(buffer_size=8*1024*1024*1024*1024, buffer_bw=64.0*90.0*2, alpha=1.0, mesh_bw=64.0*1.0, mesh_dim=10.0, nJ_per_Byte=1.0, child_arch=hier_arch_0)

gemm_sizes = [(90*200,90*200,90*200),
              (90*200*2,90*200*2,90*200*2),
              (90*200*4,90*200*4,90*200*4),
              (90*200*8,90*200*8,90*200*8),
              (90*200*16,90*200*16,90*200*16),
              (90*200*32,90*200*32,90*200*32),
              (90*200*64,90*200*64,90*200*64),
              ]

m,k,n = (90*200*64,90*200*16,90*200)
top_level_gemm(m,k,n, hier_arch_top)

