import numpy as np
import Arch
from Arch import top_level_gemm
import deprecated.arch_leaf_specified as arch_leaf_specified
import math

# ideal_arch_200x200 = arch.Arch(dram_bw=64.0*90.0*2*10, alpha=1.0, mesh_bw=64.0 * 24 , mesh_H=90.0, mesh_W=90.0, pe_arr_H=200.0, pe_arr_W=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024)


# tiled problem constraints:
# if leaf problem is specified, m, k, n should be the multiple of mesh_dim*leaf_dim
# if leaf problem is not specified, m, k, n should be the multiple of mesh_dim*pe_arr_dim



gemm_sizes = [(90*200,90*200,90*200),
              (90*200*2,90*200*2,90*200*2),
              (90*200*4,90*200*4,90*200*4),
              (90*200*8,90*200*8,90*200*8),
              (90*200*16,90*200*16,90*200*16),
              (90*200*32,90*200*32,90*200*32),
              (90*200*64,90*200*64,90*200*64),
              ]
# top_level_gemm(m,k,n, cmos_arch)
# # top_level_gemm_leaf(m,k,n, imec_arch_leaf_specified)
gemm_times = []
for (m,k,n) in gemm_sizes:
    gemm_times.append(top_level_gemm(m,k,n, Arch.imec_arch))
print(gemm_times)

(m,k,n) = gemm_sizes[-1]
imec_archs = [
    Arch.Arch(dram_bw=7812.5, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=4.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    Arch.Arch(dram_bw=15625, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=6.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    Arch.Arch(dram_bw=31250, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=8.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    Arch.Arch(dram_bw=15625, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=11.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    Arch.Arch(dram_bw=31250, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=16.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    Arch.Arch(dram_bw=62500, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=23.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    Arch.Arch(dram_bw=125000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=32.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    Arch.Arch(dram_bw=250000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=45.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    Arch.Arch(dram_bw=500000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=64.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    Arch.Arch(dram_bw=1000000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=90.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3)
]
gemm_times_arch_study = []
for imec_arch in imec_archs:
    gemm_times_arch_study.append(Arch.top_level_gemm(m,k,n, imec_arch))
print(gemm_times_arch_study)