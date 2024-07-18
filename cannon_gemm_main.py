import numpy as np
import arch
import arch_leaf_specified
import math
cmos_arch = arch.Arch(dram_bw=64.0*90.0*2, alpha=1.0, mesh_bw=64.0*4.0, mesh_dim=90.0, pe_arr_dim=200.0, pe_freq=4.0, buffer_size=20.0*1024*1024, buffer_bw=200*4.0*3)
# cmos_arch_ideal = arch.Arch(dram_bw=64.0*90.0*2, alpha=1.0, mesh_bw=64.0*3, mesh_H=90.0, mesh_W=90.0, pe_arr_H=200.0, pe_arr_W=200.0, pe_freq=4.0, buffer_size=20.0*1024*1024)
imec_arch = arch.Arch(dram_bw=1000000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=90.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3)
# ideal_arch_200x200 = arch.Arch(dram_bw=64.0*90.0*2*10, alpha=1.0, mesh_bw=64.0 * 24 , mesh_H=90.0, mesh_W=90.0, pe_arr_H=200.0, pe_arr_W=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024)
imec_arch_leaf_specified = arch_leaf_specified.Arch_leaf_specified(dram_bw=1000000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=90.0, leaf_dim=1869, leaf_time=1869*1869*1869/200.0/200.0/30.0)

# tiled problem constraints:
# if leaf problem is specified, m, k, n should be the multiple of mesh_dim*leaf_dim
# if leaf problem is not specified, m, k, n should be the multiple of mesh_dim*pe_arr_dim
def top_level_gemm(m,k,n, arch: arch.Arch):
    print("=====================================")
    arch.print()
    print(f"original problem: {m},{k},{n}")
    (m_tile, k_tile, n_tile) = tuple(arch.mesh_dim * np.array(arch.get_max_leaf_gemm_size()))
    # (m_tile, k_tile, n_tile) = (m,k,n)
    iteration = math.ceil(m/m_tile)*math.ceil(k/k_tile)*math.ceil(n/n_tile)
    if m_tile>m and k_tile>k and n_tile>n: #no need to tile
        m_tile=m
        k_tile=k
        n_tile=n
    print(f"tiled problem: {m_tile}, {k_tile}, {n_tile}, on {iteration} iterations")
    (T_prep, T_compute, T_send, T_store)=arch.cannon_gemm(m_tile, k_tile, n_tile)
    print(f"ns for each tiled problem: T_prep: {T_prep}, T_compute: {T_compute}, T_send: {T_send}, T_store: {T_store}")
    print(f"ns for original problem: T_prep: {T_prep*iteration}, T_compute: {T_compute*iteration}, T_send: {T_send*iteration}, T_store: {T_store*iteration}")
    return [T_prep*iteration, T_compute*iteration, T_send*iteration, T_store*iteration]
    print("=====================================")

def top_level_gemm_leaf(m,k,n, arch: arch_leaf_specified.Arch_leaf_specified):
    print("=====================================")
    arch.print()
    print(f"original problem: {m},{k},{n}")
    (m_tile, k_tile, n_tile) = (m,k,n)
    iteration = m/m_tile*k/k_tile*n/n_tile
    print(f"tiled problem: {m_tile}, {k_tile}, {n_tile}, on {iteration} iterations")
    (T_prep, T_compute, T_send, T_store)=arch.cannon_gemm(m_tile, k_tile, n_tile)
    print(f"ns for each tiled problem: T_prep: {T_prep}, T_compute: {T_compute}, T_send: {T_send}, T_store: {T_store}")
    print(f"ns for original problem: T_prep: {T_prep*iteration}, T_compute: {T_compute*iteration}, T_send: {T_send*iteration}, T_store: {T_store*iteration}")
    return [T_prep*iteration, T_compute*iteration, T_send*iteration, T_store*iteration]
    print("=====================================")

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
# for (m,k,n) in gemm_sizes:
#     gemm_times.append(top_level_gemm(m,k,n, imec_arch))
# print(gemm_times)

(m,k,n) = gemm_sizes[-1]
imec_archs = [
    arch.Arch(dram_bw=7812.5, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=4.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    arch.Arch(dram_bw=15625, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=6.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    arch.Arch(dram_bw=31250, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=8.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    arch.Arch(dram_bw=15625, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=11.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    arch.Arch(dram_bw=31250, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=16.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    arch.Arch(dram_bw=62500, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=23.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    arch.Arch(dram_bw=125000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=32.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    arch.Arch(dram_bw=250000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=45.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    arch.Arch(dram_bw=500000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=64.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3),
    arch.Arch(dram_bw=1000000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=90.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3)
]
gemm_times_arch_study = []
for imec_arch in imec_archs:
    gemm_times_arch_study.append(top_level_gemm(m,k,n, imec_arch))
print(gemm_times_arch_study)