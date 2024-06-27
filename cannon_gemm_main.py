
import arch
cmos_arch = arch.Arch(dram_bw=64.0*90.0*2, alpha=1.0, mesh_bw=64.0, mesh_H=90.0, mesh_W=90.0, pe_arr_H=200.0, pe_arr_W=200.0, pe_freq=4.0, buffer_size=20.0*1024*1024)
cmos_arch_ideal = arch.Arch(dram_bw=64.0*90.0*2, alpha=1.0, mesh_bw=64.0*3, mesh_H=90.0, mesh_W=90.0, pe_arr_H=200.0, pe_arr_W=200.0, pe_freq=4.0, buffer_size=20.0*1024*1024)
imec_arch = arch.Arch(dram_bw=64.0*90.0*2, alpha=1.0, mesh_bw=64.0, mesh_H=90.0, mesh_W=90.0, pe_arr_H=200.0, pe_arr_W=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024)
ideal_arch_200x200 = arch.Arch(dram_bw=64.0*90.0*2*10, alpha=1.0, mesh_bw=64.0 * 24 , mesh_H=90.0, mesh_W=90.0, pe_arr_H=200.0, pe_arr_W=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024)
(m,k,n)=(90*200*8,90*200*8,90*200*8)

def top_level_gemm(m,k,n, arch: arch.Arch):
    print("=====================================")
    arch.print()
    print(f"original problem: {m},{k},{n}")
    (m_tile, k_tile, n_tile) = (m,k,n)
    iteration = m/m_tile*k/k_tile*n/n_tile
    print(f"tiled problem: {m_tile}, {k_tile}, {n_tile}, on {iteration} iterations")
    (T_prep, T_compute, T_send, T_store)=arch.cannon_gemm(m_tile, k_tile, n_tile)
    print(f"ns for each tiled problem: T_prep: {T_prep}, T_compute: {T_compute}, T_send: {T_send}, T_store: {T_store}")
    print(f"ns for original problem: T_prep: {T_prep*iteration}, T_compute: {T_compute*iteration}, T_send: {T_send*iteration}, T_store: {T_store*iteration}")
    print("=====================================")


top_level_gemm(m,k,n, cmos_arch)
top_level_gemm(m,k,n, cmos_arch_ideal)
top_level_gemm(m,k,n, imec_arch)
top_level_gemm(m,k,n, ideal_arch_200x200)
# (T_prep, T_compute, T_send, T_store)=cannon_gemm.cannon_gemm(m, k, n, imec_arch)
# print(T_prep, T_compute, T_send, T_store)