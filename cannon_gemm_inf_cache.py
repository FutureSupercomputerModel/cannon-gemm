import math
import arch
def cannon_gemm(m,k,n, arch: arch.Arch):
    dram_bw = arch.dram_bw #DRAM bandwidth, element per cycle
    alpha = arch.alpha #time to set up connection
    beta = arch.beta #time to send 1 element

    mesh_H = arch.mesh_H
    mesh_W = arch.mesh_W
    p=arch.p #processor count
    pe_arr_H = arch.pe_arr_H
    pe_arr_W = arch.pe_arr_W
    eta = arch.eta #time to complete 1 multiply-add

    # pad m to be multiple of mesh_H * pe_arr_H
    m = math.ceil(m/(mesh_H*pe_arr_H)) * mesh_H * pe_arr_H
    # pad n to be multiple of mesh_W * pe_arr_W
    n = math.ceil(n/(mesh_W*pe_arr_W)) * mesh_W * pe_arr_W

    T_prep_A = alpha + beta*m*k/p
    T_prep_B = alpha + beta*k*n/p
    T_prep = max(T_prep_A+T_prep_B, (m*k+k*n)/dram_bw)
    T_compute = m*n*k * eta / p
    T_send_A = mesh_W*(alpha+beta*m*k/p)
    T_send_B = mesh_H*(alpha+beta*k*n/p)
    T_send = T_send_A + T_send_B
    T_store = m*n/dram_bw
    return (T_prep, T_compute, T_send, T_store)

