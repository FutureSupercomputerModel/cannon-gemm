import math

def cannon_gemm(M,K,N):
    dram_bw = 10.0 #DRAM bandwidth
    alpha = 1.0 #time to set up connection
    beta = 1.0 #time to send 1 element
    n=256.0 * 10 #width and height of matrices
    p=81.0 #processor count
    eta = 1.0 #time for 1 processor to complete 1 multiply-add

    T_prep = max(2*(alpha+beta*n**2/p), 2*n**2/dram_bw)
    T_compute = n**3 * eta / p
    T_send = 2*math.sqrt(p)*(alpha+beta*n**2/p)
    T_store = n**2/dram_bw
    T_all = T_prep + T_compute + T_send + T_store
    print("T_prep_per_iter: ", T_prep)
    print("T_compute_per_iter: ", T_compute)
    print("T_send_per_iter: ", T_send)
    print("T_all_per_iter: ", T_all)
    iteration = M/n * K/n * N/n
    print("iteration: ", iteration)
    T_prep = iteration * T_prep
    T_compute = iteration * T_compute
    T_send = iteration * T_send
    T_store = iteration * T_store
    T_all = iteration * T_all
    print("T_prep: ", T_prep)
    print("T_compute: ", T_compute)
    print("T_send: ", T_send)
    print("T_all: ", T_all)

    return(T_prep, T_compute, T_send, T_store, T_all)




