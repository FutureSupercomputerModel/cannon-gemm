import math
from myMath import ceildiv
class Arch_leaf_specified:
    #DRAM
    dram_bw = 64.0 #DRAM bandwidth, element per ns
    #interconnect
    alpha = 1.0 #ns to set up connection
    mesh_bw = 64.0 #mesh bandwidth, elements per ns
    beta = 1.0/mesh_bw #ns to send 1 element
    mesh_dim = 90.0
    mesh_dim = 90.0
    p=mesh_dim*mesh_dim #processor count
    # leaf
    
    # leaf buffer

    # leaf info
    leaf_M_min = 200
    leaf_K_min = 200
    leaf_N_min = 200
    leaf_M = 1869
    leaf_K = 1869
    leaf_N = 1869
    # leaf_time = 1869*1869*1869/200.0/200.0/4.0

    leaf_M_max = 1869
    leaf_K_max = 1869
    leaf_N_max = 1869

    def __init__(self, dram_bw:float, alpha:float, mesh_bw:float, mesh_dim:float, leaf_m_min:float, leaf_k_min:float, leaf_n_mean:float) -> None:
        self.dram_bw = dram_bw
        self.alpha = alpha
        self.mesh_bw = mesh_bw
        self.beta = 1/mesh_bw
        self.mesh_dim = mesh_dim
        self.p = mesh_dim*mesh_dim
        leaf_specified = True
        self.leaf_M_min = leaf_m_min
        self.leaf_K_min = leaf_k_min
        self.leaf_N_min = leaf_n_mean
        
 
    def cannon_gemm(self, m,k,n):
        # pad m to be multiple of mesh_dim * pe_arr_dim
        m = math.ceil(m/(self.mesh_dim*self.leaf_M)) * self.mesh_dim * self.leaf_M
        # pad n to be multiple of mesh_dim * pe_arr_dim
        n = math.ceil(n/(self.mesh_dim*self.leaf_N)) * self.mesh_dim * self.leaf_N
        k = math.ceil(k/(self.mesh_dim*self.leaf_K)) * self.mesh_dim * self.leaf_K
        # print(f"padded problem: {m},{k},{n}")
        #tile m, n
        m_leaf = self.leaf_M
        k_leaf = self.leaf_K
        n_leaf = self.leaf_N
        leaf_problem_size = m_leaf*k_leaf + k_leaf*n_leaf + m_leaf*n_leaf
        #insert timeloop interface here to get leaf_time:
        self.leaf_time = m_leaf*k_leaf*n_leaf/200.0/200.0/4.0
        print(f"leaf problem: {m_leaf},{k_leaf},{n_leaf},size={leaf_problem_size}")
        T_prep_A = self.alpha + self.beta*m*k/self.p#time to set up connection + max ( time for interconnect send, time for buffer receive)
        T_prep_B = self.alpha + self.beta*k*n/self.p
        T_prep = max(T_prep_A+T_prep_B, (m*k+k*n)/self.dram_bw )
        T_compute = self.leaf_time*self.mesh_dim
        T_send_A = self.mesh_dim*(self.alpha+self.beta*m*k/self.p)
        T_send_B = self.mesh_dim*(self.alpha+self.beta*k*n/self.p)
        T_send = T_send_A + T_send_B
        if(T_send_A > T_send_B):
            print(f"send: A is the bottleneck, T_send_A={T_send_A}, T_send_B={T_send_B}")
        elif(T_send_A < T_send_B):
            print(f"send: B is the bottleneck, T_send_A={T_send_A}, T_send_B={T_send_B}")
        T_store = self.alpha+m*n/self.dram_bw
        return (T_prep, T_compute, T_send, T_store)

    
    def cannon_gemm_tiled(self, m,k,n):
        # we don't want the tiled problem to be much larger than the original problem.
        # also we don't want the leaf problem in the tiled problem to be larger than the leaf cache.
        max_multiple_of_leaf_dim = max(ceildiv(m, self.mesh_dim*self.leaf_M_min), ceildiv(k, self.mesh_dim*self.leaf_K_min), ceildiv(n, self.mesh_dim*self.leaf_N_min), self.leaf_M_max/self.leaf_K_min, self.leaf_K_max/self.leaf_K_min, self.leaf_N_max/self.leaf_N_min)
        (m_tile_optimal, k_tile_optimal, n_tile_optimal) = (0,0,0)
        runtime_optimal = float('inf')
        T_prep,T_compute,T_send,T_store = float('inf'),float('inf'),float('inf'),float('inf')
        for multiple_of_leaf_dim in range(1,int(max_multiple_of_leaf_dim)+1):
            (self.leaf_M, self.leaf_K, self.leaf_N) = (self.leaf_M_min*multiple_of_leaf_dim, self.leaf_K_min*multiple_of_leaf_dim, self.leaf_N_min*multiple_of_leaf_dim)
            (m_tile, k_tile, n_tile) = (self.mesh_dim*self.leaf_M, self.mesh_dim*self.leaf_K, self.mesh_dim*self.leaf_N)
            iteration = ceildiv(m,m_tile)*ceildiv(k,k_tile)*ceildiv(n,n_tile)
            print(f"tiled problem: {m_tile}, {k_tile}, {n_tile}, on {iteration} iterations")
            (T_prep, T_compute, T_send, T_store)=self.cannon_gemm(m_tile, k_tile, n_tile)
            
            runtime = max(T_prep*iteration,T_compute*iteration,T_send*iteration,T_store*iteration)
            print(f"runtime: T_total: {runtime}, T_prep: {T_prep*iteration}, T_compute: {T_compute*iteration}, T_send: {T_send*iteration}, T_store: {T_store*iteration}")
            if runtime < runtime_optimal:
                runtime_optimal = runtime
                (m_tile_optimal, k_tile_optimal, n_tile_optimal) = (m_tile, k_tile, n_tile)
                T_prep_optimal, T_compute_optimal, T_send_optimal, T_store_optimal = (T_prep*iteration,T_compute*iteration,T_send*iteration,T_store*iteration)
        return [T_prep_optimal, T_compute_optimal, T_send_optimal, T_store_optimal]
    
    def print(self):
        print(f"dram_bw={self.dram_bw}, alpha={self.alpha}, mesh_bw={self.mesh_bw}, mesh_dim={self.mesh_dim}, mesh_dim={self.mesh_dim}, leaf_M_min={self.leaf_M_min}, leaf_K_min={self.leaf_K_min}, leaf_N_min={self.leaf_N_min}")

imec_arch = Arch_leaf_specified(dram_bw=1000000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=90.0, leaf_m_min=200.0, leaf_k_min=200.0, leaf_n_mean=200.0)
example_arch = Arch_leaf_specified(dram_bw=1000, alpha=1.0, mesh_bw=64.0*4.0, mesh_dim=4.0, leaf_m_min=200.0, leaf_k_min=200.0, leaf_n_mean=200.0)

def top_level_gemm(m,k,n, arch: Arch_leaf_specified):
    print("=====================================")
    arch.print()
    print(f"original problem: {m},{k},{n}")
    (T_prep, T_compute, T_send, T_store)=arch.cannon_gemm_tiled(m, k, n)
    print(f"ns for each problem: T_prep: {T_prep}, T_compute: {T_compute}, T_send: {T_send}, T_store: {T_store}")
    return [T_prep, T_compute, T_send, T_store]
    print("=====================================")
