import math
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
    leaf_M = 1869
    leaf_K = 1869
    leaf_N = 1869
    leaf_time = 1869*1869*1869/200.0/200.0/4.0

    def __init__(self, dram_bw:float, alpha:float, mesh_bw:float, mesh_dim:float, leaf_dim:float, leaf_time:float) -> None:
        self.dram_bw = dram_bw
        self.alpha = alpha
        self.mesh_bw = mesh_bw
        self.beta = 1/mesh_bw
        self.mesh_dim = mesh_dim
        self.p = mesh_dim*mesh_dim
        leaf_specified = True
        self.leaf_M = leaf_dim
        self.leaf_K = leaf_dim
        self.leaf_N = leaf_dim
        self.leaf_time = leaf_time
        
 
    def cannon_gemm(self, m,k,n):
        # pad m to be multiple of mesh_dim * pe_arr_dim
        m = math.ceil(m/(self.mesh_dim*self.leaf_M)) * self.mesh_dim * self.leaf_M
        # pad n to be multiple of mesh_dim * pe_arr_dim
        n = math.ceil(n/(self.mesh_dim*self.leaf_N)) * self.mesh_dim * self.leaf_N
        k = math.ceil(k/(self.mesh_dim*self.leaf_K)) * self.mesh_dim * self.leaf_K
        print(f"padded problem: {m},{k},{n}")
        #tile m, n
        m_leaf = self.leaf_M
        k_leaf = self.leaf_K
        n_leaf = self.leaf_N
        leaf_problem_size = m_leaf*k_leaf + k_leaf*n_leaf + m_leaf*n_leaf
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

    
    def print(self):
        print(f"dram_bw={self.dram_bw}, alpha={self.alpha}, mesh_bw={self.mesh_bw}, mesh_dim={self.mesh_dim}, mesh_dim={self.mesh_dim}, leaf_M={self.leaf_M}, leaf_K={self.leaf_K}, leaf_N={self.leaf_N}, leaf_time={self.leaf_time}")