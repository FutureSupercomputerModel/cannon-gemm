import math
class Arch:
    #DRAM
    dram_bw = 64.0 #DRAM bandwidth, element per ns
    #interconnect
    alpha = 1.0 #ns to set up connection
    mesh_bw = 64.0 #mesh bandwidth, elements per ns
    beta = 1.0/mesh_bw #ns to send 1 element
    mesh_H = 90.0
    mesh_W = 90.0
    p=mesh_H*mesh_W #processor count
    # leaf
    pe_arr_H = 200.0
    pe_arr_W = 200.0
    pe_freq = 4.0 #PE frequency, GHz 
    eta = 1.0/(pe_arr_H*pe_arr_W*pe_freq) #ns to complete 1 multiply-add
    # leaf buffer
    buffer_size = 20.0*1024*1024 #20MB

    def __init__(self, dram_bw:float, alpha:float, mesh_bw:float, mesh_H:float, mesh_W:float, pe_arr_H:float, pe_arr_W:float, pe_freq:float, buffer_size:float):
        self.dram_bw = dram_bw
        self.alpha = alpha
        self.mesh_bw = mesh_bw
        self.beta = 1/mesh_bw
        self.mesh_H = mesh_H
        self.mesh_W = mesh_W
        self.p = mesh_H*mesh_W
        self.pe_arr_H = pe_arr_H
        self.pe_arr_W = pe_arr_W
        self.pe_freq = pe_freq
        self.eta = 1.0/(pe_arr_H*pe_arr_W*pe_freq)
        self.buffer_size = buffer_size

    def get_max_leaf_gemm_size(self):
        min_problem_size_per_leaf = self.pe_arr_H*self.pe_arr_W*3
        scale_up_factor = math.floor(self.buffer_size / min_problem_size_per_leaf)
        return (self.pe_arr_H*scale_up_factor, self.pe_arr_W*scale_up_factor, self.pe_arr_W*scale_up_factor)
    
    def cannon_gemm(self, m,k,n):

        # pad m to be multiple of mesh_H * pe_arr_H
        m = math.ceil(m/(self.mesh_H*self.pe_arr_H)) * self.mesh_H * self.pe_arr_H
        # pad n to be multiple of mesh_W * pe_arr_W
        n = math.ceil(n/(self.mesh_W*self.pe_arr_W)) * self.mesh_W * self.pe_arr_W

        #tile m, n
        m_leaf = m/self.mesh_H
        k_leaf = k/self.mesh_H
        n_leaf = n/self.mesh_W
        leaf_problem_size = m_leaf*k_leaf + k_leaf*n_leaf + m_leaf*n_leaf
        print(f"leaf problem: {m_leaf},{k_leaf},{n_leaf},size={leaf_problem_size}, leaf buffer size={self.buffer_size}")
        if leaf_problem_size > self.buffer_size:
            raise ValueError("Problem size exceeds buffer size")
        T_prep_A = self.alpha + self.beta*m*k/self.p
        T_prep_B = self.alpha + self.beta*k*n/self.p
        T_prep = max(T_prep_A+T_prep_B, (m*k+k*n)/self.dram_bw)
        T_compute = m*n*k * self.eta / self.p
        T_send_A = self.mesh_W*(self.alpha+self.beta*m*k/self.p)
        T_send_B = self.mesh_H*(self.alpha+self.beta*k*n/self.p)
        T_send = T_send_A + T_send_B
        T_store = m*n/self.dram_bw
        return (T_prep, T_compute, T_send, T_store)
    
    def print(self):
        print(f"dram_bw={self.dram_bw}, alpha={self.alpha}, mesh_bw={self.mesh_bw}, mesh_H={self.mesh_H}, mesh_W={self.mesh_W}, pe_arr_H={self.pe_arr_H}, pe_arr_W={self.pe_arr_W}, pe_freq={self.pe_freq}, buffer_size={self.buffer_size}")