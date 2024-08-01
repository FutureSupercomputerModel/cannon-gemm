import math
import numpy as np
class Arch:
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
    pe_arr_dim = 200.0
    pe_arr_dim = 200.0
    pe_freq = 4.0 #PE frequency, GHz 
    eta = 1.0/(pe_arr_dim*pe_arr_dim*pe_freq) #ns to complete 1 multiply-add
    # leaf buffer
    buffer_size = 20.0*1024*1024 #20MB
    buffer_bw = 64.0 #element per ns


    def __init__(self, dram_bw:float, alpha:float, mesh_bw:float, mesh_dim:float, pe_arr_dim:float, pe_freq:float, buffer_size:float, buffer_bw:float) -> None:
        self.dram_bw = dram_bw
        self.alpha = alpha
        self.mesh_bw = mesh_bw
        self.beta = 1/mesh_bw
        self.mesh_dim = mesh_dim
        self.p = mesh_dim*mesh_dim
        self.pe_arr_dim = pe_arr_dim
        self.pe_freq = pe_freq
        self.eta = 1.0/(pe_arr_dim*pe_arr_dim*pe_freq)
        self.buffer_size = buffer_size
        self.buffer_bw = buffer_bw
        leaf_specified = False
        

    def get_max_leaf_gemm_size(self):
        min_problem_size_per_leaf = self.pe_arr_dim*self.pe_arr_dim*3
        scale_up_factor = math.floor(math.sqrt(self.buffer_size / min_problem_size_per_leaf))
        return (self.pe_arr_dim*scale_up_factor, self.pe_arr_dim*scale_up_factor, self.pe_arr_dim*scale_up_factor)

    def cannon_gemm(self, m,k,n):
        # pad m to be multiple of mesh_dim * pe_arr_dim
        m = math.ceil(m/(self.mesh_dim*self.pe_arr_dim)) * self.mesh_dim * self.pe_arr_dim
        # pad n to be multiple of mesh_dim * pe_arr_dim
        n = math.ceil(n/(self.mesh_dim*self.pe_arr_dim)) * self.mesh_dim * self.pe_arr_dim

        #tile m, n
        m_leaf = m/self.mesh_dim
        k_leaf = k/self.mesh_dim
        n_leaf = n/self.mesh_dim
        leaf_problem_size = m_leaf*k_leaf + k_leaf*n_leaf + m_leaf*n_leaf
        print(f"leaf problem: {m_leaf},{k_leaf},{n_leaf},size={leaf_problem_size}, leaf buffer size={self.buffer_size}")
        if leaf_problem_size > self.buffer_size:
            raise ValueError("Problem size exceeds buffer size")
        T_prep_A = self.alpha + max(self.beta*m*k/self.p, m*k/self.p/self.buffer_bw)#time to set up connection + max ( time for interconnect send, time for buffer receive)
        T_prep_B = self.alpha + max(self.beta*k*n/self.p, k*n/self.p/self.buffer_bw)
        T_prep = max(T_prep_A+T_prep_B, (m*k+k*n)/self.dram_bw )
        T_compute_pe = m*n*k * self.eta / self.p
        T_compute_buffer_ld = (m*k/self.p+k*n/self.p)/self.buffer_bw*math.sqrt(self.p)
        T_compute = max(T_compute_pe, T_compute_buffer_ld)
        if(T_compute_pe > T_compute_buffer_ld):
            print(f"compute: pe is the bottleneck, T_compute_pe={T_compute_pe}, T_compute_buffer_ls={T_compute_buffer_ld}")
        elif(T_compute_pe < T_compute_buffer_ld):
            print(f"compute: buffer is the bottleneck, T_compute_pe={T_compute_pe}, T_compute_buffer_ls={T_compute_buffer_ld}")
        T_send_A = self.mesh_dim*(self.alpha+max(self.beta*m*k/self.p, m*k/self.p/self.buffer_bw))
        T_send_B = self.mesh_dim*(self.alpha+max(self.beta*k*n/self.p, k*n/self.p/self.buffer_bw))
        T_send = T_send_A + T_send_B
        if(T_send_A > T_send_B):
            print(f"send: A is the bottleneck, T_send_A={T_send_A}, T_send_B={T_send_B}")
        elif(T_send_A < T_send_B):
            print(f"send: B is the bottleneck, T_send_A={T_send_A}, T_send_B={T_send_B}")
        T_store = self.alpha+max(m*n/self.dram_bw, m*n/self.p/self.buffer_bw)
        return (T_prep, T_compute, T_send, T_store)
    
    def cannon_gemm_tiled(self, m,k,n):
        (m_tile, k_tile, n_tile) = tuple(self.mesh_dim * np.array(self.get_max_leaf_gemm_size()))
        # (m_tile, k_tile, n_tile) = (m,k,n)
        iteration = math.ceil(m/m_tile)*math.ceil(k/k_tile)*math.ceil(n/n_tile)
        if m_tile>m and k_tile>k and n_tile>n: #no need to tile
            m_tile=m
            k_tile=k
            n_tile=n
        print(f"tiled problem: {m_tile}, {k_tile}, {n_tile}, on {iteration} iterations")
        (T_prep, T_compute, T_send, T_store)=self.cannon_gemm(m_tile, k_tile, n_tile)
        print(f"ns for each tiled problem: T_prep: {T_prep}, T_compute: {T_compute}, T_send: {T_send}, T_store: {T_store}")
        return [T_prep*iteration, T_compute*iteration, T_send*iteration, T_store*iteration]

    
    
    def print(self):
        print(f"dram_bw={self.dram_bw}, alpha={self.alpha}, mesh_bw={self.mesh_bw}, mesh_dim={self.mesh_dim}, mesh_dim={self.mesh_dim}, pe_arr_dim={self.pe_arr_dim}, pe_arr_dim={self.pe_arr_dim}, pe_freq={self.pe_freq}, buffer_size={self.buffer_size}")


cmos_arch = Arch(dram_bw=64.0*90.0*2, alpha=1.0, mesh_bw=64.0*4.0, mesh_dim=90.0, pe_arr_dim=200.0, pe_freq=4.0, buffer_size=20.0*1024*1024, buffer_bw=200*4.0*3)
# cmos_arch_ideal = arch.Arch(dram_bw=64.0*90.0*2, alpha=1.0, mesh_bw=64.0*3, mesh_H=90.0, mesh_W=90.0, pe_arr_H=200.0, pe_arr_W=200.0, pe_freq=4.0, buffer_size=20.0*1024*1024)
imec_arch = Arch(dram_bw=1000000, alpha=1.0, mesh_bw=200.0*30.0, mesh_dim=90.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3)

def top_level_gemm(m,k,n, arch: Arch):
    print("=====================================")
    arch.print()
    print(f"original problem: {m},{k},{n}")
    (T_prep, T_compute, T_send, T_store)=arch.cannon_gemm_tiled(m, k, n)
    print(f"ns for each problem: T_prep: {T_prep}, T_compute: {T_compute}, T_send: {T_send}, T_store: {T_store}")
    return [T_prep, T_compute, T_send, T_store]
    print("=====================================")