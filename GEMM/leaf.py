from cannon_gemm.GEMM.arch_base import Arch_base
from cannon_gemm.Leaf_Modeling.src.leaf_interface import run_leaf_modeling

# from GEMM.arch_base import Arch_base
# from Leaf_Modeling.src.leaf_interface import run_leaf_modeling

# from arch_base import Arch_base
# from cannon_gemm.Leaf_Modeling.src.leaf_interface import run_leaf_modeling

import math
class Leaf(Arch_base):
    pe_arr_dim = 200.0
    pe_freq = 5.6 #GHz
    buffer_size = 20.0*1024*1024 #20MB
    buffer_bw = 64.0 #element per ns
    nJ_per_mac = 0.38*1e-3
    bytes_per_element = 2

    min_gemm_size = pe_arr_dim

    def __init__(self, pe_arr_dim:float, pe_freq:float, buffer_size:float, buffer_bw:float, nJ_per_mac:float) -> None:
        self.pe_arr_dim = pe_arr_dim
        self.pe_freq = pe_freq
        self.buffer_size = buffer_size
        self.buffer_bw = buffer_bw
        self.nJ_per_mac = nJ_per_mac
        self.child_arch = None

    def print(self):
        print(f"pe_arr_dim: {self.pe_arr_dim}, pe_freq: {self.pe_freq}, buffer_size: {self.buffer_size}, buffer_bw: {self.buffer_bw}, min_gemm_size: {self.min_gemm_size}, max_gemm_size: {self.get_max_gemm_size()}")
    
    def get_gemm_latency_energy(self, M:int, K:int, N:int):
        M = math.ceil(M/(self.min_gemm_size)) * self.min_gemm_size
        K = math.ceil(K/(self.min_gemm_size)) * self.min_gemm_size
        N = math.ceil(N/(self.min_gemm_size)) * self.min_gemm_size

        # leaf_tech = 'cmos-gemm-7nm'
        # energy, cycles = run_leaf_modeling(leaf_tech, M, K, N)
        energy, cycles = run_leaf_modeling_fallback(self, M, K, N)
        leaf_time = cycles / self.pe_freq #ns
        energy = energy * 1e9 #nJ
        # print(f"Leaf energy: {energy}, Leaf time (ns): {leaf_time}")
        #report buffer usage
        print(f"buffer usage: {(M*K+K*N+M*N)*self.bytes_per_element}/{self.buffer_size}")
        assert M*K+K*N+M*N<=self.buffer_size
        # return max(M*K*N/self.pe_arr_dim/self.pe_arr_dim/self.pe_freq, M*K+K*N+M*N/self.buffer_bw), M*K*N*self.nJ_per_mac
        return leaf_time, energy
    
    def get_max_gemm_size(self):
        min_problem_size_per_leaf = self.min_gemm_size*self.min_gemm_size*3*self.bytes_per_element
        scale_up_factor = math.floor(math.sqrt(self.buffer_size / min_problem_size_per_leaf))
        return (self.min_gemm_size*scale_up_factor, self.min_gemm_size*scale_up_factor, self.min_gemm_size*scale_up_factor)
