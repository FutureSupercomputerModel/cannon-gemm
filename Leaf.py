from Arch_base import Arch_base
import math
class Leaf(Arch_base):
    pe_arr_dim = 200.0
    pe_freq = 30.0 #GHz
    buffer_size = 20.0*1024*1024 #20MB
    buffer_bw = 64.0 #element per ns

    matrix_block_dim_min = pe_arr_dim

    def __init__(self, pe_arr_dim:float, pe_freq:float, buffer_size:float, buffer_bw:float) -> None:
        self.pe_arr_dim = pe_arr_dim
        self.pe_freq = pe_freq
        self.buffer_size = buffer_size
        self.buffer_bw = buffer_bw
        self.child_arch = None

    def print(self):
        print(f"pe_arr_dim: {self.pe_arr_dim}, pe_freq: {self.pe_freq}, buffer_size: {self.buffer_size}, buffer_bw: {self.buffer_bw}, matrix_block_dim_min: {self.matrix_block_dim_min}")
    
    def get_gemm_latency(self, M:int, K:int, N:int):
        M = math.ceil(M/(self.matrix_block_dim_min)) * self.matrix_block_dim_min
        K = math.ceil(K/(self.matrix_block_dim_min)) * self.matrix_block_dim_min
        N = math.ceil(N/(self.matrix_block_dim_min)) * self.matrix_block_dim_min
        #report buffer usage
        print(f"buffer usage: {M*K+K*N+M*N}/{self.buffer_size}")
        # assert M*K+K*N+M*N<=self.buffer_size
        return max(M*K*N/self.pe_arr_dim/self.pe_arr_dim/self.pe_freq, M*K+K*N+M*N/self.buffer_bw)
    
    def get_max_gemm_size(self):
        min_problem_size_per_leaf = self.matrix_block_dim_min*self.matrix_block_dim_min*3
        scale_up_factor = math.floor(math.sqrt(self.buffer_size / min_problem_size_per_leaf))
        return (self.matrix_block_dim_min*scale_up_factor, self.matrix_block_dim_min*scale_up_factor, self.matrix_block_dim_min*scale_up_factor)
