from GEMM.arch_base import Arch_base
from Leaf_Modeling.src.leaf_interface import run_leaf_modeling
import math
from helper.myMath import bytes2str, str2bytes, str2GBps
class Leaf(Arch_base):
    # pe_arr_dim = 200.0
    # buffer_size = 20.0*1024*1024 #20MB
    # buffer_width = 64.0 #element per ns
    
    # pe_freq = 4.0 #GHz
    # nJ_per_mac = 0.38
    # buffer_freq = 4.0
    # interconnect_nJ_per_bit = 0.1
    # buffer_nJ_per_bit = 500
    # bytes_per_element = 2

    # min_gemm_size = pe_arr_dim
    # buffer_bw = buffer_width*buffer_freq/8.0 #bytes per ns

    def __init__(self, pe_arr_dim:float, buffer_size:str, buffer_bw:str, pe_freq:float, nJ_per_mac:float, interconnect_nJ_per_bit:float, buffer_nJ_per_bit:float, bytes_per_element:int) -> None:
        self.pe_arr_dim = pe_arr_dim
        self.buffer_size_bytes = str2bytes(buffer_size)
        self.buffer_size_elems = self.buffer_size_bytes/bytes_per_element
        self.buffer_bw_GBps = str2GBps(buffer_bw)
        self.buffer_bw = self.buffer_bw_GBps/bytes_per_element #elements per ns
        self.pe_freq = pe_freq
        self.nJ_per_mac = nJ_per_mac
        self.interconnect_nJ_per_bit = interconnect_nJ_per_bit
        self.buffer_nJ_per_bit = buffer_nJ_per_bit
        self.bytes_per_element = bytes_per_element


        self.min_gemm_size = self.pe_arr_dim
        self.child_arch = None
        self.level = 0

    def print(self):
        self.debugprint(f"pe_arr_dim: {self.pe_arr_dim}, buffer_size: {bytes2str(self.buffer_size_bytes)}, buffer_bw: {self.buffer_bw_GBps}GBps,"
                f"pe_freq: {self.pe_freq}GHz,  nJ_per_mac: {self.nJ_per_mac}nJ, "
                f"interconnect_nJ_per_bit: {self.interconnect_nJ_per_bit}nJ, buffer_nJ_per_bit: {self.buffer_nJ_per_bit}nJ, min_gemm_size: {self.min_gemm_size}, precision: {self.bytes_per_element*8},"
                f"max_gemm_size: {self.get_max_gemm_size()}")
    
    #energy in nJ, time in ns
    def run_leaf_modeling_fallback(self, M, K, N, debug=False):
        compute_time = M*K*N/self.pe_arr_dim/self.pe_arr_dim/self.pe_freq
        buffer_time = (M*K+K*N+M*N)/self.buffer_bw
        if debug:
            self.debugprint(f"leaf compute_time: {compute_time}, buffer_time: {buffer_time}")
        time = max(compute_time, buffer_time)
        # energy = (M*K*N*self.nJ_per_mac + (M*K+K*N+M*N)*(self.interconnect_nJ_per_bit + self.buffer_nJ_per_bit))
        compute_energy = M*K*N*self.nJ_per_mac
        buffer_energy = (M*K+K*N+M*N)*(self.interconnect_nJ_per_bit + self.buffer_nJ_per_bit)
        if debug:
            self.debugprint(f"leaf compute_energy: {compute_energy}, buffer_energy: {buffer_energy}")
        energy = compute_energy + buffer_energy
        return energy, time
    
    def get_gemm_latency_energy(self, M:int, K:int, N:int, debug:bool, general_tiling:bool):
        M = math.ceil(M/(self.min_gemm_size)) * self.min_gemm_size
        K = math.ceil(K/(self.min_gemm_size)) * self.min_gemm_size
        N = math.ceil(N/(self.min_gemm_size)) * self.min_gemm_size

        # leaf_tech = 'cmos-gemm-7nm'
        # energy, cycles = run_leaf_modeling(leaf_tech, M, K, N)
        # time = cycles/self.pe_freq
        energy, time = self.run_leaf_modeling_fallback(M, K, N, debug)
        if debug:
            self.debugprint(f"Leaf energy (nJ): {energy}, Leaf time (ns): {time}")
            #report buffer usage
            self.debugprint(f"buffer usage: {bytes2str((M*K+K*N+M*N)*self.bytes_per_element)}/{bytes2str(self.buffer_size_bytes)}")
        assert M*K+K*N+M*N<=self.buffer_size_elems
        # return max(M*K*N/self.pe_arr_dim/self.pe_arr_dim/self.pe_freq, M*K+K*N+M*N/self.buffer_bw), M*K*N*self.nJ_per_mac
        return time, energy
    
    def get_max_gemm_size(self):
        min_problem_size_per_leaf = self.min_gemm_size*self.min_gemm_size*3
        scale_up_factor = math.floor(math.sqrt(self.buffer_size_elems / min_problem_size_per_leaf))
        return (self.min_gemm_size*scale_up_factor, self.min_gemm_size*scale_up_factor, self.min_gemm_size*scale_up_factor)
