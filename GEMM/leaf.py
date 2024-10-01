from GEMM.arch_base import Arch_base, Log
from Leaf_Modeling.src.leaf_interface import run_leaf_modeling
import math
from helper.myMath import *
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
        self.log = Log()
    
    def update_log_recursively(self, num_iter):
        self.log.update(num_iter)
        
    def print(self):
        self.debugprint(f"pe_arr_dim: {self.pe_arr_dim}, buffer_size: {bytes2str(self.buffer_size_bytes)}, buffer_bw: {GBps2str(self.buffer_bw_GBps)}, "
                f"pe_freq: {self.pe_freq}GHz,  E_per_mac: {energy2str(self.nJ_per_mac)}, "
                f"interconnect_E_per_bit: {energy2str(self.interconnect_nJ_per_bit)}, buffer_E_per_bit: {energy2str(self.buffer_nJ_per_bit)}, min_gemm_size: {self.min_gemm_size}, precision: {self.bytes_per_element*8},"
                f"max_gemm_size: {self.get_max_gemm_size()}")
    
    def print_log(self):
        self.debugprint(self.log.toString())
    #energy in nJ, time in ns
    def run_leaf_modeling_fallback(self, M, K, N, debug=False):
        compute_time = M*K*N/self.pe_arr_dim/self.pe_arr_dim/self.pe_freq
        buffer_time = (M*K+K*N+M*N)/self.buffer_bw
        time = max(compute_time, buffer_time)
        if debug:
            self.debugprint(f"latency: {time}, T_compute: {compute_time}, T_buffer: {buffer_time}")
        # energy = (M*K*N*self.nJ_per_mac + (M*K+K*N+M*N)*(self.interconnect_nJ_per_bit + self.buffer_nJ_per_bit))
        compute_energy = M*K*N*self.nJ_per_mac
        buffer_access_bits = (M*K+K*N+M*N)*self.bytes_per_element*8
        buffer_energy = buffer_access_bits*(self.interconnect_nJ_per_bit + self.buffer_nJ_per_bit)
        energy = compute_energy + buffer_energy

        #update logs
        self.log.mac += M*K*N
        self.log.buffer_access += buffer_access_bits
        self.log.interconnect_bits += buffer_access_bits
        if debug:
            self.debugprint(f"GEMM: {M},{K},{N}")
            self.debugprint(f"energy: {energy2str(energy)}, E_compute: {energy2str(compute_energy)}, E_buffer: {energy2str(buffer_energy)}")
            self.debugprint(f"buffer load store bits: {buffer_access_bits}")
            self.debugprint(f"interconnect transfer bits: {buffer_access_bits}")
        
        return energy, time
    
    def get_gemm_latency_energy(self, M:int, K:int, N:int, debug:bool, general_tiling:bool):
        M = math.ceil(M/(self.min_gemm_size)) * self.min_gemm_size
        K = math.ceil(K/(self.min_gemm_size)) * self.min_gemm_size
        N = math.ceil(N/(self.min_gemm_size)) * self.min_gemm_size
        if debug:
            #report buffer usage
            self.debugprint(f"buffer usage: {bytes2str((M*K+K*N+M*N)*self.bytes_per_element)}/{bytes2str(self.buffer_size_bytes)}")
        # leaf_tech = 'cmos-gemm-7nm'
        # energy, cycles = run_leaf_modeling(leaf_tech, M, K, N)
        # time = cycles/self.pe_freq
        energy, time = self.run_leaf_modeling_fallback(M, K, N, debug)
        # if debug:
        #     self.debugprint(f"Leaf energy (nJ): {energy}, Leaf time (ns): {time}")
        assert M*K+K*N+M*N<=self.buffer_size_elems
        # return max(M*K*N/self.pe_arr_dim/self.pe_arr_dim/self.pe_freq, M*K+K*N+M*N/self.buffer_bw), M*K*N*self.nJ_per_mac
        return time, energy
    
    def get_max_gemm_size(self):
        min_problem_size_per_leaf = self.min_gemm_size*self.min_gemm_size*3
        scale_up_factor = math.floor(math.sqrt(self.buffer_size_elems / min_problem_size_per_leaf))
        return (self.min_gemm_size*scale_up_factor, self.min_gemm_size*scale_up_factor, self.min_gemm_size*scale_up_factor)
