import math
import numpy as np
from GEMM.arch_base import Arch_base
from GEMM.leaf import Leaf
from helper.myMath import bytes2str, str2bytes, str2GBps
class Arch(Arch_base):
    # #Buffer
    # buffer_size = 8.0*1024*1024*1024 #8GB
    # buffer_bw = 64.0 #buffer bandwidth, element per ns. This buffer is shared by mesh_dim*mesh_dim processors
    # #interconnect
    # ns_setup_interconnect = 1.0 #ns to set up connection
    # mesh_bw = 64.0 #mesh bandwidth, elements per ns
    # mesh_dim = 90.0
    # p=mesh_dim*mesh_dim #processor count
    # child_arch:Arch_base = None
    # min_gemm_size = None
    # nJ_per_Byte = 1.0

    bytes_per_element = 2

    # bottleneck flags
    T_send_bottleneck = None



    def __init__(self, mesh_dim:float, mesh_bw:str, buffer_size:str, buffer_bw:str, 
                 mesh_nJ_per_bit:float, buffer_nJ_per_bit:float, 
                 child_arch:Arch_base,
                 ns_setup_interconnect:float=0,
                 roofline:bool=True,
                 ) -> None:
        #arch params
        self.mesh_dim = mesh_dim
        self.mesh_bw_GBps = str2GBps(mesh_bw)
        self.buffer_size_bytes = str2bytes(buffer_size)
        self.buffer_bw_GBps = str2GBps(buffer_bw)
        

        #tech params
        self.mesh_nJ_per_bit = mesh_nJ_per_bit
        self.buffer_nJ_per_bit = buffer_nJ_per_bit

        self.ns_setup_interconnect = ns_setup_interconnect

        self.buffer_size_elems = self.buffer_size_bytes/self.bytes_per_element # #elements
        self.buffer_bw = self.buffer_bw_GBps/self.bytes_per_element #elements per ns
        self.mesh_bw = self.mesh_bw_GBps/self.bytes_per_element #elements per ns
        self.p = mesh_dim*mesh_dim

        self.roofline = roofline
        self.child_arch = child_arch
        self.level = child_arch.level + 1
        self.min_gemm_size = self.mesh_dim * child_arch.min_gemm_size
    
    
    def get_max_gemm_size(self):
        min_problem_dim = self.mesh_dim * self.child_arch.min_gemm_size
        min_problem_size = min_problem_dim**2*3
        scale_up_factor = math.floor(math.sqrt(self.buffer_size_elems / min_problem_size))
        return (min_problem_dim*scale_up_factor, min_problem_dim*scale_up_factor, min_problem_dim*scale_up_factor)

    def spatial_tile_gemm(self, m_in,k_in,n_in,debug:bool):
        # pad m to be multiple of mesh_dim * pe_arr_dim
        m = math.ceil(m_in/(self.mesh_dim*self.child_arch.min_gemm_size)) * self.mesh_dim * self.child_arch.min_gemm_size
        # pad n to be multiple of mesh_dim * pe_arr_dim
        n = math.ceil(n_in/(self.mesh_dim*self.child_arch.min_gemm_size)) * self.mesh_dim * self.child_arch.min_gemm_size
        # pad k to be multiple of mesh_dim * pe_arr_dim
        k = math.ceil(k_in/(self.mesh_dim*self.child_arch.min_gemm_size)) * self.mesh_dim * self.child_arch.min_gemm_size
        if debug:
            self.debugprint(f"padding: from {m_in}, {k_in}, {n_in} to padded problem: {m}, {k}, {n}")
        #tile m, n to leaf problems
        m_leaf = m/self.mesh_dim
        k_leaf = k/self.mesh_dim
        n_leaf = n/self.mesh_dim
        if debug:
            self.debugprint(f"spatial tiling: from {m}, {k}, {n}, to leaf problem: {m_leaf}, {k_leaf}, {n_leaf}")
        return (m, k, n, m_leaf, k_leaf, n_leaf)
    
    def cannon_gemm(self, m_in,k_in,n_in,debug:bool, general_tiling):
        m,k,n,m_leaf,k_leaf,n_leaf = self.spatial_tile_gemm(m_in,k_in,n_in,debug)

        T_prep_A = self.ns_setup_interconnect + max(m*k/self.p/self.mesh_bw, m*k/self.p/self.child_arch.buffer_bw)#time to set up connection + max ( time for interconnect send, time for child buffer receive)
        T_prep_B = self.ns_setup_interconnect + max(k*n/self.p/self.mesh_bw, k*n/self.p/self.child_arch.buffer_bw)
        T_prep = max(T_prep_A+T_prep_B, (m*k+k*n)/self.buffer_bw )#time to load A and B, potentially bound by dram bandwidth
        T_compute, E_compute = tuple(self.mesh_dim * x for x in self.child_arch.get_gemm_latency_energy(m_leaf, k_leaf, n_leaf, debug, general_tiling))
        
        T_send_A = (self.mesh_dim-1)*(self.ns_setup_interconnect+max(m*k/self.p/self.buffer_bw, m*k/self.p/self.child_arch.buffer_bw))
        T_send_B = (self.mesh_dim-1)*(self.ns_setup_interconnect+max(k*n/self.p/self.buffer_bw, k*n/self.p/self.child_arch.buffer_bw))
        T_send = T_send_A + T_send_B
        if(T_send_A > T_send_B):
            self.T_send_bottleneck = "A"
        elif(T_send_A < T_send_B):
            self.T_send_bottleneck = "B"
        T_store = self.ns_setup_interconnect+max(m*n/self.buffer_bw, m*n/self.p/self.child_arch.buffer_bw)
        if debug:
            self.debugprint(f"latency breakdown: T_prep: {T_prep}, T_compute: {T_compute}, T_send: {T_send}, T_store: {T_store}")
        if self.roofline:
            latency = max(T_prep, T_compute, T_send, T_store)
        else:
            latency = T_prep + T_compute + T_send + T_store

        #energy
        E_prep_buffer_read = (m_leaf*k_leaf+k_leaf*n_leaf)*self.p*self.buffer_nJ_per_bit*self.bytes_per_element*8.0
        E_prep_child_buffer_write = (m_leaf*k_leaf+k_leaf*n_leaf)*self.p*self.child_arch.buffer_nJ_per_bit*self.bytes_per_element*8.0
        E_prep_interconnect = self.mesh_dim * (1+self.mesh_dim) * self.mesh_nJ_per_bit * self.bytes_per_element * 8.0 * (m_leaf*k_leaf + k_leaf*n_leaf)/2.0
        E_prep = E_prep_buffer_read + E_prep_child_buffer_write + E_prep_interconnect
        E_send_child_buffer = self.p * self.child_arch.buffer_nJ_per_bit * 2 * (m_leaf*k_leaf + k_leaf*n_leaf) * self.bytes_per_element * 8.0
        E_send_interconnect = self.p * self.mesh_nJ_per_bit * (m_leaf*k_leaf + k_leaf*n_leaf) * self.bytes_per_element * 8.0
        E_send = E_send_child_buffer + E_send_interconnect
        E_total = E_prep + E_compute + E_send

        return latency, E_total
    

        

    def temp_tile_gemm(self, m,k,n, debug):
        #temporal tiling
        assert (m*k+n*k+m*n)*self.bytes_per_element <= self.buffer_size_bytes, f"problem size exceeds buffer size: {bytes2str((m*k+n*k+m*n)*self.bytes_per_element)} > {bytes2str(self.buffer_size_bytes)}"
        (m_tile, k_tile, n_tile) = tuple(self.mesh_dim * np.array(self.child_arch.get_max_gemm_size()))
        # (m_tile, k_tile, n_tile) = (m,k,n)
        iteration = math.ceil(m/m_tile)*math.ceil(k/k_tile)*math.ceil(n/n_tile)
        if m_tile>m and k_tile>k and n_tile>n: #no need to tile
            m_tile=m
            k_tile=k
            n_tile=n
        if debug:
            self.debugprint(f"temporal tiling: from {m}, {k}, {n}, to tiled problem: {m_tile}, {k_tile}, {n_tile}, on {iteration} iterations")
        return (m_tile, k_tile, n_tile, iteration)
    
    def temp_tile_gemm_general(self, m,k,n, debug):
        #general temporal tiling, works for unsquare matrices
        #initialize tiling factors
        p=m
        q=k
        r=n
        p_opt = p
        q_opt = q
        r_opt = r
        class ExitAllLoops(Exception):
            pass
        #p=q=r
        try:
            for pqr in range(1,int(max(m,k,n))+1):
                if pqr>p_opt and pqr>q_opt and pqr>r_opt:
                    raise ExitAllLoops
                m_tile = math.ceil(m/pqr)
                k_tile = math.ceil(k/pqr)
                n_tile = math.ceil(n/pqr)
                m_pad,k_pad,n_pad,m_leaf,k_leaf,n_leaf = self.spatial_tile_gemm(m_tile,k_tile,n_tile,debug=False)
                if (m_leaf*k_leaf+k_leaf*n_leaf+m_leaf*n_leaf) <= self.child_arch.buffer_size_elems and pqr*pqr*pqr<p_opt*q_opt*r_opt:
                    #valid pqr
                    p_opt=pqr
                    q_opt=pqr
                    r_opt=pqr
        except ExitAllLoops:
            pass  # Handle the exception and continue execution

        try:
            for p in range(1,int(m)+1):
                if p > p_opt*q_opt*r_opt:
                    break
                for q in range(1,int(k)+1):
                    if p*q > p_opt*q_opt*r_opt:
                        break
                    for r in range(1,int(n)+1):
                        if p>p_opt and q>q_opt and r>r_opt:
                            raise ExitAllLoops
                        if p*q*r > p_opt*q_opt*r_opt:
                            break
                        m_tile = math.ceil(m/p)
                        k_tile = math.ceil(k/q)
                        n_tile = math.ceil(n/r)
                        m_pad,k_pad,n_pad,m_leaf,k_leaf,n_leaf = self.spatial_tile_gemm(m_tile,k_tile,n_tile,debug=False)
                        if (m_leaf*k_leaf+k_leaf*n_leaf+m_leaf*n_leaf) <= self.child_arch.buffer_size_elems and p*q*r<p_opt*q_opt*r_opt:
                            #valid pqr
                            p_opt=p
                            q_opt=q
                            r_opt=r
        except ExitAllLoops:
            pass  # Handle the exception and continue execution
        m_tile = math.ceil(m/p_opt)
        k_tile = math.ceil(k/q_opt)
        n_tile = math.ceil(n/r_opt)
        iteration = p_opt*q_opt*r_opt
        if debug:
            self.debugprint(f"temporal tiling: from {m}, {k}, {n}, to tiled problem: {m_tile}, {k_tile}, {n_tile}, on {iteration} iterations")
        return (m_tile, k_tile, n_tile, iteration)
    
    def get_gemm_latency_energy(self, m,k,n, debug:bool, general_tiling=True):
        #report buffer usage
        if debug:
            self.debugprint(f"buffer usage: {bytes2str((m*k+k*n+m*n)*self.bytes_per_element)}/{bytes2str(self.buffer_size_bytes)}")
        if general_tiling:
            (m_tile, k_tile, n_tile, iteration) = self.temp_tile_gemm_general(m,k,n, debug)
        else:
            (m_tile, k_tile, n_tile, iteration) = self.temp_tile_gemm(m,k,n, debug)
        T_tile, E_tile=self.cannon_gemm(m_tile, k_tile, n_tile, debug, general_tiling)
        if debug:
            self.debugprint(f"latency per iteration: {T_tile}, energy per iteration: {E_tile}")
        return T_tile*iteration, E_tile*iteration

    
    
    def print(self):
        self.debugprint(f"mesh_dim={self.mesh_dim}, buffer_size={bytes2str(self.buffer_size_bytes)}, buffer_bw={self.buffer_bw_GBps}GBps, mesh_bw={self.mesh_bw_GBps}GBps, mesh_nJ_per_bit={self.mesh_nJ_per_bit}nJ, buffer_nJ_per_bit={self.buffer_nJ_per_bit}nJ, min_gemm_size={self.min_gemm_size}, max_gemm_size: {self.get_max_gemm_size()}")
        if self.child_arch is not None:
            self.child_arch.print()


# cmos_arch = Arch(buffer_bw=64.0*90.0*2, ns_setup_interconnect=1.0, mesh_bw=64.0*4.0, mesh_dim=90.0, pe_arr_dim=200.0, pe_freq=4.0, buffer_size=20.0*1024*1024, buffer_bw=200*4.0*3)
# # cmos_arch_ideal = arch.Arch(buffer_bw=64.0*90.0*2, ns_setup_interconnect=1.0, mesh_bw=64.0*3, mesh_H=90.0, mesh_W=90.0, pe_arr_H=200.0, pe_arr_W=200.0, pe_freq=4.0, buffer_size=20.0*1024*1024)
# imec_arch = Arch(buffer_bw=1000000, ns_setup_interconnect=1.0, mesh_bw=200.0*30.0, mesh_dim=90.0, pe_arr_dim=200.0, pe_freq=30.0, buffer_size=20.0*1024*1024, buffer_bw=200*30.0*3)


#return time in s and energy in J
def top_level_gemm(m,k,n, arch: Arch, debug:bool, general_tiling=True):
    if debug:
        print("================Arch=====================")
        arch.print()
        print("----------------Problem---------------------")
        print(f"top level problem: {m},{k},{n}")
    T_top, E_total=arch.get_gemm_latency_energy(m, k, n, debug, general_tiling)
    if debug:
        print(f"s for top level problem: {T_top*1e-9}")
        print(f"J for top level problem: {E_total*1e-9}")
        print("=====================================")
    return T_top*1e-9, E_total*1e-9
    