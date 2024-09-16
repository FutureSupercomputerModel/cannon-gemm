from GEMM.arch import Arch
from GEMM.arch import top_level_gemm
from GEMM.leaf import Leaf
hier_arch_leaf_cmos = Leaf(pe_arr_dim=200.0, 
                      buffer_size_bytes=20.0e6, 
                      buffer_bw_GBps = 64.0*4.0/8, 
                      pe_freq=4.0, 
                      nJ_per_mac=0.38e-3, 
                      interconnect_nJ_per_bit=0.1e-3, 
                      buffer_nJ_per_bit=0.5e-6, 
                      bytes_per_element=2)
hier_arch_leaf_imec = Leaf(pe_arr_dim=200.0, 
                      buffer_size_bytes=20.0e6, 
                      buffer_bw_GBps = 64.0*30.0/8,
                      pe_freq=30.0, 
                      nJ_per_mac=14e-6, 
                      interconnect_nJ_per_bit=1e-8, 
                      buffer_nJ_per_bit=0.021e-6, 
                      bytes_per_element=2)

hier_arch_blade = Arch(mesh_dim=9.0, mesh_bw_GBps=240, buffer_size_bytes=8.0e9, buffer_bw_GBps=30.0e3, mesh_nJ_per_bit=5e-7, buffer_nJ_per_bit=0.397e-3, child_arch=hier_arch_leaf_imec)
hier_arch_node = Arch(mesh_dim=10.0, mesh_bw_GBps=1e6, buffer_size_bytes=10.0e9, buffer_bw_GBps=3.0e6, mesh_nJ_per_bit=5e-6, buffer_nJ_per_bit=0.397e-3, child_arch=hier_arch_blade)
