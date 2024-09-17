import numpy as np
import arch_lib
from GEMM.arch import top_level_gemm


gemm_sizes = [(90*200,90*200,90*200),
              (90*200*2,90*200*2,90*200*2),
              (90*200*4,90*200*4,90*200*4),
              (90*200*8,90*200*8,90*200*8),
              (90*200*16,90*200*16,90*200*16),
              (90*200*32,90*200*32,90*200*32),
              (90*200*64,90*200*64,90*200*64),
              ]


leaf_ = Leaf(pe_arr_dim=200.0, 
                      buffer_size_bytes=20.0e6, 
                      buffer_bw_GBps = 64.0*30.0/8,
                      pe_freq=30.0, 
                      nJ_per_mac=14e-6, 
                      interconnect_nJ_per_bit=1e-8, 
                      buffer_nJ_per_bit=0.021e-6, 
                      bytes_per_element=2)
class TaggedNumber:
    def __init__(self, value, tag):
        self.value = value
        self.tag = tag
leaf_pe_arr_dim = [64, 128, 200, 256]
leaf_buffer_size_bytes = [1.0e6, 5.0e6, 10.0e6, 20.0e6, 40.0e6]
leaf_buffer_width = [16, 32, 64, 128]#x is frequency
leaf_pe_freq = [4.0, 10.0, 30.0, 60.0]

blade_mesh_dim = [4.0, 9.0, 16.0, 32.0]
blade_mesh_bw_GBps = [120, 240, 480, 960]
blade_buffer_size_bytes = [2.0e9, 4.0e9, 8.0e9, 16.0e9]
blade_buffer_bw_GBps = [15.0e3, 30.0e3, 60.0e3, 120.0e3]

node_mesh_dim = [5.0, 10.0, 20.0, 40.0]
node_mesh_bw_GBps = [0.5e6, 1e6, 2e6, 4e6]
node_buffer_size_bytes = [5.0e9, 10.0e9, 20.0e9, 40.0e9]
node_buffer_bw_GBps = [1.5e6, 3.0e6, 6.0e6, 12.0e6]

for 
m,k,n = (90*200*64,90*200*16,90*200)
top_level_gemm(m,k,n, arch_lib.hier_arch_node, debug=True)

