from GEMM.arch import Arch
from GEMM.leaf import Leaf
from GEMM.arch import top_level_gemm

leaf_cmos = Leaf(pe_arr_dim=128.0, 
                      buffer_size='20.0MB', 
                      buffer_bw = f'{73.34*4/30}TBps', 
                      pe_freq=4.0, 
                      nJ_per_mac=0.38e-3, 
                      interconnect_nJ_per_bit=0.1e-3, 
                      buffer_nJ_per_bit=0.5e-6, 
                      bytes_per_element=2,
                      buffer_bit_area=0.03125,
                      mac_area=0)
blade_cmos = Arch(mesh_dim=4.0, 
                  mesh_bw='73.34TBps', 
                  buffer_size="80GB", 
                  buffer_bw='30.0TBps', 
                  mesh_nJ_per_bit=5e-7, 
                  buffer_nJ_per_bit=0.397e-3, 
                  child_arch=leaf_cmos)
node_cmos = Arch(mesh_dim=1.0, 
                 mesh_bw='1PBps', 
                 buffer_size="8TB", 
                 buffer_bw='3.0PBps', 
                 mesh_nJ_per_bit=5e-6, 
                 buffer_nJ_per_bit=0.397e-3, 
                 child_arch=blade_cmos)
gemm_sizes = [
    (819200, 819200, 819200),
    (409600, 409600, 409600),
    (204800, 204800, 204800),
    (102400, 102400, 102400),
    (51200, 51200, 51200),
    (25600, 25600, 25600),
    (12800, 12800, 12800),
    (6400, 6400, 6400),
    (3200, 3200, 3200),
    (1600, 1600, 1600),
    (800, 800, 800)
]
for m,k,n in gemm_sizes:
    top_level_gemm(m,k,n, node_cmos, debug=True, general_tiling=True)