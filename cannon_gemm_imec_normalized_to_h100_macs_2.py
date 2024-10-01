from GEMM.arch import Arch
from GEMM.leaf import Leaf
from GEMM.arch import top_level_gemm

leaf_imec = Leaf(pe_arr_dim=64.0, 
                      buffer_size='5.0MB', 
                      buffer_bw = f'73.34TBps',
                      pe_freq=30.0, 
                      nJ_per_mac=14e-6, 
                      interconnect_nJ_per_bit=1e-8, 
                      buffer_nJ_per_bit=0.021e-6, 
                      bytes_per_element=2,
                      buffer_bit_area=2.0,
                      mac_area=6700)
blade_imec = Arch(mesh_dim=8.0, 
                  mesh_bw='73.34TBps', 
                  buffer_size="80GB", 
                  buffer_bw='30.0TBps', 
                  mesh_nJ_per_bit=5e-7, 
                  buffer_nJ_per_bit=0.397e-3, 
                  child_arch=leaf_imec)
node_imec = Arch(mesh_dim=1.0, 
                 mesh_bw='1PBps', 
                 buffer_size="8TB", 
                 buffer_bw='3.0PBps', 
                 mesh_nJ_per_bit=5e-6, 
                 buffer_nJ_per_bit=0.397e-3, 
                 child_arch=blade_imec)
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
    top_level_gemm(m,k,n, node_imec, debug=True, general_tiling=True)