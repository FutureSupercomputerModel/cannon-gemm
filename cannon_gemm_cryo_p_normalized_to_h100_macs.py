from GEMM.arch import Arch
from GEMM.leaf import Leaf
from GEMM.arch import top_level_gemm

leaf_cryo = Leaf(pe_arr_dim=128.0*4, 
                      buffer_size=f'{20.0*16}MB', 
                      buffer_bw = f'{73.34*4/30}TBps', 
                      pe_freq=5.6, 
                      E_per_mac="8.4pJ", 
                      interconnect_E_per_bit="0.1pJ", 
                      buffer_E_per_bit='11fJ', 
                      bytes_per_element=2,
                      buffer_bit_area=0.058,
                      mac_area=550)
blade_cryo = Arch(mesh_dim=1.0, 
                  mesh_bw=f'{73.34*4/30}TBps', 
                  buffer_size="80GB", 
                  buffer_bw='30.0TBps', 
                  mesh_E_per_bit='0.1pJ', 
                  buffer_E_per_bit=f'{22*0.29}pJ', 
                  child_arch=leaf_cryo)
node_cryo = Arch(mesh_dim=1.0, 
                 mesh_bw='1PBps', 
                 buffer_size="8TB", 
                 buffer_bw='3.0PBps', 
                 mesh_E_per_bit='0.1pJ', 
                 buffer_E_per_bit=f'{22*0.29}pJ', 
                 child_arch=blade_cryo)
gemm_sizes = [
    (819200, 819200, 819200),
    # (409600, 409600, 409600),
    # (204800, 204800, 204800),
    # (102400, 102400, 102400),
    # (51200, 51200, 51200),
    # (25600, 25600, 25600),
    # (12800, 12800, 12800),
    # (6400, 6400, 6400),
    # (3200, 3200, 3200),
    # (1600, 1600, 1600),
    # (800, 800, 800),
    # (400, 400, 400),
    # (200, 200, 200)
]
for m,k,n in gemm_sizes:
    top_level_gemm(m,k,n, node_cryo, debug=True, general_tiling=True)