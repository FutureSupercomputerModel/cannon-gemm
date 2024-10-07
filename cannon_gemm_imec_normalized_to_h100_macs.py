from GEMM.arch import Arch
from GEMM.leaf import Leaf
from GEMM.arch import top_level_gemm

leaf_imec = Leaf(pe_arr_dim=128.0, 
                      buffer_size='20.0MB', 
                      buffer_bw = f'73.34TBps',
                      pe_freq=30.0, 
                      E_per_mac='14fJ', 
                      interconnect_E_per_bit='0.00001pJ', 
                      buffer_E_per_bit="0.021fJ", 
                      bytes_per_element=2,
                      buffer_bit_area=3.125,
                      mac_area=3600)
blade_imec = Arch(mesh_dim=4.0, 
                  mesh_bw='73.34TBps', 
                  buffer_size="80GB", 
                  buffer_bw='30.0TBps', 
                  mesh_E_per_bit='5e-4pJ', 
                  buffer_E_per_bit=f'{22*0.029}pJ', 
                  child_arch=leaf_imec)
node_imec = Arch(mesh_dim=1.0, 
                 mesh_bw='1PBps', 
                 buffer_size="8TB", 
                 buffer_bw='3.0PBps', 
                 mesh_E_per_bit='5e-3pJ', 
                 buffer_E_per_bit=f'{22*0.029}pJ', 
                 child_arch=blade_imec)
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
    # (800, 800, 800)
]
for m,k,n in gemm_sizes:
    _,_,log = top_level_gemm(m,k,n, node_imec, debug=True, general_tiling=True)
    print(log)