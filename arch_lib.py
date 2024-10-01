from GEMM.arch import Arch
from GEMM.arch import top_level_gemm
from GEMM.leaf import Leaf
leaf_cmos = Leaf(pe_arr_dim=200.0, 
                      buffer_size='20.0MB', 
                      buffer_bw = f'{73.34*4/30}TBps', 
                      pe_freq=4.0, 
                      nJ_per_mac=0.38e-3, 
                      interconnect_nJ_per_bit=0.1e-3, 
                      buffer_nJ_per_bit=0.5e-6, 
                      bytes_per_element=2,
                      buffer_bit_area=0.03125,
                      mac_area=0)
leaf_imec = Leaf(pe_arr_dim=200.0, 
                      buffer_size='20.0MB', 
                      buffer_bw = f'73.34TBps',
                      pe_freq=30.0, 
                      nJ_per_mac=14e-6, 
                      interconnect_nJ_per_bit=1e-8, 
                      buffer_nJ_per_bit=0.021e-6, 
                      bytes_per_element=2,
                      buffer_bit_area=2.0,
                      mac_area=6700)


blade_imec = Arch(mesh_dim=9.0, 
                  mesh_bw='73.34TBps', 
                  buffer_size="80GB", 
                  buffer_bw='30.0TBps', 
                  mesh_nJ_per_bit=5e-7, 
                  buffer_nJ_per_bit=0.397e-3, 
                  child_arch=leaf_imec)
node_imec = Arch(mesh_dim=10.0, 
                 mesh_bw='1PBps', 
                 buffer_size="8TB", 
                 buffer_bw='3.0PBps', 
                 mesh_nJ_per_bit=5e-6, 
                 buffer_nJ_per_bit=0.397e-3, 
                 child_arch=blade_imec)


leaf_imec_large = Leaf(pe_arr_dim=100.0, 
                      buffer_size='40.0MB', 
                      buffer_bw = f'73.34TBps',
                      pe_freq=30.0, 
                      nJ_per_mac=14e-6, 
                      interconnect_nJ_per_bit=1e-8, 
                      buffer_nJ_per_bit=0.021e-6, 
                      bytes_per_element=2,
                      buffer_bit_area=2.0,
                      mac_area=6700)
blade_imec_large = Arch(mesh_dim=18.0, 
                        mesh_bw='73.34TBps', 
                        buffer_size="80GB", 
                        buffer_bw='30.0TBps', 
                        mesh_nJ_per_bit=5e-7, 
                        buffer_nJ_per_bit=0.397e-3, 
                        child_arch=leaf_imec_large)
node_imec_large= Arch(mesh_dim=20.0, 
                      mesh_bw='1PBps', 
                      buffer_size="8TB", 
                      buffer_bw='3.0PBps', 
                      mesh_nJ_per_bit=5e-6, 
                      buffer_nJ_per_bit=0.397e-3, 
                      child_arch=blade_imec_large)

node_imec_1blade = Arch(mesh_dim=1.0, 
                 mesh_bw='1PBps', 
                 buffer_size="8TB", 
                 buffer_bw='3.0PBps', 
                 mesh_nJ_per_bit=5e-6, 
                 buffer_nJ_per_bit=0.397e-3, 
                 child_arch=blade_imec)


blade_imec_1leaf = Arch(mesh_dim=1.0,
                         mesh_bw=f'73.34TBps', 
                         buffer_size="80GB", 
                         buffer_bw='30.0TBps', 
                         mesh_nJ_per_bit=5e-7, 
                         buffer_nJ_per_bit=0.397e-3, 
                         child_arch=leaf_imec)

node_imec_1leaf = Arch(mesh_dim=1.0,
                 mesh_bw='1PBps', 
                 buffer_size="8TB", 
                 buffer_bw='3.0PBps', 
                 mesh_nJ_per_bit=5e-6, 
                 buffer_nJ_per_bit=0.397e-3, 
                 child_arch=blade_imec_1leaf)

blade_imec_3x3leaf = Arch(mesh_dim=3.0,
                    mesh_bw=f'73.34TBps', 
                    buffer_size="80GB", 
                    buffer_bw='30.0TBps', 
                    mesh_nJ_per_bit=5e-7, 
                    buffer_nJ_per_bit=0.397e-3, 
                    child_arch=leaf_imec)
node_imec_3x3leaf = Arch(mesh_dim=1.0,
                        mesh_bw='1PBps', 
                        buffer_size="8TB", 
                        buffer_bw='3.0PBps', 
                        mesh_nJ_per_bit=5e-6, 
                        buffer_nJ_per_bit=0.397e-3, 
                        child_arch=blade_imec_3x3leaf)

blade_cmos_3x3leaf = Arch(mesh_dim=3.0,
                    mesh_bw= f'{73.34*4/30}TBps', 
                    buffer_size="80GB", 
                    buffer_bw='30.0TBps', 
                    mesh_nJ_per_bit=5e-7, 
                    buffer_nJ_per_bit=0.397e-3, 
                    child_arch=leaf_cmos)
node_cmos_3x3leaf = Arch(mesh_dim=1.0, 
                        mesh_bw='1PBps', 
                        buffer_size="8TB", 
                        buffer_bw='3.0PBps', 
                        mesh_nJ_per_bit=5e-6, 
                        buffer_nJ_per_bit=0.397e-3, 
                        child_arch=blade_cmos_3x3leaf)
