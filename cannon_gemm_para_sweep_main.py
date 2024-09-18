import numpy as np
from GEMM.arch import Arch
from GEMM.leaf import Leaf
from GEMM.arch import top_level_gemm

list_leaf_pe_arr_dim = [64, 128, 200, 256]
list_leaf_buffer_size = ['1MB', "5MB", "10MB", "20MB", "40MB"]
list_leaf_buffer_width = [16, 32, 64, 128]#x is frequency
list_leaf_pe_freq = [4.0, 10.0, 30.0, 60.0]

list_blade_mesh_dim = [4.0, 9.0, 16.0, 32.0]
list_blade_mesh_bw = ['120GBps', '240GBps', '480GBps', '960GBps']
list_blade_buffer_size = ['2.0TB', '4.0TB', '8.0TB', '16.0TB']
list_blade_buffer_bw = ['15.0TBps', '30.0TBps', '60.0TBps', '120.0TBps']

list_node_mesh_dim = [5.0, 10.0, 20.0, 40.0]
list_node_mesh_bw = ['0.5PBps', '1PBps', '2PBps', '4PBps']
list_node_buffer_size = ['8TB']
list_node_buffer_bw = ['1.5PBps', '3.0PBps', '6.0PBps', '12.0PBps']

# list_leaf_pe_arr_dim = [128, 200]
# list_leaf_buffer_size = ["10MB", "20MB"]
# list_leaf_buffer_width = [16, 32]#x is frequency
# list_leaf_pe_freq = [10.0, 30.0]

# list_blade_mesh_dim = [4.0, 9.0]
# list_blade_mesh_bw = ['120GBps', '240GBps']
# list_blade_buffer_size = ['4.0TB', '8.0TB']
# list_blade_buffer_bw = ['15.0TBps', '30.0TBps']

# list_node_mesh_dim = [5.0, 10.0]
# list_node_mesh_bw = ['0.5PBps', '1PBps']
# list_node_buffer_size = ['8TB']
# list_node_buffer_bw = ['1.5PBps', '3.0PBps']

class Sys_arch:
    def __init__(self, leaf_pe_arr_dim,leaf_buffer_size,leaf_buffer_width,leaf_pe_freq,\
              blade_mesh_dim,blade_mesh_bw,blade_buffer_size,blade_buffer_bw,\
                node_mesh_dim,node_mesh_bw,node_buffer_size,node_buffer_bw):
        self.leaf_pe_arr_dim = leaf_pe_arr_dim
        self.leaf_buffer_size = leaf_buffer_size
        self.leaf_buffer_width = leaf_buffer_width
        self.leaf_pe_freq = leaf_pe_freq
        self.blade_mesh_dim = blade_mesh_dim
        self.blade_mesh_bw = blade_mesh_bw
        self.blade_buffer_size = blade_buffer_size
        self.blade_buffer_bw = blade_buffer_bw
        self.node_mesh_dim = node_mesh_dim
        self.node_mesh_bw = node_mesh_bw
        self.node_buffer_size = node_buffer_size
        self.node_buffer_bw = node_buffer_bw
    def cannon_gemm(self, m, k, n, debug=False):
        hier_arch_leaf = Leaf(pe_arr_dim=self.leaf_pe_arr_dim, 
                      buffer_size=self.leaf_buffer_size, 
                      buffer_bw = f'{self.leaf_buffer_width*self.leaf_pe_freq/8}GBps', 
                      pe_freq=self.leaf_pe_freq, 
                      nJ_per_mac=14e-6, 
                      interconnect_nJ_per_bit=1e-8, 
                      buffer_nJ_per_bit=0.021e-6, 
                      bytes_per_element=2)
        hier_arch_blade = Arch(mesh_dim=self.blade_mesh_dim, mesh_bw=self.blade_mesh_bw, buffer_size=self.blade_buffer_size, buffer_bw=self.blade_buffer_bw, mesh_nJ_per_bit=5e-7, buffer_nJ_per_bit=0.397e-3, child_arch=hier_arch_leaf)
        hier_arch_node = Arch(mesh_dim=self.node_mesh_dim, mesh_bw=self.node_mesh_bw, buffer_size=self.node_buffer_size, buffer_bw=self.node_buffer_bw, mesh_nJ_per_bit=5e-6, buffer_nJ_per_bit=0.397e-3, child_arch=hier_arch_blade)
        T_top, E_total = top_level_gemm(m,k,n, hier_arch_node, debug=debug, general_tiling=False)
        return T_top, E_total

sys_arch_list = [Sys_arch(leaf_pe_arr_dim,leaf_buffer_size,leaf_buffer_width,leaf_pe_freq,\
              blade_mesh_dim,blade_mesh_bw,blade_buffer_size,blade_buffer_bw,\
                node_mesh_dim,node_mesh_bw,node_buffer_size,node_buffer_bw) \
                    for leaf_pe_arr_dim in list_leaf_pe_arr_dim for leaf_buffer_size in list_leaf_buffer_size for leaf_buffer_width in list_leaf_buffer_width for leaf_pe_freq in list_leaf_pe_freq \
                        for blade_mesh_dim in list_blade_mesh_dim for blade_mesh_bw in list_blade_mesh_bw for blade_buffer_size in list_blade_buffer_size for blade_buffer_bw in list_blade_buffer_bw \
                            for node_mesh_dim in list_node_mesh_dim for node_mesh_bw in list_node_mesh_bw for node_buffer_size in list_node_buffer_size for node_buffer_bw in list_node_buffer_bw]
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

m,k,n = 90*200*64,90*200*64,90*200*64
list_T = np.array([])
list_E = np.array([])
data = []
print(f"number of experiments: {len(sys_arch_list)}")
finished_exps = 0
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import json
import csv

def exp(sys_arch:Sys_arch):
    T_top, E_total = sys_arch.cannon_gemm(m,k,n, debug=False)
    return sys_arch, T_top, E_total

max_processes = int(multiprocessing.cpu_count())
print("Maximum number of processes:", max_processes)
pool = Pool(max_processes)
# res = tqdm(pool.imap_unordered(exp, sys_arch_list), total=len(sys_arch_list)) 
res = list(tqdm(pool.imap_unordered(exp, sys_arch_list), total=len(sys_arch_list)))
print(f"finished all experiments")

with open("cannon_gemm_para_sweep_dumped_data.csv", "w") as fp:
    writer = csv.writer(fp)
    writer.writerow(["leaf_pe_arr_dim","leaf_buffer_size","leaf_buffer_width","leaf_pe_freq",\
              "blade_mesh_dim","blade_mesh_bw","blade_buffer_size","blade_buffer_bw",\
                "node_mesh_dim","node_mesh_bw","node_buffer_size","node_buffer_bw", "T_top(s)", "E_total(J)"])
    for sys_arch, T_top, E_total in res:
        #for plot
        # list_T = np.append(list_T, T_top)
        # list_E = np.append(list_E, E_total)
        #for dump
        data=[sys_arch.leaf_pe_arr_dim,sys_arch.leaf_buffer_size,sys_arch.leaf_buffer_width,sys_arch.leaf_pe_freq,\
                sys_arch.blade_mesh_dim,sys_arch.blade_mesh_bw,sys_arch.blade_buffer_size,sys_arch.blade_buffer_bw,\
                    sys_arch.node_mesh_dim,sys_arch.node_mesh_bw,sys_arch.node_buffer_size,sys_arch.node_buffer_bw, T_top, E_total]
        writer.writerow(data)
# with open("cannon_gemm_para_sweep_dumped_data", "w") as fp:
#     json.dump(data, fp)



# matplotlib.use('agg')
# fig1 = plt.figure("Latency vs Energy")
# plt.scatter(list_T, list_E)
# plt.xlabel('Latency (s)')
# plt.ylabel('Energy (J)')
# plt.savefig('cannon_gemm_para_sweep_latency_energy.pdf')

# list_throughput = m*k*n/list_T*1e-9
# list_power = list_E/list_T
# fig2 = plt.figure("Throughput vs Power")
# plt.scatter(list_throughput, list_power)
# plt.xlabel('Throughput (TOPS/s)')
# plt.ylabel('Power (W)')
# plt.savefig('cannon_gemm_para_sweep_throughput_power.pdf')

# print(f"list_T: {list_T}")
# print(f"list_E: {list_E}")
# print(f"list_throughput: {list_throughput}")
# print(f"list_power: {list_power}")