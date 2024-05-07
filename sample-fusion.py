#! /usr/bin/env python3

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import inspect
import os
import pprint
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml
import matplotlib.pyplot as plt
import copy

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
root_dir = os.path.join(os.path.dirname(this_file_path), '..')

sys.path.append(os.path.join(root_dir, 'scripts'))
from cnn_layers import *
import layerFuserRecursiveDP_gemm as layerFuserRecursiveDP
import layerFuserHelper as helper
import conv2gemm
import cannon_gemm_inf_cache as cannon_gemm

if len(sys.argv) > 1:
    buffer_size     = float(sys.argv[1])
    # raw_result_dir  = sys.argv[2]
    # stats_dir       = sys.argv[5]
else:
    print("Usage: python3 sample-fusion.py 2000 strategies > output")
    sys.exit(1)

# Create array to store important stats  
cycles_list = [] #cycles
energy_list = [] #energy_pJ
energy_per_mac_list = [] #energy_per_mac
macs_num_list = [] #macs

# Create total stats variables
total_cycles = 0 
total_energy_net = 0 

# Just test that path points to a valid config file.
# optimal_fused_groups = layerFuser.fuse_layer(config_no_dram, cnn_layers, buffer_size)
# fused_groups = optimal_fused_groups[0]

strategies = layerFuserRecursiveDP.fuse_layer_recursive_start( cnn_layers, pooling_layers, buffer_size)
helper.printStrategies(strategies)
# helper.summarizeStrategies(strategies, buffer_size)

# gemm_workload_groups = []
# for fused_groups in strategies:
#     for i in range(0, len(fused_groups)):

all_strategies_T_total = []
strategy_index=0
strategies_copy = copy.deepcopy(strategies)
for fused_groups in strategies:
    T_total = 0 
    index = 0
    cycles_list = [] #cycles
    energy_list = [] #energy_pJ
    energy_per_mac_list = [] #energy_per_mac
    macs_num_list = [] #macs
    for i in range(0, len(fused_groups)):
        group_total = 0
        input_tile_count = fused_groups[i][0][3]
        print("input_tile_count: ", input_tile_count)
        for j in range(0, len(fused_groups[i])):
            
            # print(fused_groups[i][j])
            
            
            problem = fused_groups[i][j]
            problem[3]=1

            print(f"Preparing to run cannon-gemm for strategy {strategy_index} problem {index} ")
            index+=1
            print("Problem: ", problem)

            # dirname = str(raw_result_dir) + '/strategy_' + str(strategy_index)+ '/problem_' + str(index) + '/'
            # subprocess.check_call(['mkdir', '-p', dirname])

            if len(fused_groups[i])>1:
                if j==0:
                    gemm_problem = conv2gemm.conv2gemm(problem)
                    T_prep, T_compute, T_send, T_store = cannon_gemm.cannon_gemm(gemm_problem[0], gemm_problem[1], gemm_problem[2])
                    T_layer = T_prep + T_compute + T_send
                    group_total += T_layer
                elif j==len(fused_groups[i])-1:
                    gemm_problem = conv2gemm.conv2gemm(problem)
                    T_prep, T_compute, T_send, T_store = cannon_gemm.cannon_gemm(gemm_problem[0], gemm_problem[1], gemm_problem[2])
                    T_layer = T_compute + T_send + T_store
                    group_total += T_layer
                else:
                    gemm_problem = conv2gemm.conv2gemm(problem)
                    T_prep, T_compute, T_send, T_store = cannon_gemm.cannon_gemm(gemm_problem[0], gemm_problem[1], gemm_problem[2])
                    T_layer = T_compute + T_send
                    group_total += T_layer
            else:
                gemm_problem = conv2gemm.conv2gemm(problem)
                T_prep, T_compute, T_send, T_store = cannon_gemm.cannon_gemm(gemm_problem[0], gemm_problem[1], gemm_problem[2])
                T_layer = T_prep + T_compute + T_send + T_store
                group_total += T_layer
            # if len(fused_groups[i])>1:
            #     gemm_problem = conv2gemm.conv2gemm(problem)
            #     print("GEMM Problem: ", gemm_problem)
            # else:
            #     gemm_problem = conv2gemm.conv2gemm(problem)
            #     print("GEMM Problem: ", gemm_problem)

            # for k in range(0, len(gemm_problem)):
            #     gemm_problem[i] = int(gemm_problem[i])
        if len(fused_groups[i])>1:
            group_total = group_total * input_tile_count
        T_total += group_total
    print("T_total: ", T_total)
    all_strategies_T_total.append(T_total)
    strategy_index += 1

print("all_strategies_T_total: ", all_strategies_T_total)

macs = []
offchip_access = []
for i in range(len(strategies_copy)):
    total_macs = helper.get_total_macs(strategies_copy[i])
    total_offchip_access = helper.get_total_offchip_access(strategies_copy[i])
    macs.append(total_macs)
    offchip_access.append(total_offchip_access)

size = [10/((x/max(all_strategies_T_total)) **7)  for x in all_strategies_T_total]
print("macs:", macs)
print("offchip_access:", offchip_access)
plt.scatter(macs, offchip_access, s=size, alpha=0.5)
plt.xlabel("Total MACs")
plt.ylabel("Total Off-chip Access")


for i in range(len(strategies_copy)):
        print(f"strategy {i}: {all_strategies_T_total[i]} cycles")
        print("\ttotal macs="+ str(helper.get_total_macs(strategies_copy[i])) + " total offchip access="+ str(helper.get_total_offchip_access(strategies_copy[i])))
        helper.printStrategy(strategies_copy[i])

index_min = np.argmin(all_strategies_T_total)
print("Best strategy: ", index_min)
helper.printStrategy(strategies_copy[index_min])

plt.scatter(macs[index_min], offchip_access[index_min], s=size[index_min], alpha=0.5, c='red')

plt.show()
plt.savefig(f"macs_vs_offchip_access_{buffer_size}_cycles.png")