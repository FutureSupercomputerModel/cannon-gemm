import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
with open("cannon_gemm_para_sweep_dumped_data.csv", "r") as fp:
    reader = csv.reader(fp)
    next(reader)
    sample_size = 5000
    data = random.sample(list(reader), sample_size)


list_T = np.array([float(row[-2]) for row in data])
list_E = np.array([float(row[-1]) for row in data])
matplotlib.use('agg')
fig1 = plt.figure("Energy vs Latency")
plt.scatter(list_T, list_E, s=1)
plt.xlabel('Latency (s)')
plt.ylabel('Energy (J)')
plt.savefig('fig_latency_energy.pdf')

m,k,n = 90*200*64,90*200*64,90*200*64
fig2 = plt.figure("OPs/J vs OPs/s")
op_per_J = m*k*n/list_E
op_per_s = m*k*n/list_T
plt.scatter(op_per_J, op_per_s, s=1)
plt.xlabel('Throughput (OP/s)')
plt.ylabel('Energy Efficiency (OP/J)')
plt.savefig('fig_efficiency_vs_throughput.pdf')


list_throughput = m*k*n/list_T*1e-9
list_power = list_E/list_T
fig3 = plt.figure("Throughput vs Power")
plt.scatter(list_throughput, list_power, s=1)
plt.xlabel('Throughput (TOPS/s)')
plt.ylabel('Power (W)')
plt.savefig('fig_throughput_power.pdf')

# print(f"list_T: {list_T}")
# print(f"list_E: {list_E}")
# print(f"list_throughput: {list_throughput}")
# print(f"list_power: {list_power}")

LEAF_PE_ARR_DIM = 0
LEAF_BUFFER_SIZE = 1
LEAF_BUFFER_WIDTH = 2
LEAF_PE_FREQ = 3

BLADE_MESH_DIM = 4
BLADE_MESH_BW = 5
BLADE_BUFFER_SIZE = 6
BLADE_BUFFER_BW = 7

NODE_MESH_DIM = 8
NODE_MESH_BW = 9
NODE_BUFFER_SIZE = 10
NODE_BUFFER_BW = 11

list_param_name = ["LEAF_PE_ARR_DIM",
            "LEAF_BUFFER_SIZE",
            "LEAF_BUFFER_WIDTH",
            "LEAF_PE_FREQ",
            "BLADE_MESH_DIM",
            "BLADE_MESH_BW",
            "BLADE_BUFFER_SIZE",
            "BLADE_BUFFER_BW",
            "NODE_MESH_DIM",
            "NODE_MESH_BW",
            "NODE_BUFFER_SIZE",
            "NODE_BUFFER_BW"]


def bucket_plot_efficiency_vs_throughput(data, index):
    # Create buckets based on LEAF_PE_ARR_DIM values
    buckets = {}
    for row in data:
        key = row[index]
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(row)

    # Example: Print the number of entries in each bucket
    for key, value in buckets.items():
        print(f"{list_param_name[index]} = {key}: {len(value)} entries")

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig4 = plt.figure(f"Efficiency vs Throughput {list_param_name[index]}")
    for counter, (key, value) in enumerate(buckets.items()):
        list_T = np.array([float(row[-2]) for row in value])
        list_E = np.array([float(row[-1]) for row in value])
        op_per_J = m*k*n/list_E
        op_per_s = m*k*n/list_T
        plt.scatter(op_per_J, op_per_s, s=4, color=colors[counter%len(colors)], label=f"{list_param_name[index]} = {key}")

    plt.xlabel('Throughput (OP/s)')
    plt.ylabel('Energy Efficiency (OP/J)')
    plt.legend(loc="upper left")
    plt.savefig(f'fig_efficiency_vs_throughput_{list_param_name[index]}.pdf')

def bucket_plot_energy_vs_latency(data, index):
    # Create buckets based on LEAF_PE_ARR_DIM values
    buckets = {}
    for row in data:
        key = row[index]
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(row)

    # Example: Print the number of entries in each bucket
    for key, value in buckets.items():
        print(f"{list_param_name[index]} = {key}: {len(value)} entries")

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig4 = plt.figure(f"Energy vs Latency {list_param_name[index]}")
    for counter, (key, value) in enumerate(buckets.items()):
        list_T = np.array([float(row[-2]) for row in value])
        list_E = np.array([float(row[-1]) for row in value])
        plt.scatter(list_E, list_T, s=4, color=colors[counter%len(colors)], label=f"{list_param_name[index]} = {key}")

    plt.xlabel('Latency (s)')
    plt.ylabel('Energy (J)')
    plt.legend(loc="upper left")
    plt.savefig(f'fig_energy_vs_latency_{list_param_name[index]}.pdf')

for index in range(12):
    bucket_plot_efficiency_vs_throughput(data, index)
    bucket_plot_energy_vs_latency(data, index)