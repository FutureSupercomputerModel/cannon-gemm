import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import pandas as pd
import os
from collections import Counter

def filter_unique(lst):
    return [item for item, count in Counter(lst).items() if count > 1]

def filter_top_percent(df, list_columns, percent):
    """
    Filters the DataFrame to include only the rows where the values in the specified columns
    are within the top percent.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        list_columns (list): The list of column names to analyze.
        percent (float): The percentage (between 0 and 1) representing the top values.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows within the top percent range for the specified columns.
    """
    thresholds = {column: df[column].quantile(1 - percent) for column in list_columns}
    print(thresholds)
    filtered_df = df.copy()
    for column, threshold in thresholds.items():
        filtered_df = filtered_df[filtered_df[column] >= threshold]
    return filtered_df

def filter_bottom_percent(df, list_columns, percent):
    """
    Filters the DataFrame to include only the rows where the values in the specified columns
    are within the bottom percent.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        list_columns (list): The list of column names to analyze.
        percent (float): The percentage (between 0 and 1) representing the bottom values.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows within the bottom percent range for the specified columns.
    """
    thresholds = {column: df[column].quantile(percent) for column in list_columns}
    print(thresholds)
    filtered_df = df.copy()
    for column, threshold in thresholds.items():
        filtered_df = filtered_df[filtered_df[column] <= threshold]
    return filtered_df


# if filtered data does not exist, load the raw data and filter it
if not os.path.exists("cannon_gemm_para_sweep_dumped_data_filtered.csv"):
    print("loading raw data...")
    with open("cannon_gemm_para_sweep_dumped_data.csv", "r") as fp:
        reader = csv.reader(fp)
        next(reader)
        # sample_size = 5000
        # data = random.sample(list(reader), sample_size)
        data = list(reader)
    print(f"number of experiments: {len(data)}")

    #convert data to pandas dataframe
    df = pd.DataFrame(data, columns=["leaf_pe_arr_dim","leaf_buffer_size","leaf_buffer_width","leaf_pe_freq",\
                "blade_mesh_dim","blade_mesh_bw","blade_buffer_size","blade_buffer_bw",\
                    "node_mesh_dim","node_mesh_bw","node_buffer_size","node_buffer_bw", "T_top(s)", "E_total(J)"])
    df["T_top(s)"] = pd.to_numeric(df["T_top(s)"])
    df["E_total(J)"] = pd.to_numeric(df["E_total(J)"])
    print("filtering good designs...")
    df = filter_bottom_percent(df, ["T_top(s)", "E_total(J)"], 0.05)
    #save the filtered data
    df.to_csv("cannon_gemm_para_sweep_dumped_data_filtered.csv", index=False)
    print("filtered data saved")

with open("cannon_gemm_para_sweep_dumped_data_filtered.csv", "r") as fp:
    print("loading filtered data...")
    reader = csv.reader(fp)
    next(reader)
    data = list(reader)
# list_T = np.array([float(row[-2]) for row in data])
# list_E = np.array([float(row[-1]) for row in data])
# print("finding good designs...")
# least_T_idx = np.argmin(list_T)
# least_T = list_T[least_T_idx]
# least_T_E = list_E[least_T_idx]
# least_E_idx = np.argmin(list_E)
# least_E = list_E[least_E_idx]
# least_E_T = list_T[least_E_idx]

# data_good_designs = []
# for row in data:
#     T = float(row[-2])
#     E = float(row[-1])
#     if E <= least_T_E and T <= least_E_T:
#         data_good_designs.append(row)
# list_good_T = []
# list_good_E = []
# for T,E in zip(list_T, list_E):
#     if E <= least_T_E and T <= least_E_T:
#         list_good_T.append(T)
#         list_good_E.append(E)

print(f"number of good designs: {len(data)}")
print("ploting good designs...")
# list_good_T = np.array(list_good_T)
# list_good_E = np.array(list_good_E)
# matplotlib.use('agg')
# fig1 = plt.figure("Energy vs Latency")
# plt.scatter(list_good_T, list_good_E, s=1)
# plt.xlabel('Latency (s)')
# plt.ylabel('Energy (J)')
# plt.savefig('fig_latency_energy.pdf')

m,k,n = 90*200*64,90*200*64,90*200*64
# fig2 = plt.figure("OPs/J vs OPs/s")
# op_per_J = m*k*n/list_good_E
# op_per_s = m*k*n/list_good_T
# plt.scatter(op_per_J, op_per_s, s=1)
# plt.xlabel('Throughput (OP/s)')
# plt.ylabel('Energy Efficiency (OP/J)')
# plt.savefig('fig_efficiency_vs_throughput.pdf')


# list_throughput = m*k*n/list_good_T*1e-9
# list_power = list_good_E/list_good_T
# fig3 = plt.figure("Throughput vs Power")
# plt.scatter(list_throughput, list_power, s=1)
# plt.xlabel('Throughput (TOPS/s)')
# plt.ylabel('Power (W)')
# plt.savefig('fig_throughput_power.pdf')

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

ALPHA = 0.3
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
    shapes = ['o', 's', 'D', '^', 'v', 'p', 'P']
    fig4 = plt.figure(f"Efficiency vs Throughput {list_param_name[index]}")
    for counter, (key, value) in enumerate(buckets.items()):
        # list_T = np.array([float(row[-2]) for row in value])
        # list_E = np.array([float(row[-1]) for row in value])
        TE_pairs = [(float(row[-2]), float(row[-1])) for row in value]
        TE_pairs = filter_unique(TE_pairs)
        list_T = np.array([TE_pair[0] for TE_pair in TE_pairs])
        list_E = np.array([TE_pair[1] for TE_pair in TE_pairs])
        op_per_J = m*k*n/list_E
        op_per_s = m*k*n/list_T
        plt.scatter(op_per_s, op_per_J, s=20, color=colors[counter%len(colors)], edgecolor=None, alpha=ALPHA, marker=shapes[counter%len(shapes)], label=f"{list_param_name[index]} = {key}")

    plt.xlabel('Throughput (OP/s)')
    plt.ylabel('Energy Efficiency (OP/J)')
    plt.legend(loc="upper left")
    plt.savefig(f'fig_efficiency_vs_throughput_{list_param_name[index]}.pdf')
    matplotlib.pyplot.close()

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
    shapes = ['o', 's', 'D', '^', 'v', 'p', 'P']
    fig4 = plt.figure(f"Energy vs Latency {list_param_name[index]}")
    for counter, (key, value) in enumerate(buckets.items()):
        TE_pairs = [(float(row[-2]), float(row[-1])) for row in value]
        TE_pairs = filter_unique(TE_pairs)
        list_T = np.array([TE_pair[0] for TE_pair in TE_pairs])
        list_E = np.array([TE_pair[1] for TE_pair in TE_pairs])
        plt.scatter(list_T, list_E, s=20, color=colors[counter%len(colors)], edgecolor=None,  alpha=ALPHA, marker=shapes[counter%len(shapes)], label=f"{list_param_name[index]} = {key}")

    plt.xlabel('Latency (s)')
    plt.ylabel('Energy (J)')
    plt.legend(loc="upper left")
    plt.savefig(f'fig_energy_vs_latency_{list_param_name[index]}.pdf')
    matplotlib.pyplot.close()

for index in range(12):
    bucket_plot_efficiency_vs_throughput(data, index)
    bucket_plot_energy_vs_latency(data, index)