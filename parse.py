from tabulate import tabulate
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from params import *

cur_dir = os.getcwd()
#path = f"{cur_dir}/gemmforge_remote/stdout_only_run.txt"
path = f"{cur_dir}/stdout.txt"

# The simple state machine
# Initial ->(1) Kernel Region ->(2) Initial
# For change (1) read: ==PROF== Connected to process
# In Kernel region we need to read:
# Dense x Dense kernel took ...
# Then
# Dense x Sparse kernel took ..., or Sparse x Dense kernel took ...
# Then we return to the initial state with
# ==PROF== Disconnected from process
#Peak Memory Bandwidth: 176.032GB/s
#Peak Floating-Point Performance: 270385 GFLOPS

report_dense_sparse = []
report_sparse_dense = []

# LOPS = sockets * (cores per socket) * (number of clock cycles per second) * (number of floating point operations per cycle)
# 



#row_a = 56
#col_a = 9
#row_b = 9
#col_b = 9

FLOAT_SIZE = 4

def get_load_store_size(b_el_count):
    load_store = 0
    load_store += b_el_count
    load_store += (row_a * col_a)
    # Nsight compute says to remvoe this
    # load_store + ((row_c * col_c) * 1)
    #1 thread loads 1 row of a, 1 roww of b and 1 row of c

    # Adding C to end result, row_c = row_a, col_c = col_b
    if Beta != 0.0:
        Flops += row_a * col_b;

    load_store *= FLOAT_SIZE
    return load_store

def calculate_ops_dense():
  #Flops = (col_a + (row_a - 1)) * col_a * col_b;
  #FMA of 1 row = (2 * col_a * col_b)
  #Done for every row (row_a) *
  ops = (row_a) * (2 * col_a * col_b)
  ops += row_c * col_c
  Flops = ops 

  if Alpha != 1.0:
    Flops += row_a * col_b;

  # Adding C to end result, row_c = row_a, col_c = col_b
  if Beta != 0.0:
    Flops += row_a * col_b;

  load_store =  get_load_store_size(row_b * col_b)

  return Flops, load_store

"""
def calculate_ops_dense():
    # Matrix mul (ffma = 2 op)
    # This is Ravil's calculation of FLOP/s
    ops = (row_a) * (2 * (col_a - 1)) * col_b
    #2 * (k - 1) * m * n
    # I believe it should be this?
    # ops = (row_a) * (2 * col_a) * col_b
    # ops += row_c * col_c

    # Load A, B, C, store C, collective load of B, each thread then needs a row of A
    # and a row of C
    load_store =  get_load_store_size(row_b * col_b)
    #return (ops, load_store)
    return ops, load_store
"""

sparsity = 0.25

def calculate_ops_sparse_dense(typestr):
    if typestr == "full":
        return calculate_ops_dense()
    elif typestr == "random":
        el_count = int(sparsity * (row_a * col_a))
        #ops = 2 * el_count
        ops = (0.25 * row_a * col_a) * 2 * col_b
        ops += row_a * col_a
        load_store =  get_load_store_size(el_count)
        return (ops, load_store)
    else:
        raise Exception("TODO for type: " + typestr)

def calculate_ops_dense_sparse(typestr):
    if typestr == "band":
        ops = 0
        ops += 2 * col_a * 2 * 2 # first and last row
        ops += (row_a - 2) * col_a * 3 * 2 # for every other row
        ops += row_a * col_a
        load_store =  get_load_store_size(3*row_b - 2)
        return (ops, load_store)
    elif typestr == "full":
        return calculate_ops_dense()
    elif typestr == "single_row":
        # Every row of A will be multiplied with a single row of B
        ops = row_a * col_a * 1 * 2
        ops += row_a * col_a
        load_store =  get_load_store_size(col_b)
        return (ops, load_store)
    elif typestr == "single_column":
        # Every row of A will be multiple with 1 element
        ops = row_a * col_a * 1 * 2
        ops += row_a * col_a
        load_store =  get_load_store_size(row_b)
        return (ops, load_store)
    elif typestr == "random":
        el_count = int(sparsity * (row_b * col_b))
        ops = row_a * col_a * col_b * sparsity * 2
        ops += row_a * col_a
        load_store =  get_load_store_size(el_count)
        return (ops, load_store)
    elif typestr == "chequered":
        a = row_b // 2 + (row_b % 2)
        b = row_b // 2
        c = col_b // 2 + (col_b % 2)
        d = col_b // 2
        el_count = a*b + c*d
        ops = row_a * el_count * 2
        ops += row_a * col_a
        load_store =  get_load_store_size(el_count)
        return (ops, load_store)
    else:
        raise Exception(typestr + " is undefined")

with open(path, "r") as file:
    identifier = ""
    dense_time = 0.0
    sparse_time = 0.0
    speed_up = 0.0
    op_diff = 0.0
    load_diff = 0.0
    flops_per_byte_dd = 0.0
    flops_per_byte_ds = 0.0
    dense_sparse_type = ""
    sparse_dense_type = ""
    gemmforge_flops_per_el = 0.0
    my_flops_per_el = 0.0
    state = "initial"
    num_items = 0.0

    i = 1
    for line in file:
        print(i)
        if "Number of elements:" in line:
            el_count = int(line.split(":")[1].strip())
            num_items = float(el_count)
            i += 1
            continue
        if "Dense FLOP/s per element from gemmforge:" in line:
            gemmforge_flops_per_el = int(line.split(":")[1].strip())
            gemmforge_flops_per_el = float(gemmforge_flops_per_el)
            i += 1
            continue

        if state == "initial" and "Gemm-Type:" in line:
            l = line.split("Type:")
            identifier = l[-1]
            # inside the identifier there has to be smth like: dense_sparse_At_mul_B_full_compiler_time_value
            # A{T if transA else NT}_{a_type}_B{T if transB else NT}_DenseXDense
            # or
            # A{T if transA else NT}_B{T if transB else NT}_{b_type}_DenseXDense
            tokens = identifier.split("_")
            assert(len(tokens) >= 4)
            if "DenseXSparse" in identifier:
                dense_sparse_type = tokens[2]
                if dense_sparse_type == "single":
                    dense_sparse_type += "_" + tokens[3]
                assert(dense_sparse_type != "")
                print(i, "DS: ", dense_sparse_type)
            elif "SparseXDense" in identifier:
                sparse_dense_type = tokens[1]
                if sparse_dense_type == "single":
                    sparse_dense_type += "_" + tokens[2]
                assert(sparse_dense_type != "")
                print(i, "SD: ", sparse_dense_type)
            state = "kernel"
            print(i, " initial -> kernel")
            i += 1
            continue
        elif state == "kernel" and "Dense x Dense kernel took" in line:
            duration = line.split("Dense x Dense kernel took ")[1][:-3]
            dense_time = float(duration)
            state = "kernel-2"
            print(i, " kernel -> kernel-2")
            i += 1
            continue
        elif state == "kernel-2" and "Dense x Sparse kernel took" in line:
            duration = line.split("Dense x Sparse kernel took ")[1][:-3]
            sparse_time = float(duration)
            state = "write-ds"
            print(i, " kernel-2 -> write-ds")
            i += 1
            continue
        elif state == "kernel-2" and "Sparse x Dense kernel took" in line:
            duration = line.split("Sparse x Dense kernel took ")[1][:-3]
            sparse_time = float(duration)
            state = "write-sd"
            print(i, " kernel-2 -> write-sd")
            i += 1
            continue
        elif state == "write-ds" and "Freeing device memory" in line: 
            speed_up =  dense_time / sparse_time
            dd_ops, dd_load_store = calculate_ops_dense()
            print(dense_sparse_type)
            ds_ops, ds_load_store = calculate_ops_dense_sparse(dense_sparse_type)
            op_diff = dd_ops / ds_ops
            load_store_diff = dd_load_store / ds_load_store
            flops_per_byte_dd = float(dd_ops) / float(dd_load_store)
            flops_per_byte_ds = float(ds_ops) / float(ds_load_store)
            speed_up_per = 100*(dense_time - sparse_time) / dense_time
            dd_flops = float(num_items) * float(dd_ops) * 1e-9 / (float(dense_time)*1e-3)
            dd_flops = 1e-6 * (float(dd_ops) / (float(dense_time) / float(num_items)))
            gemmforge_dd_flops = float(num_items) * float(gemmforge_flops_per_el) * 1e-9 / (float(dense_time)*1e-3)
            total_bytes = dd_load_store
            print(num_items, " * ", dd_ops, " * 1e3 / (", dense_time, " * 1e9) = ", dd_flops)
            print(gemmforge_flops_per_el, " =? ", dd_ops)
            if num_items == 0.0:
                raise Exception("Number items should have been found")
            ds_flops = float(num_items) * float(ds_ops) * 1e-9 / (float(sparse_time)*1e-3)
            report_dense_sparse.append([identifier, 
                                        round(dense_time, 4), 
                                        round(sparse_time, 4), 
                                        round(speed_up, 4),
                                        #round(speed_up_per, 2),
                                        round(op_diff, 4),
                                        round(load_store_diff, 4),
                                        round(flops_per_byte_dd, 4),
                                        round(flops_per_byte_ds, 4),
                                        round(dd_flops, 4),
                                        round(ds_flops, 4),
                                        round(gemmforge_dd_flops, 4),
                                        ])
            state = "return"
            i += 1
            print(i, " write-ds -> return")
            continue
        elif state == "write-sd" and "Freeing device memory" in line:
            speed_up =  dense_time / sparse_time
            dd_ops, dd_load_store = calculate_ops_dense()
            print(sparse_dense_type)
            sd_ops, sd_load_store = calculate_ops_sparse_dense(sparse_dense_type)
            op_diff = dd_ops / sd_ops
            load_store_diff = dd_load_store / sd_load_store
            flops_per_byte_dd = dd_ops / (dd_load_store)
            flops_per_byte_sd = sd_ops / (sd_load_store)
            speed_up_per = 100*(dense_time - sparse_time) / dense_time
            report_sparse_dense.append([identifier, 
                                        round(dense_time, 6), 
                                        round(sparse_time, 6), 
                                        round(speed_up, 6),
                                        round(speed_up_per, 6),
                                        round(op_diff, 6),
                                        round(load_store_diff, 6),
                                        round(flops_per_byte_dd, 6),
                                        round(flops_per_byte_sd, 6),
                                        ])
            state = "return"
            i += 1
            print(i, " write-sd -> return")
            continue
        elif state == "return":
            identifier = ""
            dense_time = 0.0
            sparse_time = 0.0
            speed_up = 0.0
            state = "initial"
            substate = ""
            op_diff = 0.0
            load_store_diff = 0.0
            flops_per_byte_dd = 0.0
            flops_per_byte_ds = 0.0
            sparse_dense_type = ""
            dense_sparse_type = ""
            num_items = 0.0
            print(i, " return -> initial")
            i += 1
            continue
        else:
            i += 1

report_dense_sparse = list(sorted(report_dense_sparse, key = lambda x: x[0]))
report_sparse_dense = list(sorted(report_sparse_dense, key = lambda x: x[0]))
report_dense_sparse = list(filter(lambda x: "_ctv" not in x[0], report_dense_sparse))
report_sparse_dense = list(filter(lambda x: "_ctv" not in x[0], report_sparse_dense))

print(
    tabulate(report_dense_sparse, 
             headers=["Identifier", 
                      "DD Time", 
                      "DS Time",
                      "Speed-up",
                      #"%",
                      "Flop. Ceil",
                      "LS Ceil",
                      "DD Flop/b",
                      "DS Flop/b",
                      "DD GFlop/s",
                      "DS GFlop/s",
                      "Ge. DD GFlop/s"],
             tablefmt="github"))
print(
    tabulate(report_sparse_dense,
             headers=["Identifier", 
                      "DD Time", 
                      "DS Time",
                      "Speed-up",
                      "%",
                      "Flop. Ceil",
                      "LS Ceil",
                      "DD GFlop/b",
                      "DS GFlop/b"],
             tablefmt="github"))

p_ds = pd.DataFrame(data=report_dense_sparse, columns=[
                      "Identifier",
                      "DD Time", 
                      "DS Time",
                      "Speed-up",
                      #"%",
                      "Flop. Ceil",
                      "LS Ceil",
                      "DD Flop/b",
                      "DS Flop/b",
                      "DD GFlop/s",
                      "DS GFlop/s",
                      "Gemmforge DD GFlop/s"
                      ])

p_sd = pd.DataFrame(data=report_sparse_dense, columns=[
                      "Identifier",
                      "DD Time", 
                      "DS Time",
                      "Speed-up",
                      "%",
                      "Flop. Ceil",
                      "LS Ceil",
                      "DD Flop/b",
                      "DS Flop/b"
                      ])

cov = p_ds[[
    "Speed-up",
    "Flop. Ceil",
    "LS Ceil",
    "DD Flop/b",
    "DS Flop/b"
    ]].cov()
print(cov)
heatmap = sns.heatmap(cov, annot=True, fmt=".2f")
fig = heatmap.get_figure()
fig.tight_layout()
fig.savefig(f"{cur_dir}/heatmap-cov-ds.png") 
fig.clear()

corr =  p_ds[[
    "Speed-up",
    "Flop. Ceil",
    "LS Ceil",
    "DD Flop/b",
    "DS Flop/b"
    ]].corr()
print(corr)
heatmap = sns.heatmap(corr, annot=True, fmt=".2f")
fig = heatmap.get_figure()
fig.tight_layout()
fig.savefig(f"{cur_dir}/heatmap-corr-ds.png") 
fig.clear()

cov = p_sd[[
    "Speed-up",
    "Flop. Ceil",
    "LS Ceil",
    "DD Flop/b",
    "DS Flop/b"
    ]].cov()
print(cov)
heatmap = sns.heatmap(cov, annot=True, fmt=".2f")
fig = heatmap.get_figure()
fig.tight_layout()
fig.savefig(f"{cur_dir}/heatmap-cov-sd.png") 
fig.clear()

corr =  p_sd[[
    "Speed-up",
    "Flop. Ceil",
    "LS Ceil",
    "DD Flop/b",
    "DS Flop/b"
    ]].corr()
print(corr)
heatmap = sns.heatmap(corr, annot=True, fmt=".2f")
fig = heatmap.get_figure()
fig.tight_layout()
fig.savefig(f"{cur_dir}/heatmap-corr-sd.png") 


#benchmark says 192 mem bandwidth
#website says 5.5 Tflops?
peakMemoryBandwidthTheo = 176.032 # GB /s
peakFLOPTheo  = 4329.47 # GFlop /s

# Experimental values from NVPROF
peakMemoryBandwidth  = 175
peakFLOP = 2918

lookupPeakMemoryBandwidth = 192
lookupPeakFLOP = 5500

fig.clear()

dd_points = p_ds[["DD Flop/b", "DD GFlop/s"]]
ds_points = p_ds[["DS Flop/b", "DS GFlop/s"]]
print(dd_points)

def plot_roofline(peak_memory_bandwidth, peak_floating_point_perf, title):
    roof = lambda val: min(peak_floating_point_perf, (peak_memory_bandwidth * val))
    xpts = np.linspace(0, 40, 250)
    plt.plot(xpts, [roof(x) for x in xpts])
    plt.scatter(x=dd_points["DD Flop/b"],y=dd_points["DD GFlop/s"], c="blue")
    #plt.scatter(x=ds_points["DS Flop/b"],y=ds_points["DS GFlop/s"], c="red")
    plt.title(title)
    # Show the plot
    plt.show()
    
plot_roofline(peakMemoryBandwidth, peakFLOP, "Experimental from Nsight Compute")
plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo, "Theoretical from CUDA Library Calls")
plot_roofline(lookupPeakMemoryBandwidth, lookupPeakFLOP, "Theoretical from Web")