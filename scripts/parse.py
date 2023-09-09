from copy import deepcopy
import math
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from params import *

# path = f"{data_dir}/gemmforge_remote/stdout_only_run.txt"


# The simple state machine
# Initial ->(1) Kernel Region ->(2) Initial
# For change (1) read: ==PROF== Connected to process
# In Kernel region we need to read:
# Dense x Dense kernel took ...
# Then
# Dense x Sparse kernel took ..., or Sparse x Dense kernel took ...
# Then we return to the initial state with
# ==PROF== Disconnected from process
# Peak Memory Bandwidth: 176.032GB/s
# Peak Floating-Point Performance: 270385 GFLOPS

FLOAT_SIZE = 4


def get_load_store_size(b_el_count, ctv):
    load_store = 0

    # If adressing is none then the matrix is loaded only 1 time in the whole batch
    # Read A
    if adressingA != "none":
        load_store += row_a * col_a

    # Read B
    if adressingB != "none" and not ctv:
        load_store += b_el_count

    # Write C
    if adressingC != "none":
        load_store += row_c * col_c

    # If Beta is not 0 then we need to read C
    if Beta != 0.0:
        load_store += row_c * col_c

    load_store *= FLOAT_SIZE
    return load_store


def calculate_ops_dense(ctv=False):
    # Flops = (col_a + (row_a - 1)) * col_a * col_b;
    # FMA of 1 row = (2 * col_a * col_b)
    # Done for every row (row_a) *

    Flops = (row_a) * (2 * col_a * col_b)
    #Flops -= col_a * col_b # First row

    if Alpha != 1.0:
        Flops += row_c * col_c

    #Flops += row_a * col_a

    # Adding C to end result, row_c = row_a, col_c = col_b
    if Beta != 0.0:
        Flops += 2 * row_c * col_c

    load_store = get_load_store_size(row_b * col_b, ctv)

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

def calculate_ops_dense_sparse(typestr, ctv):
    if typestr == "band":
        ops = 0
        ops += 2 * col_a * 2 * 2  # first and last row
        ops += (row_a - 2) * col_a * 3 * 2  # for every other row
        if Alpha != 1.0:
            ops += row_c * col_c
        if Beta != 0.0:
            ops += 2 * row_c * col_c
        #ops += row_a * col_a
        #ops -= row_a * col_a
        load_store = get_load_store_size(3*col_b - 2, ctv)
        return (ops, load_store)
    elif typestr == "full":
        return calculate_ops_dense(ctv)
    elif typestr == "random":
        el_count = int(sparsity * (row_b * col_b))
        ops = row_a * el_count * 2
        if Alpha != 1.0:
            ops += row_c * col_c
        if Beta != 0.0:
            ops += 2 * row_c * col_c
        #ops += row_a * col_a
        #ops -= row_a * col_a
        load_store = get_load_store_size(el_count, ctv)
        return (ops, load_store)
    elif typestr == "chequered":
        a = row_b // 2 + (row_b % 2)
        b = row_b // 2
        c = col_b // 2 + (col_b % 2)
        d = col_b // 2
        el_count = a*b + c*d
        ops = row_a * el_count * 2
        if Alpha != 1.0:
            ops += row_c * col_c
        if Beta != 0.0:
            ops += 2 * row_c * col_c
        #ops += row_a * col_a
        #ops -= row_a * col_a
        load_store = get_load_store_size(el_count, ctv)
        return (ops, load_store)
    else:
        raise Exception(typestr + " is undefined")


pd_ds_dataframes = list()
pd_ds_ctv_dataframes = list()

for i in range(runs):
    path = f"{stdout_dir}/run{i}.txt"
    report_dense_sparse = []
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
        cublas_time = 0.0
        cusparse_time = 0.0

        i = 1
        for line in file:
            #print(i)
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
                assert (len(tokens) >= 4)
                if "DenseXSparse" in identifier:
                    dense_sparse_type = tokens[2]
                    if dense_sparse_type == "single":
                        dense_sparse_type += "_" + tokens[3]
                    assert (dense_sparse_type != "")
                    #print(i, "DS: ", dense_sparse_type)
                elif "SparseXDense" in identifier:
                    sparse_dense_type = tokens[1]
                    if sparse_dense_type == "single":
                        sparse_dense_type += "_" + tokens[2]
                    assert (sparse_dense_type != "")
                    #print(i, "SD: ", sparse_dense_type)
                state = "kernel"
                #print(i, " initial -> kernel")
                i += 1
                continue
            elif state == "kernel" and "Dense x Dense kernel took" in line:
                duration = line.split("Dense x Dense kernel took ")[1][:-3]
                dense_time = float(duration)
                state = "kernel-2"
                #print(i, " kernel -> kernel-2")
                i += 1
                print([dense_time, 0.0, 0.0, 0.0])
                continue
            elif state == "kernel-2" and "Dense x Sparse kernel took" in line:
                duration = line.split("Dense x Sparse kernel took ")[1][:-3]
                sparse_time = float(duration)
                state = "kernel-3"
                #print(i, " kernel-2 -> kernel-3")
                print([dense_time, sparse_time, 0.0, 0.0])
                i += 1
                continue
                """
                elif state == "kernel-2" and "Sparse x Dense kernel took" in line:
                    duration = line.split("Sparse x Dense kernel took ")[1][:-3]
                    sparse_time = float(duration)
                    state = "write-sd"
                    print(i, " kernel-2 -> write-sd")
                    i += 1
                    continue
                """
            elif state == "kernel-3" and "cuBlas DxD kernel took" in line:
                duration = line.split("cuBlas DxD kernel took ")[1][:-3]
                cublas_time = float(duration)
                state = "kernel-4"
                #print(i, " kernel-3 -> kernel-4")
                i += 1
                print([dense_time, sparse_time, cublas_time, 0.0])
                continue
            elif state == "kernel-4" and "cuSparse DxS kernel" in line:
                duration = line.split("cuSparse DxS kernel took ")[1][:-3]
                cusparse_time = float(duration)
                state = "write-ds"
                #print(i, " kernel-4 -> write-ds")
                i += 1
                print([dense_time, sparse_time, cublas_time, cusparse_time])
                continue
            elif state != "initial" and "Freeing device memory" in line:
                speed_up = 0.0 if sparse_time == 0.0 else dense_time / sparse_time
                dd_ops, dd_load_store = calculate_ops_dense(False)
                #print(dense_sparse_type)
                if "_ctv" in identifier:
                    ds_ops, ds_load_store = calculate_ops_dense_sparse(
                        dense_sparse_type, True)
                else:
                    ds_ops, ds_load_store = calculate_ops_dense_sparse(
                        dense_sparse_type, False)
                op_diff = float(dd_ops) / float(ds_ops)
                load_store_diff = float(dd_load_store) / float(ds_load_store)
                flops_per_byte_dd = float(dd_ops) / float(dd_load_store)
                flops_per_byte_ds = float(ds_ops) / float(ds_load_store)
                speed_up_per = 100*(dense_time - sparse_time) / dense_time
                dd_flops = (float(num_items) * float(dd_ops)) / (float(dense_time)*1e6)
                if num_items == 0.0:
                    raise Exception("Number items should have been found")
                ds_flops = (float(num_items) * float(ds_ops)) / (float(sparse_time)*1e6)
                report_dense_sparse.append([identifier,
                                            dense_time,
                                            sparse_time,
                                            cublas_time,
                                            cusparse_time,
                                            speed_up,
                                            # round(speed_up_per, 2),
                                            op_diff,
                                            load_store_diff,
                                            flops_per_byte_dd,
                                            flops_per_byte_ds,
                                            dd_flops,
                                            ds_flops,
                                            ])
                i += 1
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
                cublas_time = 0.0
                cusparse_time = 0.0
                #print(i, " write-ds -> return")
                continue
            elif state == "write-sd" and "Freeing device memory" in line:
                """
                speed_up = dense_time / sparse_time
                dd_ops, dd_load_store = calculate_ops_dense()
                print(sparse_dense_type)
                sd_ops, sd_load_store = calculate_ops_sparse_dense(
                    sparse_dense_type)
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
                """
                state = "return"
                i += 1
                #print(i, " write-sd -> return")
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
                cublas_time = 0.0
                cusparse_time = 0.0
                #print(i, " return -> initial")
                i += 1
                continue
            else:
                i += 1

    """
    print(
        tabulate(report_dense_sparse,
                headers=["Identifier",
                        "DD Time",
                        "DS Time",
                        "cuBlas Time",
                        "cuSparse Time",
                        "Speed-up",
                        "Flop. Ceil",
                        "LS Ceil",
                        "DD Flop/b",
                        "DS Flop/b",
                        "DD GFlop/s",
                        "DS GFlop/s"],
                tablefmt="github"))
    """

    report_dense_sparse_both = list(sorted(report_dense_sparse, key=lambda x: x[0]))
    #report_sparse_dense = list(sorted(report_sparse_dense, key=lambda x: x[0]))
    report_dense_sparse = list(
        filter(lambda x: not "_ctv" in x[0], report_dense_sparse_both))
    #report_sparse_dense = list(
    #    filter(lambda x: "_ctv" not in x[0], report_sparse_dense))
    report_dense_sparse_ctv = list(
        filter(lambda x: "_ctv" in x[0], report_dense_sparse_both))

    """
    print(
        tabulate(report_dense_sparse,
                headers=["Identifier",
                        "DD Time",
                        "DS Time",
                        "cuBlas Time",
                        "cuSparse Time",
                        "Speed-up",
                        "Flop. Ceil",
                        "LS Ceil",
                        "DD Flop/b",
                        "DS Flop/b",
                        "DD GFlop/s",
                        "DS GFlop/s"],
                tablefmt="github"))
    print(
        tabulate(report_dense_sparse_ctv,
                headers=["Identifier",
                        "DD Time",
                        "DS Time",
                        "cuBlas Time",
                        "cuSparse Time",
                        "Speed-up",
                        # "%",
                        "Flop. Ceil",
                        "LS Ceil",
                        "DD Flop/b",
                        "DS Flop/b",
                        "DD GFlop/s",
                        "DS GFlop/s"],
                tablefmt="github"))
    """

    tmp1 = pd.DataFrame(data=deepcopy(report_dense_sparse), columns=[
        "Identifier",
        "DD Time",
        "DS Time",
        "cuBlas Time",
        "cuSparse Time",
        "Speed-up",
        # "%",
        "Flop. Ceil",
        "LS Ceil",
        "DD Flop/b",
        "DS Flop/b",
        "DD GFlop/s",
        "DS GFlop/s"
    ])
    tmp1 = tmp1.sort_values(by="Identifier").copy()
    print(tmp1)
    pd_ds_dataframes.append(tmp1.copy())
    tmp2 = pd.DataFrame(data=deepcopy(report_dense_sparse_ctv), columns=[
        "Identifier",
        "DD Time",
        "DS Time",
        "cuBlas Time",
        "cuSparse Time",
        "Speed-up",
        # "%",
        "Flop. Ceil",
        "LS Ceil",
        "DD Flop/b",
        "DS Flop/b",
        "DD GFlop/s",
        "DS GFlop/s"
    ])
    tmp2 = tmp2.sort_values(by="Identifier").copy()
    print(tmp2)
    pd_ds_ctv_dataframes.append(deepcopy(tmp2))

#print(len(pd_ds_dataframes))
#print("---------------------")

# This takes the average and covariance of 2 pandas data frames,
# Runtime for denses parse dense sparse with ctv 

p_ds_sum = pd_ds_dataframes[0].copy()

# Take average except the str row (ops on str not supported sadly)
for df in pd_ds_dataframes[1:]:
    p_ds_sum.iloc[:, 1:] += df.iloc[:, 1:].copy()

p_ds_sum.iloc[:, 1:] = p_ds_sum.iloc[:, 1:].copy() / float(runs)
p_ds = p_ds_sum.copy()
print(p_ds)
#print(p_ds)
#print("---------------------")
p_ds_ctv_sum = pd_ds_ctv_dataframes[0].copy()

# Take average except the str row (ops on str not supported sadly)
for df in pd_ds_ctv_dataframes[1:]:
    p_ds_ctv_sum.iloc[:, 1:] += df.iloc[:, 1:].copy()

p_ds_ctv_sum.iloc[:, 1:] = p_ds_ctv_sum.iloc[:, 1:].copy() / float(runs)
p_ds_ctv = p_ds_ctv_sum.copy()
print(p_ds_ctv)
#print(p_ds_ctv)
#print("======================")
p_ds_var_dist = pd_ds_dataframes[0].copy()
p_ds_var_dist.iloc[:, 1:] -= p_ds_sum.iloc[:, 1:].copy()
p_ds_var_dist.iloc[:, 1:] = p_ds_var_dist.iloc[:, 1:].copy().pow(2)
for df in pd_ds_dataframes[1:]:
    tmp = df.iloc[:, 1:].copy() - p_ds.iloc[:, 1:].copy()
    p_ds_var_dist += tmp.pow(2).copy()
p_ds_var = p_ds_var_dist.iloc[:, 1:].copy() / float(runs)

p_ds_ctv_var_dist = pd_ds_ctv_dataframes[0].copy()
p_ds_ctv_var_dist.iloc[:, 1:] -= p_ds_ctv.iloc[:, 1:].copy()
p_ds_ctv_var_dist.iloc[:, 1:] = p_ds_ctv_var_dist.iloc[:, 1:].copy().pow(2)
for df in pd_ds_ctv_dataframes[1:]:
    tmp = df.iloc[:, 1:].copy() - p_ds_ctv.iloc[:, 1:].copy()
    p_ds_ctv_var_dist += tmp.pow(2).copy()
p_ds_ctv_var = p_ds_ctv_var_dist.iloc[:, 1:].copy() / float(runs)

#p_ds_ctv_var = sum([(x - p_ds_ctv).pow(2) for x in pd_ds_ctv_dataframes]) / runs

#print(p_ds)
#print("~~~~~~~~~~~~~~~~~~~")
#print(p_ds_ctv)
print(p_ds_var)
print(p_ds_ctv_var)
#raise Exception("uwu")

#fig.clear()
dd_points = p_ds[["DD Flop/b", "DD GFlop/s"]]
ds_points = p_ds[["DS Flop/b", "DS GFlop/s"]]
ds_points_ctv = p_ds_ctv[["DS Flop/b", "DS GFlop/s"]]
dd_points_ctv = p_ds_ctv[["DD Flop/b", "DD GFlop/s"]]



def round_up_to_power_of_ten(n):
    if n == 0:
        return 1
    elif n < 0:
        return -round_up_to_power_of_ten(-n)
    else:
        power = math.ceil(math.log10(n))
        return 10 ** power

def plot_roofline(peak_memory_bandwidth, peak_floating_point_perf, title, ds_points, dd_points, p_ds, addname):
    plt.clf()
    done = [False for _ in b_matrix_types]
    txt = ["" for _ in p_ds["Identifier"]]
    def roof(val): return min(peak_floating_point_perf,
                              (peak_memory_bandwidth * val))
    xpts = np.linspace(0, 40, 250)
    plt.plot(xpts, [roof(x) for x in xpts], label="Roofline")
    xit = 0
    for x in p_ds["Identifier"]:
        for i in range(len(b_matrix_types)):
            if done[i] == False:
                if b_matrix_types[i] in x:
                    done[i] = True
                    txt[xit] = b_matrix_types[i].capitalize()
        xit += 1
    plt.scatter(x=ds_points["DS Flop/b"],y=ds_points["DS GFlop/s"], c=sparse_rose, s=10, label="Dense-Sparse")
    for i, (xi, yi) in enumerate(zip(ds_points["DS Flop/b"], ds_points["DS GFlop/s"])):
        roofline = roof(xi)
        implementation = yi
        percentage = implementation*100.0 / roofline
        if txt[i] != "":
            txt[i] = txt[i] + " " + str(round(percentage,2)) + "%"
        plt.annotate(txt[i] , (xi, yi), textcoords="offset points", xytext=(8,-2), ha='left', size=9)

    (xi, yi) = (dd_points["DD Flop/b"][0], dd_points["DD GFlop/s"][0])
    roofline = roof(xi)
    implementation = yi
    percentage = implementation*100.0 / roofline
    t = "Dense" + " " + str(round(percentage,2)) + "%"
    if addname == "":
        plt.annotate(t, (xi, yi), textcoords="offset points", xytext=(8,-12), ha='left', size=9)
    else:
        plt.annotate(t, (xi, yi), textcoords="offset points", xytext=(8,-2), ha='left', size=9)

    plt.scatter(x=dd_points["DD Flop/b"],
                y=dd_points["DD GFlop/s"], c=dense_blue, s=10, label="Dense-Dense")
    ymax = max(max(dd_points["DD GFlop/s"]), max(ds_points["DS GFlop/s"]))
    xmax =  max(max(dd_points["DD Flop/b"]), max(ds_points["DS Flop/b"]))
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim((0, round_up_to_power_of_ten(ymax)))
    plt.xlim((0, round_up_to_power_of_ten(xmax)))
    plt.title(title)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('Operational Intensity (Flop/byte)')
    plt.ylabel('Performance (GFlops/second)')
    # Show the plot
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{data_dir}/plots/dense-sparse{addname}-roofline-A{row_a}x{col_a}-B{row_b}x{col_b}-C{row_c}x{col_c}-alpha{Alpha}-beta{Beta}.pdf.pdf")
    plt.clf()

#plot_roofline(peakMemoryBandwidth, peakFLOP,
#              "Experimental from Nsight Compute")
if save_plots:
    plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo,
              "Roofline Model for Dense-Dense and Dense-Sparse Kernels", ds_points=ds_points, dd_points=dd_points, p_ds=p_ds, addname="")
    plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo,
              "Roofline Model for Dense-Dense and Dense-Sparse Kernels\nWith Compile Time Matrix Values", ds_points=ds_points_ctv,
                dd_points=dd_points_ctv, p_ds=p_ds_ctv, addname="-ctv")
#plot_roofline(lookupPeakMemoryBandwidth,
#              lookupPeakFLOP, "Theoretical from Web")

def plot_in_a_grid(p_ds, addname):
    times_only = p_ds[[
        "Identifier",
        "DD Time",
        "DS Time",
        "cuBlas Time",
        "cuSparse Time"
    ]]

    def modify_row(row):
        row["Identifier"] = row["Identifier"][:-1]
        row['DD Time'] = row['cuBlas Time'] / row['DD Time']
        row['DS Time'] = row['cuBlas Time'] / row['DS Time']
        row['cuSparse Time'] = row['cuBlas Time'] / row['cuSparse Time']
        row['cuBlas Time'] = row['cuBlas Time'] / row['cuBlas Time']
        return row

    relative_speed_up = times_only.apply(modify_row, axis=1)
    print(relative_speed_up)

    groups = list()
    max_group_len = 0
    for b_type in b_matrix_types:
        mask = [b_type in x for x in relative_speed_up['Identifier']]
        filtered_df = relative_speed_up[[b_type in x for x in relative_speed_up['Identifier']]]
        if len(filtered_df) > max_group_len:
            max_group_len = len(filtered_df)
        print(filtered_df)
        groups.append(filtered_df)

    # Create a 4x4 grid of subplots
    fig, axarr = plt.subplots(len(groups), max_group_len, figsize=(len(groups)*4, max_group_len*(4)))


    order = ["A_B_", "A_Bt_", "At_B_", "At_Bt_"]

    plotted = list()

    y_max = 0.0
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            row_data = groups[i].iloc[j].tolist()[1:]
            if max(row_data) > y_max:
                y_max = max(row_data)
    y_max += 0.3
    plt.rcParams["text.usetex"] = True

    for i in range(len(groups)):
        if i == 0:
            for j in range(max_group_len):
                x = order[j] 
                x = x.replace("_", " ")
                x = x.replace("t", "^T")
                s = r"$" + r"C \leftarrow \alpha \cdot " + x + r" + \beta \cdot C" + r"$"
                axarr[0, j].set_title(s)
        #plt.xticks(rotation=75)
        for orderr in order:
            order_found = False
            for j in range(len(groups[i])):
                #print(orderr, groups[i].iloc[j]["Identifier"], orderr in groups[i]["Identifier"])
                if orderr in groups[i].iloc[j]["Identifier"]:
                    order_found = True
                    plotted.append((i, j))
                    row_data = groups[i].iloc[j].tolist()[1:]
                    labels = groups[i].columns.tolist()[1:]
                    labels = [deepcopy(x).split(" Time")[0] for x in labels]
                    labels = ["Dense-Dense" if "DD" in x else x for x in labels]
                    labels = ["Dense-Sparse" if "DS" in x else x for x in labels]
                    print(labels)
                    print(row_data)
                    colors = [nvidia_green if "cuBlas" in x or "cuSparse" in x else sparse_rose if "Dense-Sparse" in x else dense_blue for x in labels]
                    #colors = [sparse_rose if "Dense-Sparse" in x else deepcopy(str(x)) for x in colors]
                    print(colors)
                    bars = axarr[i, j].bar(labels, row_data, color=colors)
                    axarr[i, j].tick_params(axis='x', rotation=20, labelsize=9, pad=-2)
                    #axarr[i, j].set_title(f'Plot {4*i + j + 1}')
                    axarr[i, j].set_ylim((0.0, y_max+0.3))
                    
                    if j == 0:
                        g = groups[i].iloc[j]["Identifier"].split(orderr)[1].split("_DenseXSparse")[0].capitalize()
                        if g == "Random":
                            g += "\n(" + str(round(sparsity, 2)) + ")"
                        axarr[i, j].annotate(g, xy=(0, 0.5), xytext=(-axarr[i, j].yaxis.labelpad - 10, 0),
                                xycoords=axarr[i, j].yaxis.label, textcoords='offset points',
                                size='large', ha='right', va='center')
                    axarr[i, j].set_ylim((0.0, y_max))
                    axarr[i, j].yaxis.set_major_locator(MaxNLocator(nbins=11))
                    for bar in bars:
                        height = bar.get_height()
                        axarr[i, j].axhline(y=height, color='gray', linestyle='--', alpha=0.6, zorder=-1)  # Add dashed line

                        # Annotate text
                        axarr[i, j].annotate(str(round(height, 2)), 
                                    xy=(bar.get_x() + bar.get_width()*0.80, height - 0.08),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')

    print(plotted)
    for i in range(len(groups)):
        for j in range(max_group_len):
            if not (i,j) in plotted:
                print(f"({i}, {j})")
                axarr[i, j].text(0.5, 0.5, 'Not Enough\nShared Memory', 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    transform=axarr[i, j].transAxes,
                    fontsize=14)  # Use the Axes coordinate system
                #row_data = groups[i].iloc[j].tolist()[1:]
                row_data = [0.0 for _ in labels]
                labels = groups[i].columns.tolist()[1:]
                labels = [deepcopy(x).split(" Time")[0] for x in labels]
                labels = ["Dense-Dense" if "DD" in x else x for x in labels]
                labels = ["Dense-Sparse" if "DS" in x else x for x in labels]
                print(labels)
                print(row_data)
                colors = [nvidia_green if "cuBlas" in x or "cuSparse" in x else my_blue for x in labels]
                bars = axarr[i, j].bar(labels, row_data, color=colors)
                axarr[i, j].tick_params(axis='x', rotation=20, labelsize=9, pad=-2)
                #axarr[i, j].set_title(f'Plot {4*i + j + 1}')
                axarr[i, j].set_ylim((0.0, y_max))
                axarr[i, j].yaxis.set_major_locator(MaxNLocator(nbins=11))

    #for ax in axarr[0]:  # Only adjust titles for the first row
    #    ax.title.set(y=1.05)  # Adjust as necessary
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(f"{data_dir}/plots/dense-sparse{addname}-grid-A{row_a}x{col_a}-B{row_b}x{col_b}-C{row_c}x{col_c}-alpha{Alpha}-beta{Beta}.pdf")
    plt.clf()
    #plt.show()

if save_plots:
    plot_in_a_grid(p_ds=p_ds, addname="")
    plot_in_a_grid(p_ds=p_ds_ctv, addname="-ctv")