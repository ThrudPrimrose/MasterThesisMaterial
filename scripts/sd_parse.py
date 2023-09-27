import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from aux import *
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

if not os.path.exists(f"{stdout_dir}/plots"):
    os.mkdir(f"{stdout_dir}/plots")


def get_load_store_size(a_el_count, ctv):
    load_store = 0

    # If adressing is none then the matrix is loaded only 1 time in the whole batch
    # Read A
    if adressingA != "none" and not ctv:
        load_store += a_el_count

    # Read B
    if adressingB != "none":
        load_store += row_b * col_b

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

    Flops = (col_b) * (2 * row_a * row_b)
    # Flops -= col_a * col_b # First row

    if Alpha != 1.0:
        Flops += row_c * col_c

    # Flops += row_a * col_a

    # Adding C to end result, row_c = row_a, col_c = col_b
    if Beta != 0.0:
        Flops += 2 * row_c * col_c

    load_store = get_load_store_size(row_a * col_a, ctv)

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

    # Load A, B, C, store C, collective load of B, each thread then neesd a row of A
    # and a row of C
    load_store =  get_load_store_size(row_b * col_b)
    #return (ops, load_store)
    return ops, load_store
"""


def calculate_ops_sparse_dense(typestr, ctv):
    if typestr == "band":
        block_count = int(row_a // col_a)
        elcount = 0
        for i in range(block_count):
            elcount += 2 * 2
            elcount += 3 * (col_a - 2)
        remainder = row_a - block_count*col_a
        if remainder > 2:
            elcount += 2 * 2
            elcount += 3 * (remainder - 2)
        else:
            elcount += 2 * remainder
        ops = col_b * elcount * 2
        if Alpha != 1.0:
            ops += row_c * col_c
        if Beta != 0.0:
            ops += 2 * row_c * col_c
        # ops += row_a * col_a
        # ops -= row_a * col_a
        load_store = get_load_store_size(elcount, ctv)
        return (ops, load_store)
    elif typestr == "full":
        return calculate_ops_dense(ctv)
    elif typestr == "random":
        el_count = int(sparsity * (row_a * col_a))
        ops = col_b * el_count * 2
        if Alpha != 1.0:
            ops += row_c * col_c
        if Beta != 0.0:
            ops += 2 * row_c * col_c
        # ops += row_a * col_a
        # ops -= row_a * col_a
        load_store = get_load_store_size(el_count, ctv)
        return (ops, load_store)
    elif typestr == "chequered":
        a = row_a // 2 + (row_a % 2)
        b = row_a // 2
        c = col_a // 2 + (col_a % 2)
        d = col_a // 2
        el_count = a * c + b * d
        ops = col_b * el_count * 2
        if Alpha != 1.0:
            ops += row_c * col_c
        if Beta != 0.0:
            ops += 2 * row_c * col_c
        # ops += row_a * col_a
        # ops -= row_a * col_a
        load_store = get_load_store_size(el_count, ctv)
        return (ops, load_store)
    else:
        raise Exception(typestr + " is undefined")


pd_sd_dataframes = list()
pd_sd_ctv_dataframes = list()

for r in range(runs):
    path = f"{stdout_dir}/run{r}.txt"
    report_sparse_dense = []
    with open(path, "r") as file:
        identifier = ""
        dense_time = 0.0
        sparse_time = 0.0
        speed_up = 0.0
        op_diff = 0.0
        load_diff = 0.0
        flops_per_byte_dd = 0.0
        flops_per_byte_sd = 0.0
        sparse_dense_type = ""
        sparse_dense_type = ""
        gemmforge_flops_per_el = 0.0
        my_flops_per_el = 0.0
        state = "initial"
        num_items = 0.0
        cublas_time = 0.0
        cusparse_time = 0.0

        i = 1
        for line in file:
            # print(i)
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
                # inside the identifier there has to be smth like: sparse_dense_At_mul_B_full_compiler_time_value
                # A{T if transA else NT}_{a_type}_B{T if transB else NT}_DenseXDense
                # or
                # A{T if transA else NT}_B{T if transB else NT}_{b_type}_DenseXDense
                tokens = identifier.split("_")
                assert (len(tokens) >= 4)
                if "SparseXDense" in identifier:
                    sparse_dense_type = tokens[1]
                    assert (sparse_dense_type != "")
                    # print(i, "SD: ", sparse_dense_type)
                    print(sparse_dense_type)
                    #raise Exception(sparse_dense_type)
                state = "kernel"
                # print(i, " initial -> kernel")
                i += 1
                continue
            elif state == "kernel" and "Dense x Dense kernel took" in line:
                duration = line.split("Dense x Dense kernel took ")[1][:-3]
                dense_time = float(duration)
                state = "kernel-2"
                # print(i, " kernel -> kernel-2")
                i += 1
                print([dense_time, 0.0, 0.0, 0.0])
                continue
            elif state == "kernel-2" and "Sparse x Dense kernel took" in line:
                duration = line.split("Sparse x Dense kernel took ")[1][:-3]
                sparse_time = float(duration)
                state = "kernel-3"
                # print(i, " kernel-2 -> kernel-3")
                print([dense_time, sparse_time, 0.0, 0.0])
                i += 1
                continue
            elif state == "kernel-3" and "cuBlas DxD kernel took" in line:
                duration = line.split("cuBlas DxD kernel took ")[1][:-3]
                cublas_time = float(duration)
                state = "kernel-4"
                # print(i, " kernel-3 -> kernel-4")
                i += 1
                print([dense_time, sparse_time, cublas_time, 0.0])
                continue
            elif state == "kernel-4" and "cuSparse SxD kernel" in line:
                duration = line.split("cuSparse SxD kernel took ")[1][:-3]
                cusparse_time = float(duration)
                state = "write-sd"
                i += 1
                print([dense_time, sparse_time, cublas_time, cusparse_time])
                continue
            elif state != "initial" and "Freeing device memory" in line:
                speed_up = 0.0 if sparse_time == 0.0 else dense_time / sparse_time
                dd_ops, dd_load_store = calculate_ops_dense(False)
                # print(sparse_dense_type)
                if "_ctv" in identifier:
                    sd_ops, sd_load_store = calculate_ops_sparse_dense(
                        sparse_dense_type, True)
                else:
                    sd_ops, sd_load_store = calculate_ops_sparse_dense(
                        sparse_dense_type, False)
                op_diff = float(dd_ops) / float(sd_ops)
                load_store_diff = float(dd_load_store) / float(sd_load_store)
                flops_per_byte_dd = float(dd_ops) / float(dd_load_store)
                flops_per_byte_sd = float(sd_ops) / float(sd_load_store)
                speed_up_per = 100 * (dense_time - sparse_time) / dense_time
                dd_flops = (float(num_items) * float(dd_ops)) / (float(dense_time) * 1e6)
                if num_items == 0.0:
                    raise Exception("Number items should have been found")
                sd_flops = (float(num_items) * float(sd_ops)) / (float(sparse_time) * 1e6)
                report_sparse_dense.append([identifier,
                                            dense_time,
                                            sparse_time,
                                            cublas_time,
                                            cusparse_time,
                                            speed_up,
                                            # round(speed_up_per, 2),
                                            op_diff,
                                            load_store_diff,
                                            flops_per_byte_dd,
                                            flops_per_byte_sd,
                                            dd_flops,
                                            sd_flops,
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
                flops_per_byte_sd = 0.0
                sparse_dense_type = ""
                sparse_dense_type = ""
                num_items = 0.0
                cublas_time = 0.0
                cusparse_time = 0.0
                continue
            else:
                i += 1

    """
    print(
        tabulate(report_sparse_dense,
                headers=["Identifier",
                        "DD Time",
                        "SD Time",
                        "cuBlas Time",
                        "cuSparse Time",
                        "Speed-up",
                        "Flop. Ceil",
                        "LS Ceil",
                        "DD Flop/b",
                        "SD Flop/b",
                        "DD GFlop/s",
                        "SD GFlop/s"],
                tablefmt="github"))
    """

    report_sparse_dense_both = list(sorted(report_sparse_dense, key=lambda x: x[0]))
    # report_sparse_dense = list(sorted(report_sparse_dense, key=lambda x: x[0]))
    report_sparse_dense = list(
        filter(lambda x: not "_ctv" in x[0], report_sparse_dense_both))
    # report_sparse_dense = list(
    #    filter(lambda x: "_ctv" not in x[0], report_sparse_dense))
    report_sparse_dense_ctv = list(
        filter(lambda x: "_ctv" in x[0], report_sparse_dense_both))

    """
    print(
        tabulate(report_sparse_dense,
                headers=["Identifier",
                        "DD Time",
                        "SD Time",
                        "cuBlas Time",
                        "cuSparse Time",
                        "Speed-up",
                        "Flop. Ceil",
                        "LS Ceil",
                        "DD Flop/b",
                        "SD Flop/b",
                        "DD GFlop/s",
                        "SD GFlop/s"],
                tablefmt="github"))
    print(
        tabulate(report_sparse_dense_ctv,
                headers=["Identifier",
                        "DD Time",
                        "SD Time",
                        "cuBlas Time",
                        "cuSparse Time",
                        "Speed-up",
                        # "%",
                        "Flop. Ceil",
                        "LS Ceil",
                        "DD Flop/b",
                        "SD Flop/b",
                        "DD GFlop/s",
                        "Sd GFlop/s"],
                tablefmt="github"))
    """

    tmp1 = pd.DataFrame(data=deepcopy(report_sparse_dense), columns=[
        "Identifier",
        "DD Time",
        "SD Time",
        "cuBlas Time",
        "cuSparse Time",
        "Speed-up",
        # "%",
        "Flop. Ceil",
        "LS Ceil",
        "DD Flop/b",
        "SD Flop/b",
        "DD GFlop/s",
        "SD GFlop/s"
    ])
    tmp1 = tmp1.sort_values(by="Identifier").copy()
    print(f"SD DATAFRAME {r}:")
    print(tmp1)
    pd_sd_dataframes.append(tmp1.copy())
    tmp2 = pd.DataFrame(data=deepcopy(report_sparse_dense_ctv), columns=[
        "Identifier",
        "DD Time",
        "SD Time",
        "cuBlas Time",
        "cuSparse Time",
        "Speed-up",
        # "%",
        "Flop. Ceil",
        "LS Ceil",
        "DD Flop/b",
        "SD Flop/b",
        "DD GFlop/s",
        "SD GFlop/s"
    ])
    tmp2 = tmp2.sort_values(by="Identifier").copy()
    print(f"SD CTV DATAFRAME {r}:")
    print(tmp2)
    pd_sd_ctv_dataframes.append(tmp2.copy())

# This takes the average and covariance of 2 pandas data frames,
# Runtime for denses parse dense sparse with ctv 

p_sd_sum = pd_sd_dataframes[0].copy()

# Take average except the str row (ops on str not supported sadly)
for df in pd_sd_dataframes[1:]:
    p_sd_sum.iloc[:, 1:] += df.iloc[:, 1:].copy()

p_sd_sum.iloc[:, 1:] = p_sd_sum.iloc[:, 1:].copy() / float(runs)
p_sd = p_sd_sum.copy()
print("AVG SD DATAFRAME:")
print(p_sd)
# print(p_sd)
# print("---------------------")
p_sd_ctv_sum = pd_sd_ctv_dataframes[0].copy()

# Take average except the str row (ops on str not supported sadly)
for df in pd_sd_ctv_dataframes[1:]:
    p_sd_ctv_sum.iloc[:, 1:] += df.iloc[:, 1:].copy()

p_sd_ctv_sum.iloc[:, 1:] = p_sd_ctv_sum.iloc[:, 1:].copy() / float(runs)
p_sd_ctv = p_sd_ctv_sum.copy()
print("AVG SD CTV DATAFRAME:")
print(p_sd_ctv)
# print(p_sd_ctv)
# print("======================")
p_sd_var_dist = pd_sd_dataframes[0].copy()
p_sd_var_dist.iloc[:, 1:] -= p_sd_sum.iloc[:, 1:].copy()
p_sd_var_dist.iloc[:, 1:] = p_sd_var_dist.iloc[:, 1:].copy().pow(2)
for df in pd_sd_dataframes[1:]:
    tmp = df.iloc[:, 1:].copy() - p_sd.iloc[:, 1:].copy()
    p_sd_var_dist += tmp.pow(2).copy()
p_sd_var = p_sd_var_dist.copy()
p_sd_var.iloc[:, 1:] = p_sd_var_dist.iloc[:, 1:].copy() / float(runs)
p_sd_var["Identifier"] = p_sd["Identifier"]

p_sd_ctv_var_dist = pd_sd_ctv_dataframes[0].copy()
p_sd_ctv_var_dist.iloc[:, 1:] -= p_sd_ctv.iloc[:, 1:].copy()
p_sd_ctv_var_dist.iloc[:, 1:] = p_sd_ctv_var_dist.iloc[:, 1:].copy().pow(2)
for df in pd_sd_ctv_dataframes[1:]:
    tmp = df.iloc[:, 1:].copy() - p_sd_ctv.iloc[:, 1:].copy()
    p_sd_ctv_var_dist += tmp.pow(2).copy()
p_sd_ctv_var = p_sd_ctv_var_dist.copy()
p_sd_ctv_var["Identifier"] = p_sd_ctv["Identifier"]

# p_sd_ctv_var = sum([(x - p_sd_ctv).pow(2) for x in pd_sd_ctv_dataframes]) / runs

# print(p_sd)
# print("~~~~~~~~~~~~~~~~~~~")
# print(p_sd_ctv)
print("VAR SD DATAFRAME:")
print(p_sd_var)
print("VAR SD CTV DATAFRAME:")
print(p_sd_ctv_var)

# fig.clear()
dd_points = p_sd[["DD Flop/b", "DD GFlop/s"]]
sd_points = p_sd[["SD Flop/b", "SD GFlop/s"]]
sd_points_ctv = p_sd_ctv[["SD Flop/b", "SD GFlop/s"]]
dd_points_ctv = p_sd_ctv[["DD Flop/b", "DD GFlop/s"]]


def round_up_to_power_of_ten(n):
    if n == 0:
        return 1
    elif n < 0:
        return -round_up_to_power_of_ten(-n)
    else:
        power = math.ceil(math.log10(n))
        return 10 ** power


def plot_roofline(peak_memory_bandwidth, peak_floating_point_perf, title, sd_points, dd_points, p_sd, addname,
                  second_values=False, sd_points_2=None, dd_points_2=None, p_sd_2=None):
    plt.clf()
    done = [False for _ in a_matrix_types]
    txt = ["" for _ in p_sd["Identifier"]]

    def roof(val):
        return min(peak_floating_point_perf,
                   (peak_memory_bandwidth * val))

    xpts = np.linspace(0, 40, 250)
    plt.plot(xpts, [roof(x) for x in xpts], label="Roofline")
    xit = 0
    for x in p_sd["Identifier"]:
        for i in range(len(a_matrix_types)):
            if done[i] == False:
                if a_matrix_types[i] in x:
                    done[i] = True
                    txt[xit] = a_matrix_types[i].capitalize()
        xit += 1
    txt2 = deepcopy(txt)

    if not second_values:
        l = [(sd_points, dd_points, p_sd)]
    else:
        l = [(sd_points, dd_points, p_sd), (sd_points_2, dd_points_2, p_sd_2)]

    for m, (sd_points, dd_points, p_sd) in enumerate(l):
        plt.scatter(x=sd_points["SD Flop/b"], y=sd_points["SD GFlop/s"], c=sparse_rose, s=10, label="Dense-Sparse")
        for name in a_matrix_types:
            avg = 0.0
            n = 0
            for i, (xi, yi) in enumerate(zip(sd_points["SD Flop/b"], sd_points["SD GFlop/s"])):
                if name in p_sd["Identifier"].iloc[i]:
                    roofline = roof(xi)
                    implementation = yi
                    percentage = implementation * 100.0 / roofline
                    avg += percentage
                    n += 1
            for i, (xi, yi) in enumerate(zip(sd_points["SD Flop/b"], sd_points["SD GFlop/s"])):
                if name in p_sd["Identifier"].iloc[i]:
                    if m == 0:
                        if txt[i] != "":
                            ctv = ""
                            if addname == "-ctv":
                                ctv = "(ctv)"
                            txt[i] = txt[i] + ctv + " " + str(round(avg / n, 1)) + "%"
                    else:
                        if txt2[i] != "":
                            ctv = "(ctv)"
                            txt2[i] = txt2[i] + ctv + " " + str(round(avg / n, 1)) + "%"
                    if m == 0:
                        o = 6
                        if "(ctv)" in txt[i]:
                            if "Full" in txt[i]:
                                o = -86
                            if "Chequered" in txt[i]:
                                o = -126
                        if row_a == 56 and not "(ctv)" in txt[i]:
                            if "Full" in txt[i]:
                                if (len(l) == 1):
                                    o = 56
                                else:
                                    o = 36
                        yadd = 0
                        if not "(ctv)" in txt[i]:
                            if "Random" in txt[i]:
                                yadd = 7
                            if "Band" in txt[i]:
                                yadd = -7
                        plt.annotate(txt[i], (xi, yi), textcoords="offset points", xytext=(o, yadd -2), ha='left',
                                     size=9 if len(l) == 1 else 7)
                    else:
                        o = 6
                        if "(ctv)" in txt2[i]:
                            if "Full" in txt2[i]:
                                o = -56
                            if "Chequered" in txt2[i]:
                                o = -86
                        if not "(ctv)" in txt2[i]:
                            if "Full" in txt2[i]:
                                o = +56
                        if "(ctv)" in txt2[i] and "Band" in txt2[i]:
                            o = -56
                        plt.annotate(txt2[i], (xi, yi), textcoords="offset points", xytext=(o, 7), ha='left',
                                     size=9 if len(l) == 1 else 7)
        avg = 0.0
        n = 0
        for i, (xi, yi) in enumerate(zip(dd_points["DD Flop/b"], dd_points["DD GFlop/s"])):
            roofline = roof(xi)
            implementation = yi
            percentage = implementation * 100.0 / roofline
            avg += percentage
            n += 1
        xi = dd_points["DD Flop/b"][0]
        yi = dd_points["DD GFlop/s"][0]
        t = "Dense" + " " + str(round(avg / n, 1)) + "%"
        if m == 0:
            if len(l) == 1:
                plt.annotate(t, (xi, yi), textcoords="offset points", xytext=(-86, -2), ha='left', size=9)
            else:
                plt.annotate(t, (xi, yi), textcoords="offset points", xytext=(6, -9), ha='left', size=7)

        plt.scatter(x=dd_points["DD Flop/b"],
                    y=dd_points["DD GFlop/s"], c=dense_blue, s=10, label="Dense-Dense")
        ymax = max(max(dd_points["DD GFlop/s"]), max(sd_points["SD GFlop/s"]))
        xmax = max(max(dd_points["DD Flop/b"]), max(sd_points["SD Flop/b"]))
        if m == 0:
            plt.legend(loc="lower right")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim((0, round_up_to_power_of_ten(ymax)))
    plt.xlim((0, round_up_to_power_of_ten(xmax)))
    plt.title(title)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('Operational Intensity (Flop/byte)')
    plt.ylabel('Performance (GFlops/second)')
    # Show the plot
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/sparse-dense{addname}-roofline-A{row_a}x{col_a}-B{row_b}x{col_b}-C{row_c}x{col_c}-alpha{Alpha}-beta{Beta}.pdf.pdf")
    plt.clf()


# plot_roofline(peakMemoryBandwidth, peakFLOP,
#              "Experimental from Nsight Compute")
if save_plots:
    plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "Roofline Model for Dense-Dense and Sparse-Dense Kernels", sd_points=sd_points, dd_points=dd_points,
                  p_sd=p_sd, addname="")
    plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "Roofline Model for Dense-Dense and Sparse-Dense Kernels\nWith Compile Time Matrix Values",
                  sd_points=sd_points_ctv,
                  dd_points=dd_points_ctv, p_sd=p_sd_ctv, addname="-ctv")
    plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "Roofline Model for Dense-Dense and Sparse-Dense Kernels\nWith and Without Compile Time Values",
                  sd_points=sd_points,
                  dd_points=dd_points, p_sd=p_sd, addname="-both", second_values=True, sd_points_2=sd_points_ctv,
                  dd_points_2=dd_points_ctv, p_sd_2=p_sd_ctv)


# plot_roofline(lookupPeakMemoryBandwidth,
#              lookupPeakFLOP, "Theoretical from Web")


def plot_in_a_grid(p_sd, p_sd_var, addname, plot_relative_speed_up=True):
    times_only = p_sd[[
        "Identifier",
        "DD Time",
        "SD Time",
        "cuBlas Time",
        "cuSparse Time"
    ]]
    print(p_sd_var)
    times_only_var = p_sd_var[[
        "Identifier",
        "DD Time",
        "SD Time",
        "cuBlas Time",
        "cuSparse Time"
    ]]

    def modify_row(row):
        row["Identifier"] = row["Identifier"][:-1]
        row['DD Time'] = row['cuBlas Time'] / row['DD Time']
        row['SD Time'] = row['cuBlas Time'] / row['SD Time']
        row['cuSparse Time'] = row['cuBlas Time'] / row['cuSparse Time']
        row['cuBlas Time'] = row['cuBlas Time'] / row['cuBlas Time']
        return row

    if plot_relative_speed_up:
        values = times_only.apply(modify_row, axis=1)
        print(values)
    else:
        values = times_only
        variances = times_only_var

    groups = list()
    var_groups = list()
    max_group_len = 0
    for a_type in a_matrix_types:
        mask = [a_type in x for x in values['Identifier']]
        filtered_df = values[[a_type in x for x in values['Identifier']]]
        if len(filtered_df) > max_group_len:
            max_group_len = len(filtered_df)
        print(filtered_df)
        groups.append(filtered_df)
        if not plot_relative_speed_up:
            filtered_df_var = variances[[a_type in x for x in values['Identifier']]]
            var_groups.append(filtered_df_var)
            print(filtered_df_var)

    # Create a 4x4 grid of subplots
    fig, axarr = plt.subplots(len(groups), max_group_len, figsize=(len(groups) * 4, max_group_len * (4)))

    order = [("A_","B_"), ("A_","Bt_"), ("At_","B_"), ("At_","Bt_")]

    plotted = list()

    y_max = 0.0
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            row_data = groups[i].iloc[j].tolist()[1:]
            print(groups[i].iloc[j].tolist()[0])
            if "At_" in groups[i].iloc[j].tolist()[0]:
                if max(row_data[:-1]) > y_max:
                    y_max = max(row_data[:-1])
            else:
                if max(row_data) > y_max:
                    y_max = max(row_data)
    y_max += 0.3
    plt.rcParams["text.usetex"] = True

    for i in range(len(groups)):
        if i == 0:
            for j in range(max_group_len):
                x = order[j][0] + order[j][1]
                x = x.replace("_", " ")
                x = x.replace("t", "^T")
                xs = x.split()
                s = r"$" + r"C \leftarrow \alpha \cdot " + xs[0] + r" \times " + xs[1] + r" + \beta \cdot C" + r"$"
                axarr[0, j].set_title(s)
        # plt.xticks(rotation=75)
        for orderr in order:
            order_found = False
            for j in range(len(groups[i])):
                # print(orderr, groups[i].iloc[j]["Identifier"], orderr in groups[i]["Identifier"])
                if orderr[0] in groups[i].iloc[j]["Identifier"] and \
                    orderr[1] in groups[i].iloc[j]["Identifier"]:
                    order_found = True
                    plotted.append((i, j))
                    row_data = groups[i].iloc[j].tolist()[1:]
                    if not plot_relative_speed_up:
                        var_data = var_groups[i].iloc[j].tolist()[1:]
                        std_dev_data = np.sqrt(var_data)
                        yerr = 1.96 * (std_dev_data / np.sqrt(runs))
                    labels = groups[i].columns.tolist()[1:]
                    labels = [deepcopy(x).split(" Time")[0] for x in labels]
                    labels = ["Dense-Dense" if "DD" in x else x for x in labels]
                    labels = ["Sparse-Dense" if "SD" in x else x for x in labels]
                    print("Row-Labels:", labels)
                    print("Row-Data:", row_data)
                    colors = [
                        nvidia_green if "cuBlas" in x or "cuSparse" in x else sparse_rose if "Sparse-Dense" in x else dense_blue
                        for x in labels]
                    # colors = [sparse_rose if "Dense-Sparse" in x else deepcopy(str(x)) for x in colors]
                    print(colors)
                    if orderr[0] == "At_":
                        for m, label in enumerate(labels):
                            if "cuSparse" in label:
                                row_data[m] = 0.0
                                if not plot_relative_speed_up:
                                    yerr[m] = 0.0
                    if not plot_relative_speed_up:
                        bars = axarr[i, j].bar(labels, row_data, color=colors)
                        # axarr[i, j].errorbar(labels, row_data, yerr=yerr, fmt='-o')
                        shade_darker = [darken_hex_color(color, 0.60) for color in colors]
                        for pos, y, err, color in zip(labels, row_data, yerr, shade_darker):
                            axarr[i, j].errorbar(pos, y, err, lw=1, capsize=5, capthick=1, color=color)
                    else:
                        bars = axarr[i, j].bar(labels, row_data, color=colors)
                    axarr[i, j].tick_params(axis='x', rotation=20, labelsize=9, pad=-2)
                    # axarr[i, j].set_title(f'Plot {4*i + j + 1}')
                    axarr[i, j].set_ylim((0.0, y_max + 0.05 * y_max))

                    if j == 0:
                        g = groups[i].iloc[j]["Identifier"].split(orderr[0])[1].split(orderr[1])[0][:-1].capitalize()
                        if g == "Random":
                            g += "\n(" + str(round(sparsity, 2)) + ")"
                        axarr[i, j].annotate(g, xy=(0, 0.5), xytext=(-axarr[i, j].yaxis.labelpad - 10, 0),
                                             xycoords=axarr[i, j].yaxis.label, textcoords='offset points',
                                             size='large', ha='right', va='center')
                    axarr[i, j].set_ylim((0.0, y_max + 0.05 * y_max))
                    axarr[i, j].yaxis.set_major_locator(MaxNLocator(nbins=11))
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0.0:
                            axarr[i, j].axhline(y=height, color='gray', linestyle='--', alpha=0.6,
                                            zorder=-1)  # Add dashed line

                            # Annotate text
                            axarr[i, j].annotate(str(round(height, 2)),
                                             xy=(bar.get_x() + bar.get_width() * 0.80, height - 0.08),
                                             xytext=(0, 3),  # 3 points vertical offset
                                             textcoords="offset points",
                                             ha='center', va='bottom')

    """
    print(plotted)
    for i in range(len(groups)):
        for j in range(max_group_len):
            if not (i, j) in plotted:
                print(f"({i}, {j})")
                axarr[i, j].text(0.5, 0.5, 'Not Enough\nShared Memory',
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=axarr[i, j].transAxes,
                                 fontsize=14)  # Use the Axes coordinate system
                # row_data = groups[i].iloc[j].tolist()[1:]
                row_data = [0.0 for _ in labels]
                labels = groups[i].columns.tolist()[1:]
                labels = [deepcopy(x).split(" Time")[0] for x in labels]
                labels = ["Dense-Dense" if "DD" in x else x for x in labels]
                labels = ["Dense-Sparse" if "SD" in x else x for x in labels]
                print(labels)
                print(row_data)
                colors = [nvidia_green if "cuBlas" in x or "cuSparse" in x else my_blue for x in labels]
                bars = axarr[i, j].bar(labels, row_data, color=colors)
                axarr[i, j].tick_params(axis='x', rotation=20, labelsize=9, pad=-2)
                # axarr[i, j].set_title(f'Plot {4*i + j + 1}')
                axarr[i, j].set_ylim((0.0, y_max))
                axarr[i, j].yaxis.set_major_locator(MaxNLocator(nbins=11))
    """

    # for ax in axarr[0]:  # Only adjust titles for the first row
    #    ax.title.set(y=1.05)  # Adjust as necessary
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/sparse-dense-{addname}-grid-{'relSpeedup' if plot_relative_speed_up else 'abs_time'}-A{row_a}x{col_a}-B{row_b}x{col_b}-C{row_c}x{col_c}-alpha{Alpha}-beta{Beta}.pdf")
    plt.clf()
    # plt.show()


if save_plots:
    plot_in_a_grid(p_sd=p_sd, p_sd_var=p_sd_var, addname="", plot_relative_speed_up=False)
    plot_in_a_grid(p_sd=p_sd_ctv, p_sd_var=p_sd_ctv_var, addname="-ctv", plot_relative_speed_up=False)
    plot_in_a_grid(p_sd=p_sd, p_sd_var=p_sd_var, addname="", plot_relative_speed_up=True)
    plot_in_a_grid(p_sd=p_sd_ctv, p_sd_var=p_sd_ctv_var, addname="-ctv", plot_relative_speed_up=True)

l1 = list()
l2 = list()
for index, row in p_sd.iterrows():
    bytefactor = row["DD Flop/b"] / row["SD Flop/b"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficeint for the case without CTV:", correlation_coefficient)

l1.clear()
l2.clear()
for index, row in p_sd_ctv.iterrows():
    bytefactor = row["DD Flop/b"] / row["SD Flop/b"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For CTV: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficient for the case with CTV:", correlation_coefficient)

l1 = list()
l2 = list()
for index, row in p_sd.iterrows():
    bytefactor = row["DD GFlop/s"] / row["SD GFlop/s"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficeint for the case without CTV GFLOP/s:", correlation_coefficient)

l1.clear()
l2.clear()
for index, row in p_sd_ctv.iterrows():
    bytefactor = row["DD GFlop/s"] / row["SD GFlop/s"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For CTV: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficient for the case with CTV GFLOP/s:", correlation_coefficient)

l1 = list()
l2 = list()
for index, row in p_sd.iterrows():
    bytefactor = row["DD GFlop/s"] / row["SD GFlop/s"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficeint for the case without CTV GFLOP/s:", correlation_coefficient)

l1.clear()
l2.clear()
for index, row in p_sd_ctv.iterrows():
    bytefactor = row["DD Flop/b"] / row["SD Flop/b"]
    speed_up = row["DD GFlop/s"] / row["SD GFlop/s"]
    # print(f"For CTV: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficient for the case with CTV GFLOP/s from FLOP/b:", correlation_coefficient)

l1.clear()
l2.clear()
for index, row in p_sd.iterrows():
    bytefactor = row["DD Flop/b"] / row["SD Flop/b"]
    speed_up = row["DD GFlop/s"] / row["SD GFlop/s"]
    # print(f"For CTV: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficient for the case without CTV GFLOP/s from FLOP/b:", correlation_coefficient)
