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

pd_dataframes = list()

for r in range(runs):
    path = f"{stdout_dir}/run{r}.txt"
    report = []
    with open(path, "r") as file:
        state = "initial"
        kernel = ""
        cutensor = 0.0
        gemmforge = 0.0
        cutensor_efficiency = 0.0
        gemmforge_efficiency = 0.0
        gflops = 0.0
        operational_intensity = 0.0
        
        
        for i, line in enumerate(file):
          if state == "initial" and "compute the kernel" in line and "with Gemmforge" in line:
            state = "check-gemmforge"
            tokens = line.split(", with")
            kernel = tokens[0].split("kernel: ")[-1]
            print(kernel)
          if state == "check-gemmforge" and  "Gemmforge Tensor Contraction took" in line:
            state = "check-cutensor"
            runtime = line.split("ms")[0].split("took: ")[-1]
            gemmforge = float(runtime)
            print(gemmforge)
          if state == "check-cutensor" and  "cuTensor Tensor Contraction took" in line:
            state = "check-gflops"
            runtime = line.split("ms")[0].split("took: ")[-1]
            cutensor = float(runtime)
            print(cutensor)
          if state == "check-gflops" and "GFLOPs/s" in line:
            state = "check-operational-intensity"
            flops = line.split()[-1]
            gflops = float(flops)/2.0
            print(gflops)
          if state == "check-operational-intensity" and "intensity:" in line:
            state = "check-gemmforge-eff"
            intensity = line.split("intensity:")[-1]
            operational_intensity = float(intensity)/2.0
            print(operational_intensity)
          if state == "check-gemmforge-eff" and "roof w. respect to operational intensity achieved with Gemmforge" in line:
            state = "check-cutensor-eff"
            eff = line.split()[0]
            gemmforge_efficiency = float(eff)
            print(gemmforge_efficiency)
          if state == "check-cutensor-eff" and "roof w. respect to operational intensity achieved with cuTensor" in line:
            state = "final"
            eff = line.split()[0]
            eff = float(eff)
            cutensor_efficiency = float(eff)
            print(cutensor_efficiency)
          if state == "final":
            report.append([kernel, gemmforge, 
                           cutensor, gemmforge_efficiency, 
                           cutensor_efficiency, gflops, gflops*gemmforge/cutensor,
                           operational_intensity])
            state = "initial"
            kernel = ""
            cutensor = 0.0
            gemmforge = 0.0
            cutensor_efficiency = 0.0
            gemmforge_efficiency = 0.0
            gflops = 0.0
            operational_intensity = 0.0
            print(report)


    tmp1 = pd.DataFrame(data=deepcopy(report), columns=[
        "Kernel",
        "Gemmforge Time",
        "cuTensor Timer",
        "Gemmforge Efficiency",
        "cuTensor efficiency",
        "Gemmforge GFLOP/s",
        "cuTensor GFLOP/s",
        "Operational Intensity"
    ])
    tmp1 = tmp1.sort_values(by="Kernel").copy()
    print(f"DATAFRAME {r}:")
    print(tmp1)

    pd_dataframes.append(tmp1.copy())

# This takes the average and covariance of 2 pandas data frames,
# Runtime for denses parse dense sparse with ctv 

pd_sum = pd_dataframes[0].copy()

# Take average except the str row (ops on str not supported sadly)
for df in pd_dataframes[1:]:
    pd_sum.iloc[:, 1:] += df.iloc[:, 1:].copy()

pd_sum.iloc[:, 1:] = pd_sum.iloc[:, 1:].copy() / float(runs)
pd_avg = pd_sum.copy()
print("AVG DATAFRAME:")
print(pd_avg)

pd_var_dist = pd_dataframes[0].copy()
pd_var_dist.iloc[:, 1:] -= pd_sum.iloc[:, 1:].copy()
pd_var_dist.iloc[:, 1:] = pd_var_dist.iloc[:, 1:].copy().pow(2)
for df in pd_dataframes[1:]:
    tmp = df.iloc[:, 1:].copy() - pd_avg.iloc[:, 1:].copy()
    pd_var_dist += tmp.pow(2).copy()
pd_var = pd_var_dist.copy()
pd_var.iloc[:, 1:] = pd_var_dist.iloc[:, 1:].copy() / float(runs)
pd_var["Kernel"] = pd_avg["Kernel"]


gemmforge_points = pd_avg[["Kernel", "Operational Intensity", "Gemmforge GFLOP/s"]]
cutensor_points = pd_avg[["Kernel", "Operational Intensity", "cuTensor GFLOP/s"]]

gemmforge_var = pd_var[["Kernel", "Operational Intensity", "Gemmforge GFLOP/s"]]
cutensor_var = pd_var[["Kernel", "Operational Intensity", "cuTensor GFLOP/s"]]

def round_up_to_power_of_ten(n):
    if n == 0:
        return 1
    elif n < 0:
        return -round_up_to_power_of_ten(-n)
    else:
        power = math.ceil(math.log10(n))
        return 10 ** power


def plot_roofline(peak_memory_bandwidth, peak_floating_point_perf, 
                  title, gemmforge_points, cutensor_points, pd_avg):
    plt.clf()
    fig, ax = plt.subplots(figsize=(9, 6))  # Adjust the width and height as needed

    def roof(val):
        return min(peak_floating_point_perf,
                   (peak_memory_bandwidth * val))

    # Set the width of each bar
    bar_width = 0.4
    bar_separation = 0.2
    # Calculate the positions for the bars
    
    x_positions1 = np.arange(len(gemmforge_points["Kernel"]), dtype=float)
    x_positions2 = x_positions1 + bar_width
    x_positions = x_positions1 + bar_width 
    x_positions_f = np.arange(len(gemmforge_points["Kernel"]) + 1, dtype=float) - 0.5

    xpts = np.linspace(0, 40, 250)
    #plt.plot(xpts, [roof(x) for x in xpts], label="Roofline")
    plt.bar(x_positions1,
            gemmforge_points["Gemmforge GFLOP/s"], color=dense_blue, width = bar_width,
            label="Gemmfogre")
    plt.bar(x_positions2,
            cutensor_points["cuTensor GFLOP/s"], color=nvidia_green, width = bar_width,
            label = "cuTensor")
    ymax = max(max(gemmforge_points["Gemmforge GFLOP/s"]), max(cutensor_points["cuTensor GFLOP/s"]))
    xmax = max(max(gemmforge_points["Operational Intensity"]), max(cutensor_points["Operational Intensity"]))
    theo_roof_for_intensity = roof(max(gemmforge_points["Operational Intensity"]))
    line_values = np.full(len(gemmforge_points["Kernel"]) + 1, theo_roof_for_intensity)
    plt.plot(x_positions_f, line_values, marker='', color='gray', linestyle='--', label='Roofline')

    kernel_strs = list()
    for kernel_str in gemmforge_points["Kernel"]:
      a_c_perm = kernel_str[3:6]
      b_perm = kernel_str[-5:-2]
      kernel_strs.append(a_c_perm + "-" + b_perm)

    std_dev_data = np.sqrt(gemmforge_var["Gemmforge GFLOP/s"])
    yerr = 1.96 * (std_dev_data / np.sqrt(runs))
    plt.errorbar(x_positions1, gemmforge_points["Gemmforge GFLOP/s"], 
                 yerr=yerr, fmt='none', ecolor='black', capsize=5, 
                 capthick=2, label='_nolegend_')
    std_dev_data = np.sqrt(cutensor_var["cuTensor GFLOP/s"])
    yerr = 1.96 * (std_dev_data / np.sqrt(runs))
    plt.errorbar(x_positions2, cutensor_points["cuTensor GFLOP/s"], 
                 yerr=yerr, fmt='none', ecolor='black', 
                 capsize=5, capthick=2,
                 label='_nolegend_')

    for i, value in enumerate(gemmforge_points["Gemmforge GFLOP/s"]):
        float_number = 100*gemmforge_points["Gemmforge GFLOP/s"].iloc[i] / theo_roof_for_intensity
        formatted_number = f"{float_number:.1f}%"
        plt.text(x_positions1[i] + bar_width/2, value + 1, str(formatted_number), ha='center', va='bottom', fontsize=8, c="gray")

    for i, value in enumerate(cutensor_points["cuTensor GFLOP/s"]):
        float_number = 100*cutensor_points["cuTensor GFLOP/s"].iloc[i] / theo_roof_for_intensity
        formatted_number = f"{float_number:.1f}%"
        plt.text(x_positions2[i] + bar_width/2, value + 1, str(formatted_number), ha='center', va='bottom', fontsize=8, c="gray")


    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=12)
    #plt.yscale("log")
    #plt.xscale("log")
    #plt.ylim((0, round_up_to_power_of_ten(ymax)))
    #plt.xlim((0, round_up_to_power_of_ten(xmax)))
    plt.title(title, fontsize=14)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('Index Permutations', fontsize=12)
    plt.ylabel('Performance (GFLOPs/s)', fontsize=12)
    plt.xticks(x_positions1 + (bar_width) / 2, kernel_strs, rotation=70, ha='right', fontsize=14)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/component-wise-product-roofline.pdf")
    plt.clf()


if save_plots:
    plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "", 
                  gemmforge_points=gemmforge_points,
                  cutensor_points=cutensor_points,
                  pd_avg=pd_avg)

"""
def plot_in_a_grid(pd_avg, pd_var, addname, plot_relative_speed_up=True):
    times_only = pd_avg[[
        "Identifier",
        "DD Time",
        "SD Time",
        "cuBlas Time",
        "cuSparse Time"
    ]]
    print(pd_var)
    times_only_var = pd_var[[
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

    # for ax in axarr[0]:  # Only adjust titles for the first row
    #    ax.title.set(y=1.05)  # Adjust as necessary
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/ComponentWiseProduct-{'relSpeedup' if plot_relative_speed_up else 'abs_time'}-A{row_a}x{col_a}-B{row_b}x{col_b}-C{row_c}x{col_c}-alpha{Alpha}-beta{Beta}.pdf")
    plt.clf()
    # plt.show()


if save_plots:
    plot_in_a_grid(pd_avg=pd_avg, pd_var=pd_var, addname="", plot_relative_speed_up=False)
    plot_in_a_grid(pd_avg=pd_ctv, pd_var=pd_ctv_var, addname="-ctv", plot_relative_speed_up=False)
    plot_in_a_grid(pd_avg=pd_avg, pd_var=pd_var, addname="", plot_relative_speed_up=True)
    plot_in_a_grid(pd_avg=pd_ctv, pd_var=pd_ctv_var, addname="-ctv", plot_relative_speed_up=True)

l1 = list()
l2 = list()
for index, row in pd_avg.iterrows():
    bytefactor = row["Operational Intensity"] / row["Operational Intensity"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficeint for the case without CTV:", correlation_coefficient)

l1.clear()
l2.clear()
for index, row in pd_ctv.iterrows():
    bytefactor = row["Operational Intensity"] / row["Operational Intensity"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For CTV: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficient for the case with CTV:", correlation_coefficient)

l1 = list()
l2 = list()
for index, row in pd_avg.iterrows():
    bytefactor = row["DD GFLOP/s"] / row["SD GFLOP/s"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficeint for the case without CTV GFLOP/s:", correlation_coefficient)

l1.clear()
l2.clear()
for index, row in pd_ctv.iterrows():
    bytefactor = row["DD GFLOP/s"] / row["SD GFLOP/s"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For CTV: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficient for the case with CTV GFLOP/s:", correlation_coefficient)

l1 = list()
l2 = list()
for index, row in pd_avg.iterrows():
    bytefactor = row["DD GFLOP/s"] / row["SD GFLOP/s"]
    speed_up = row["DD Time"] / row["SD Time"]
    # print(f"For: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficeint for the case without CTV GFLOP/s:", correlation_coefficient)

l1.clear()
l2.clear()
for index, row in pd_ctv.iterrows():
    bytefactor = row["Operational Intensity"] / row["Operational Intensity"]
    speed_up = row["DD GFLOP/s"] / row["SD GFLOP/s"]
    # print(f"For CTV: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficient for the case with CTV GFLOP/s from FLOP/b:", correlation_coefficient)

l1.clear()
l2.clear()
for index, row in pd_avg.iterrows():
    bytefactor = row["Operational Intensity"] / row["Operational Intensity"]
    speed_up = row["DD GFLOP/s"] / row["SD GFLOP/s"]
    # print(f"For CTV: {row['Identifier']}, speed-up {speed_up} and load-store-decrease {bytefactor}")
    l1.append(speed_up)
    l2.append(bytefactor)
correlation_coefficient = np.corrcoef(l1, l2)[0, 1]
print("Correlation coefficient for the case without CTV GFLOP/s from FLOP/b:", correlation_coefficient)
"""