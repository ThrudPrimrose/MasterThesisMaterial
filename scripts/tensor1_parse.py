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
kernel = ""

for r in range(runs):
    path = f"{stdout_dir}/run{r}.txt"
    report = []
    with open(path, "r") as file:
        state = "initial"
        cutensor = 0.0
        gemmforge = 0.0
        cutensor_efficiency = 0.0
        gemmforge_efficiency = 0.0
        gflops = 0.0
        operational_intensity = 0.0
        sizeStr = ""
        
        
        for i, line in enumerate(file):
          if state == "initial" and "compute the kernel" in line and "with Gemmforge" in line:
            state = "check-shape"
            tokens = line.split(", with")
            kernel = tokens[0].split("kernel: ")[-1]
            print(kernel)
          if state == "check-shape"  and "Shapes and dims:" in line:
            tokens = line.split("dims:")[-1]
            t = tokens[1:-1]
            sizeStr = " ".join(t)
            state = "check-gemmforge"
            print(sizeStr)
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
            report.append([sizeStr, gemmforge, 
                           cutensor, gemmforge_efficiency, 
                           cutensor_efficiency, gflops, gflops*gemmforge/cutensor,
                           operational_intensity])
            state = "initial"
            cutensor = 0.0
            gemmforge = 0.0
            cutensor_efficiency = 0.0
            gemmforge_efficiency = 0.0
            gflops = 0.0
            operational_intensity = 0.0
            sizeStr = ""
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
        raise Exception("NO")
    else:
        power = math.ceil(math.log10(n))
        return 10 ** power

from matplotlib.markers import MarkerStyle

def plot_roofline_2(peak_memory_bandwidth, peak_floating_point_perf, 
                  title, gemmforge_points, cutensor_points, pd_avg):
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the width and height as needed
    shapes = [MarkerStyle("1"), MarkerStyle("+"), MarkerStyle("x"), MarkerStyle("4"), MarkerStyle("2"), MarkerStyle("3")]

    def roof(val):
        return min(peak_floating_point_perf,
                   (peak_memory_bandwidth * val))

    xpts = np.linspace(0, 40, 250)
    plt.plot(xpts, [roof(x) for x in xpts], label="Roofline")
    for i in range(6):
        plt.scatter(gemmforge_points["Operational Intensity"].iloc[i],
                gemmforge_points["Gemmforge GFLOP/s"].iloc[i], 
                color=dense_blue,
                marker=shapes[i],
                label=gemmforge_points["Kernel"].iloc[i].replace(" ", ""), s=120 if i != 2 else 70)
        plt.scatter(cutensor_points["Operational Intensity"].iloc[i],
                cutensor_points["cuTensor GFLOP/s"].iloc[i], 
                color=nvidia_green,
                marker=shapes[i],
                label = "_nolabel_", s=120 if i != 2 else 70)
    ymax = max(max(gemmforge_points["Gemmforge GFLOP/s"]), max(cutensor_points["cuTensor GFLOP/s"]))
    xmax = max(max(gemmforge_points["Operational Intensity"]), max(cutensor_points["Operational Intensity"]))
    #cutensor_points["Kernel"].iloc[i].replace(" ", "")
    kernel_strs = list()

    #new_xticks.append(1)
    #new_xticks.append(10)
    #new_labels.append(1)
    #new_labels.append(10)

    plt.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=12)
    plt.yscale("log")
    plt.xscale("log")
    print(xmax)
    plt.ylim((0, round_up_to_power_of_ten(ymax)))
    plt.xlim((0, round_up_to_power_of_ten(xmax)))
    plt.title(title + "\n" + kernel, fontsize=14)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('Operational Intensity (FLOP/byte)', fontsize=12)
    plt.ylabel('Performance (GFLOPs/s)', fontsize=12)
    #plt.xticks(new_xticks, new_labels, rotation=70, ha="left")

    new_xticks = list()
    new_labels = list()

    for i, kernel_str in enumerate(gemmforge_points["Kernel"]):
      #a_c_perm = kernel_str[3:6]
      #b_perm = kernel_str[-5:-2]
      #kernel_strs.append(a_c_perm + "-" + b_perm)
      #TODO:
      kernel_strs.append(kernel_str)
      new_xticks.append(gemmforge_points["Operational Intensity"].iloc[i])
      new_labels.append(gemmforge_points["Kernel"].iloc[i])
    #plt.xticks(new_xticks, new_labels, rotation=90, ha="left")

    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/Log-Kernel-1-roofline.pdf")
    plt.clf()

def plot_roofline(peak_memory_bandwidth, peak_floating_point_perf, 
                  title, gemmforge_points, cutensor_points, pd_avg):
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 7))  # Adjust the width and height as needed

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
            label="Gemmforge")
    plt.bar(x_positions2,
            cutensor_points["cuTensor GFLOP/s"], color=nvidia_green, width = bar_width,
            label = "cuTensor")
    ymax = max(max(gemmforge_points["Gemmforge GFLOP/s"]), max(cutensor_points["cuTensor GFLOP/s"]))
    xmax = max(max(gemmforge_points["Operational Intensity"]), max(cutensor_points["Operational Intensity"]))
    theo_roof_for_intensity = roof(max(gemmforge_points["Operational Intensity"]))
    line_values = np.full(len(gemmforge_points["Kernel"]) + 1, theo_roof_for_intensity)
    #plt.plot(x_positions_f, line_values, marker='', color='gray', linestyle='--', label='Roofline')
    for i, x_pos in enumerate(x_positions1):
        xpts = np.linspace(x_pos, x_pos+0.5)
        ypts = [roof(gemmforge_points["Operational Intensity"].iloc[i]) for x in xpts]
        plt.hlines(roof(gemmforge_points["Operational Intensity"].iloc[i]), x_pos-0.2, x_pos+0.6, color=["gray"], linestyles=["--"], label="Roofline" if i == 0 else "_nolabel_")

    kernel_strs = list()
    for kernel_str in gemmforge_points["Kernel"]:
      s = kernel_str.replace(" ", "")
      toks = s.split(",C(")
      s = toks[0] + "\n" + "C(" + toks[1]
      kernel_strs.append(s)

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
        float_number = 100*gemmforge_points["Gemmforge GFLOP/s"].iloc[i] / roof(gemmforge_points["Operational Intensity"].iloc[i])
        formatted_number = f"{float_number:.1f}%"
        yloc = value + 1
        if float_number > 90.0:
            yloc = value + (100-float_number) + 25
        plt.text(x_positions1[i] + bar_width/2, yloc, str(formatted_number), ha='center', va='bottom', fontsize=8, c="gray")

    for i, value in enumerate(cutensor_points["cuTensor GFLOP/s"]):
        float_number = 100*cutensor_points["cuTensor GFLOP/s"].iloc[i] /  roof(gemmforge_points["Operational Intensity"].iloc[i])
        formatted_number = f"{float_number:.1f}%"
        plt.text(x_positions2[i] + bar_width/2, value + 1, str(formatted_number), ha='center', va='bottom', fontsize=8, c="gray")


    #plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    #plt.yscale("log")
    #plt.xscale("log")
    #plt.ylim((0, round_up_to_power_of_ten(ymax)))
    #plt.xlim((0, round_up_to_power_of_ten(xmax)))
    plt.title(title, fontsize=14)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('Kernel Type', fontsize=12)
    plt.ylabel('Performance (GFLOPs/s)', fontsize=12)
    
    plt.xticks(x_positions1 + (bar_width) / 2, kernel_strs, rotation=70, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/LoG-Kernel-1-bar-roofline.pdf")
    plt.clf()

if save_plots:
    plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "Roofline Model for Dense Loop-over-GEMM kernel:", 
                  gemmforge_points=gemmforge_points,
                  cutensor_points=cutensor_points,
                  pd_avg=pd_avg)
    plot_roofline_2(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "Roofline Model for Dense Loop-over-GEMM kernel:", 
                  gemmforge_points=gemmforge_points,
                  cutensor_points=cutensor_points,
                  pd_avg=pd_avg)