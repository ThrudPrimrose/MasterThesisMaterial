from functools import reduce
import math
from copy import deepcopy
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from aux import *
from params import *
from itertools import combinations


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

stdout_dir = f"{data_dir}/TensorKernel3-LoadBoth-NewLoad"
if not os.path.exists(f"{stdout_dir}/plots"):
    os.mkdir(f"{stdout_dir}/plots")

pd_dataframes1 = list()
pd_dataframes2 = list()
pd_daraframes3 = list()

for r in range(runs):
    path = f"{stdout_dir}/run{r}.txt"
    report1 = []
    report2 = []
    report3 = []
    with open(path, "r") as file:
        state = "initial"
        cutensor_time = 0.0
        gemmforge_time = 0.0
        cutensor_efficiency_1 = 0.0
        gemmforge_efficiency_1 = 0.0
        gemmforge_gflops = 0.0
        cutensor_gflops = 0.0
        operational_intensity_1 = 0.0
        operational_intensity_2 = 0.0
        sizeStr = ""
        N = 0
        ID = 0
        indices = ""
        
        
        for i, line in enumerate(file):
          if state == "initial" and "Kernel" in line:
            state = "indices"
            tokens = line.split("Kernel")
            N = int(tokens[-1][-3:-1])
            ID = int(tokens[1].split(",")[0])
            print(N, ID)
          if state == "indices" and "Indices" in line:
            state = "gemmforge-time"
            tokens = line.split("Indices:")
            tripair = tokens[-1]
            sizeStr = tripair
            print(indices)
          if state == "gemmforge-time" and "Gemmforge Tensor Contraction took:" in line:
            state = "gemmforge-gflops"
            runtime = line.split("ms")[0].split("took: ")[-1]
            gemmforge_time = float(runtime)
            print(gemmforge_time)
          if state == "gemmforge-gflops" and "Gemmforge Kernel" in line:
            state = "operational-intensity-1"
            flops = line.split()[-1]
            gemmforge_gflops = float(flops)
            print(gemmforge_gflops)
          if state == "operational-intensity-1" and "Operational intensity:" in line:
            state = "gemmforge_efficiency-1"
            intense = line.split()[-1]
            operational_intensity_1 = float(intense)
            print(operational_intensity_1)
          if state == "gemmforge_efficiency-1" and "of roof w. respect to operational intensity achieved with Gemmforge" in line:
            state = "cutensor_efficiency-1"
            perc = line.split("%")[0]
            gemmforge_efficiency_1 = float(perc)
            print(gemmforge_efficiency_1)
          if state == "cutensor_efficiency-1" and "of roof w. respect to operational intensity achieved with cuTensor" in line:
            state = "final"
            perc = line.split("%")[0]
            cutensor_efficiency_1 = float(perc)
            print(cutensor_efficiency_1)
          if state == "final":
            state = "initial"


            total_gflops = gemmforge_gflops*gemmforge_time*1e-3
            cutensor_gflops = 1e-2 * cutensor_efficiency_1 * min(operational_intensity_1* peakMemoryBandwidthTheo, peakFLOPTheo)
            cutensor_time = 1e3 * total_gflops / cutensor_efficiency_1 
            r = [sizeStr, 
                  gemmforge_time, 
                  cutensor_time, 
                  gemmforge_efficiency_1,
                  cutensor_efficiency_1,
                  gemmforge_gflops, 
                  cutensor_gflops,
                  operational_intensity_1]

            cutensor_time = 0.0
            gemmforge_time = 0.0
            cutensor_efficiency_1 = 0.0
            gemmforge_gflops = 0.0
            cutensor_gflops = 0.0
            operational_intensity = 0.0
            cutensor_efficiency_1 = 0.0
            sizeStr = ""
            ID = 0
            indices = ""
            if int(N) == 16:
              report1.append(r)
            elif int(N)  == 31:
              report2.append(r)
            elif int(N)  == 32:
              report3.append(r)
            N = 0


    for report, pd_dataframes in [(report1, pd_dataframes1), (report2, pd_dataframes2), (report3, pd_daraframes3)]:
      tmp = pd.DataFrame(data=deepcopy(report), columns=[
          "Kernel",
          "Gemmforge Time",
          "cuTensor Time",
          "Gemmforge Efficiency",
          "cuTensor Efficiency",
          "Gemmforge GFLOP/s",
          "cuTensor GFLOP/s",
          "Operational Intensity",
      ])
      tmp1 = tmp.sort_values(by="Kernel").copy()
      print(f"DATAFRAME {r}:")
      print(tmp1)

      pd_dataframes.append(tmp1.copy())

#raise Exception(report1, report2, report3)
#raise Exception(pd_dataframes1)

pd_avgs = list()
pd_vars = list()

Ns = [16, 31, 32]

for dfs in [pd_dataframes1, pd_dataframes2, pd_daraframes3]:
  for df in dfs:
    df = df.sort_values(by='Kernel')

for pd_dataframes in [pd_dataframes1, pd_dataframes2, pd_daraframes3]:
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

  pd_vars.append(pd_var)
  pd_avgs.append(pd_avg)


def round_up_to_power_of_ten(n):
    if n == 0:
        return 1
    elif n < 0:
        return -round_up_to_power_of_ten(-n)
    else:
        power = math.ceil(math.log10(n))
        return 10 ** power


def plot_roofline(peak_memory_bandwidth, peak_floating_point_perf, 
                  title):
    plt.clf()
    fig, ax = plt.subplots(1, 3, figsize=(10, 6))  # Adjust the width and height as needed
    def roof(val):
        return min(peak_floating_point_perf,
                  (peak_memory_bandwidth * val))

    max_op = max(max(pd_avgs[1]["Operational Intensity"]),max(pd_avgs[2]["Operational Intensity"]),max(pd_avgs[0]["Operational Intensity"]))
    y_max = roof(max_op)
    #raise Exception(y_max)
    
    for m, order in enumerate([131,132,133]):
      pd_avg = pd_avgs[m]
      pd_var = pd_vars[m]


      # Set the width of each bar
      bar_width = 0.4
      # Calculate the positions for the bars
      kernel_strs = list()
      for kernel_str in pd_avg["Kernel"]:
        print(kernel_str)
        s = kernel_str.replace(" ", "")
        kernel_strs.append(s)

      gf_points = pd_avg["Gemmforge GFLOP/s"]

      x_positions1 = np.arange(len(kernel_strs), dtype=float)

      ax[m].bar(x_positions1 - 0.2,
              gf_points, color=dense_blue, 
              label="Gemmfogre", width = bar_width) #,
      ax[m].bar(x_positions1 + 0.2,
              pd_avg["cuTensor GFLOP/s"], color=nvidia_green, 
              label="cuTensor", width = bar_width) #width = bar_width,
      theo_roof_for_intensity = roof(max(pd_avg["Operational Intensity"]))
      #y = [roof(x) for x in pd_avg["Operational Intensity 2"]]
      #plt.bar(x_positions1, y, label="Roof Unfused", color="gray", linestyle="--")
      #y = [roof(x) for x in pd_avg["Operational Intensity 1"]]
      #plt.bar(x_positions1, y, label="Roof Fused", color="darkgray", linestyle="-.")

      for i, x_pos in enumerate(x_positions1):
          xpts = np.linspace(x_pos-0.6, x_pos+0.6)
          ypts = [roof(pd_avg["Operational Intensity"].iloc[i]) for x in xpts]
          ax[m].hlines(roof(pd_avg["Operational Intensity"].iloc[i]), x_pos-0.4, x_pos+0.4, color=["gray"], linestyles=["--"], label="Roofline" if i == 0 else "_nolabel_")
          #plt.hlines(roof(pd_avg["Operational Intensity"].iloc[i]), x_pos-0.4, x_pos+0.4, color=["gray"], linestyles=["solid"], label="Roofline" if i == 0 else "_nolabel_")


      std_dev_data = np.sqrt(pd_var["Gemmforge GFLOP/s"])
      yerr = 1.96 * (std_dev_data / np.sqrt(runs))
      ax[m].errorbar(x_positions1 -0.2, pd_avg["Gemmforge GFLOP/s"], 
                  yerr=yerr, fmt='none', ecolor='black', capsize=2, 
                  capthick=2, label='_nolegend_')
      std_dev_data = np.sqrt(pd_var["cuTensor GFLOP/s"])
      yerr = 1.96 * (std_dev_data / np.sqrt(runs))
      ax[m].errorbar(x_positions1 +0.2, pd_avg["cuTensor GFLOP/s"], 
                  yerr=yerr, fmt='none', ecolor='black', 
                  capsize=2, capthick=2,
                  label='_nolegend_')

      ax[m].set_xticks(x_positions1, kernel_strs,  rotation=70, ha='center', fontsize=9)
      ax[m].grid(visible=True, which="both", axis="both", linestyle=':')
      ax[m].set_xlabel(f"N = {Ns[m]}", fontsize=14)

      for i, value in enumerate(pd_avg["Gemmforge GFLOP/s"]):
          float_number = float(pd_avg["Gemmforge Efficiency"].iloc[i])
          formatted_number = f"{float_number:.1f}%"
          yloc = value + 1
          if float_number > 90.0:
              yloc = value + (100-float_number) + 25
          if m == 2 and (i == 3 or i == 1):
            ax[m].text(x_positions1[i] - bar_width/2, value + 230, str(formatted_number), ha='center', va='bottom', fontsize=8, c="gray")
          elif m == 1 and i == 1:
            ax[m].text(x_positions1[i] - bar_width/2, value + 210, str(formatted_number), ha='center', va='bottom', fontsize=8, c="gray")
          elif m == 1:
            ax[m].text(x_positions1[i] - bar_width/2, value + 100, str(formatted_number), ha='center', va='bottom', fontsize=8, c="gray")
          else:
            ax[m].text(x_positions1[i] - bar_width/2, value + 50, str(formatted_number), ha='center', va='bottom', fontsize=8, c="gray")

      for i, value in enumerate(pd_avg["cuTensor GFLOP/s"]):
          float_number = float(pd_avg["cuTensor Efficiency"].iloc[i])
          formatted_number = f"{float_number:.1f}%"
          ax[m].text(x_positions1[i] + bar_width, value + 50, str(formatted_number), ha='center', va='bottom', fontsize=8, c="gray")

      ax[m].set_ylim(0,y_max+100)

    ax[1].set_xlabel(f"N = {Ns[1]}\n Index permutations", fontsize=14)
    ax[2].legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=12)
    plt.suptitle(title, fontsize=14)
    #fig.text(1, 0.00, 'Loop Unrolling Parameters', ha='center', fontsize=14)
    #plt.xlabel('Loop Unrolling Parameters', fontsize=12)
    ax[0].set_ylabel('Performance (GFLOPs/s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/kernel-3-bar.pdf")
    plt.clf()

def round_up_to_power_of_ten(n):
    if n == 0:
        return 1
    elif n < 0:
        return -round_up_to_power_of_ten(-n)
    else:
        power = math.ceil(math.log10(n))
        return 10 ** power

"""
def plot_roofline2(peak_memory_bandwidth, peak_floating_point_perf, 
                  title):
    plt.clf()
    fig, ax = plt.subplots(figsize=(9, 6))  # Adjust the width and height as needed

    def roof(val):
        return min(peak_floating_point_perf,
                   (peak_memory_bandwidth * val))

    kernel_strs = list()
    for kernel_str in pd_avg["Kernel"]:
      s = kernel_str.replace(" ", "")
      kernel_strs.append(s)

    x_positions1 = np.arange(len(kernel_strs), dtype=float)

    plt.scatter(x_positions1,
            pd_avg["Gemmforge Efficiency K1"],  c="orangered", #linewidth=0.1, linestyle="--",
            label="Gemmfogre Kernel 1", marker="o")
    plt.scatter(x_positions1,
            pd_avg["Gemmforge Efficiency K2"], c="darkorchid", #linewidth=0.1, linestyle="--",
            label="Gemmfogre Kernel 2", marker="x")
    plt.scatter(x_positions1,
            pd_avg["Gemmforge Efficiency K3"],  c=sparse_rose, #linewidth=0.1, linestyle="--",
            label="Gemmfogre Kernel 3", marker="s")
    plt.scatter(x_positions1,
            pd_avg["Gemmforge Efficiency 2"], c=dense_blue, #linewidth=0.1, linestyle="--",
            label="Gemmfogre Kernel All", marker="v")
    plt.scatter(x_positions1,
            pd_avg["cuTensor Efficiency 2"], c=nvidia_green, #linewidth=0.1, linestyle="--",
            label="cuTensor Kernel All", marker="*")

    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=12)
    plt.ylim((0, 100.0))
    plt.title(title, fontsize=14)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('Loop Unrolling Parameters', fontsize=12)
    plt.ylabel('Performance (GFLOPs/s)', fontsize=12)
    plt.xticks(x_positions1, kernel_strs,  rotation=70, ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/kernel-3-eff.pdf")
    plt.clf()
"""

if save_plots:
    plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "")
    #plot_roofline2(peakMemoryBandwidthTheo, peakFLOPTheo,
    #              "")

for i, pd_avg in enumerate(pd_avgs):
  correlation_matrix = pd_avg.corr()

  plt.clf()
  plt.figure(figsize=(24, 18))  # Set the figure size
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)

  # Add labels and title
  plt.title('Correlation Matrix Heatmap')
  plt.tight_layout()
  plt.savefig(
      f"{stdout_dir}/plots/kernel-3-corr-{i}.pdf")