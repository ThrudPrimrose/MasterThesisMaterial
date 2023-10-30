from functools import reduce
import math
from copy import deepcopy
import operator
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from aux import *
from params import *
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

def roof(val):
    return min(peakFLOPTheo,
                (peakMemoryBandwidthTheo * val))

FLOAT_SIZE = 4

stdout_dir = f"{data_dir}/TensorKernel2-LoadBoth-3"
if not os.path.exists(f"{stdout_dir}/plots"):
    os.mkdir(f"{stdout_dir}/plots")

pd_dataframes = list()

for r in range(runs):
    path = f"{stdout_dir}/run{r}.txt"
    report = []
    with open(path, "r") as file:
        state = "initial"
        cutensor_time = 0.0
        gemmforge_time = 0.0
        cutensor_efficiency_1 = 0.0
        cutensor_efficiency_2 = 0.0
        gemmforge_efficiency_1 = 0.0
        gemmforge_efficiency_2 = 0.0
        gemmforge_efficiency_k1 = 0.0
        gemmforge_efficiency_k2 = 0.0
        gemmforge_efficiency_k3 = 0.0
        gemmforge_gflops = 0.0
        cutensor_gflops = 0.0
        operational_intensity_1 = 0.0
        operational_intensity_2 = 0.0
        sizeStr = ""
        k = 0
        l = 0
        m = 0
        p = 0
        q = 0
        tokens = []
        
        
        for i, line in enumerate(file):
          if state == "initial" and "Dimensions" in line:
            state = "gemmforge-time"
            tokens = line.split("Dimensions:")
            sizeStr = ",".join(tokens[-1].split(","))
            print(sizeStr)
            dims = sizeStr.split(", ")
            #raise Exception(dims)
            #Old kernels had order k-p-m-l-q
            # New
            k = int(dims[0])
            l = int(dims[1])
            m = int(dims[2])
            p = int(dims[3])
            q = int(dims[4])
            # Old
            #k = int(dims[0])
            #q = int(dims[1])
            #m = int(dims[2])
            #l = int(dims[3])
            #p = int(dims[4])
            kernel_str = str(k) + "," + str(l) + "," + str(m) + "," + str(p) + "," + str(q)
            kpad = "0" if k<10 else ""
            lpad = "0" if l<10 else ""
            mpad = "0" if m<10 else ""
            ppad = "0" if p<10 else ""
            qpad = "0" if q<10 else ""
            kernel_int = int(str(k) + lpad + str(l) + mpad + str(m) + ppad + str(p) + qpad + str(q))
          if state == "gemmforge-time" and "Gemmforge Tensor Contraction took:" in line:
            state = "gemmforge-gflops"
            runtime = line.split("ms")[0].split("took: ")[-1]
            gemmforge_time = float(runtime)
            print(gemmforge_time)
          if state == "gemmforge-gflops" and "Gemmforge Theoretical Fused Kernel" in line:
            state = "operational-intensity-1"
            flops = line.split()[-1]
            gemmforge_gflops = float(flops)
            print(gemmforge_gflops)
          if state == "operational-intensity-1" and "Operational Theoretical Fused intensity:" in line:
            state = "operational-intensity-2"
            intense = line.split()[-1]
            operational_intensity_1 = float(intense)
            print(operational_intensity_1)
          if state == "operational-intensity-2" and "Operational intensity:" in line:
            state = "gemmforge_efficiency-1"
            intense = line.split()[-1]
            operational_intensity_2 = float(intense)
            print(operational_intensity_2)
          if state == "gemmforge_efficiency-1" and "of roof w. respect to operational intensity achieved with Gemmforge" in line:
            state = "gemmforge_efficiency-2"
            perc = line.split("%")[0]
            gemmforge_efficiency_1 = float(perc)
            print(gemmforge_efficiency_1)
          if state == "gemmforge_efficiency-2" and "roof w. respect to unfused operational intensity achieved with Gemmforge" in line:
            state = "gemmforge_efficiency-k1"
            perc = line.split("%")[0]
            gemmforge_efficiency_2 = float(perc)
            print(gemmforge_efficiency_2)
          if state == "gemmforge_efficiency-k1" and "of roof w. respect to Kernel1 intensity achieved with Gemmforge" in line:
            state = "gemmforge_efficiency-k2"
            perc = line.split("%")[0]
            gemmforge_efficiency_k1 = float(perc)
            print(gemmforge_efficiency_k1)
          if state == "gemmforge_efficiency-k2" and "of roof w. respect to Kernel2 intensity achieved with Gemmforge" in line:
            state = "gemmforge_efficiency-k3"
            perc = line.split("%")[0]
            gemmforge_efficiency_k2 = float(perc)
            print(gemmforge_efficiency_k2)
          if state == "gemmforge_efficiency-k3" and "of roof w. respect to Kernel3 intensity achieved with Gemmforge" in line:
            #state = "final"
            state = "cutensor_efficiency-1"
            perc = line.split("%")[0]
            gemmforge_efficiency_k3 = float(perc)
            print(gemmforge_efficiency_k3)
        
          if state == "cutensor_efficiency-1" and "of roof w. respect to operational intensity achieved with cuTensor" in line:
            state = "cutensor_efficiency-2"
            perc = line.split("%")[0]
            cutensor_efficiency_1 = float(perc)
            print(cutensor_efficiency_1)
          if state == "cutensor_efficiency-2" and "of roof w. respect to unfused operational intensity achieved with cuTensor" in line:
            state = "final"
            perc = line.split("%")[0]
            cutensor_efficiency_2 = float(perc)
            print(cutensor_efficiency_2)

          if state == "final":
            state = "initial"
            ls1 = k*p + k*q + q*p
            ls1*=4
            ls2 = k*p*m*2 + l*m + k*p*l
            ls2*=4
            ls3 = k*p*m*2 + m + k*p
            ls3*=4
            ops1 = k*q*p*2
            ops2 = l*m*k*p*2 + k*p*m
            ops3 = k*m*p*2
            ops = ops1 + ops2 + ops3
            ls = k*p*m*4 + m*1 + k*q*1 + q*p*1 + k*p*l*1 + l*m*1 + k*p*2 
            ls *= 4
            
            intensity1 = ops1 / ls1 
            intensity2 = ops2 / ls2
            intensity3 = ops3 / ls3
            intensity =  ops / ls
            r1 = roof(intensity1)
            r2 = roof(intensity2)
            r3 = roof(intensity3)
            

            r = [kernel_str, kernel_int,
                           gemmforge_time, 
                           cutensor_time, 
                           gemmforge_efficiency_1,
                           gemmforge_efficiency_2, 
                           cutensor_efficiency_1,
                           cutensor_efficiency_2,
                           gemmforge_efficiency_k1,
                           gemmforge_efficiency_k2,
                           gemmforge_efficiency_k3,
                           gemmforge_gflops, 
                           gemmforge_gflops*cutensor_efficiency_1/gemmforge_efficiency_1,
                           operational_intensity_1,
                           operational_intensity_2]
            # List of 5 variables
            variables = [k,l,m,p,q]

            # Iterate through all possible subsets of the variables
            for red in range(1, len(variables) + 1):
                for subset in combinations(variables, red):
                    print(subset)
                    sd = reduce(operator.mul, subset, 1)
                    r.append(float(sd))

            r.append(min(k,l,m,p,q))
            r.append(max(k,l,m,p,q))
            r.append(np.var([k,l,m,p,q]))
            r.append(float(p)/float(q))
            op_intensity_1 = k*q*p*2 / (4*(k*p + k*q + q*p))
            op_intensity_2 = (l*m*k*p*2 + k*p*m) / (4*(k*p*m + l*m + k*p*l))
            op_intensity_3 = (k*m*p*3) / (4*(k*p*m + m + k*p))
            k*p
            r.append(op_intensity_1)
            r.append(k*p/q)
            r.append(1/(k*l*m*p*q))
            r.append(p%32)
            lines_needed_k1 = (k*p+31)//32+ (k*q+31)//32 + (q*p+31)//32
            load_difference = lines_needed_k1*32 / (k*p + k*q + q*p)
            r.append(lines_needed_k1)
            r.append(load_difference)
            r.append(k%32)
            t_rounded_up = (k*q + q*p + k*p) / (k+31)//32
            r.append(t_rounded_up)
            r.append(op_intensity_2)
            r.append(op_intensity_3)
            r.append(p/k)
            r.append(q/k)
            r.append(p*q/k)
            r.append(t_rounded_up* q/k)
            r.append(l/(k*p))
            r.append(k/(p*m))
            r.append((k*p*m + m + k*p)/(k*m))
            r.append(p/m)
            r.append((k*p*m*2 + m + k*p)/k*m)
            r.append(max(m, k*p)/(k*m))
            report.append(r)
            
            """
            variables = ["K", "L", "M", "P", "Q"]
            for r in range(1, len(variables) + 1):
                for subset in combinations(variables, r):
                    sd = "".join(subset)
                    print(f"\"{sd}\",")
            raise Exception("UWU")
            """

            cutensor_time = 0.0
            gemmforge_time = 0.0
            cutensor_efficiency_1 = 0.0
            gemmforge_efficiency_1 = 0.0
            cutensor_efficiency_2 = 0.0
            gemmforge_efficiency_2 = 0.0
            gemmforge_efficiency_k1 = 0.0
            gemmforge_efficiency_k2 = 0.0
            gemmforge_efficiency_k3 = 0.0
            gemmforge_gflops = 0.0
            cutensor_gflops = 0.0
            operational_intensity = 0.0
            cutensor_efficiency_1 = 0.0
            cutensor_efficiency_2 = 0.0
            sizeStr = ""
            kernel_str = ""
            k = 0
            l = 0
            m = 0
            p = 0
            q = 0
            tokens = []
            print(report)


    tmp1 = pd.DataFrame(data=deepcopy(report), columns=[
        "Kernel", "Kernel Int",
        "Gemmforge Time",
        "cuTensor Time",
        "Gemmforge Efficiency 1",
        "Gemmforge Efficiency 2",
        "cuTensor Efficiency 1",
        "cuTensor Efficiency 2",
        "Gemmforge Efficiency K1",
        "Gemmforge Efficiency K2",
        "Gemmforge Efficiency K3",
        "Gemmforge GFLOP/s",
        "cuTensor GFLOP/s",
        "Operational Intensity 1",
        "Operational Intensity 2",
        "K",
        "L",
        "M",
        "P",
        "Q",
        "KL",
        "KM",
        "KP",
        "KQ",
        "LM",
        "LP",
        "LQ",
        "MP",
        "MQ",
        "PQ",
        "KLM",
        "KLP",
        "KLQ",
        "KMP",
        "KMQ",
        "KPQ",
        "LMP",
        "LMQ",
        "LPQ",
        "MPQ",
        "KLMP",
        "KLMQ",
        "KLPQ",
        "KMPQ",
        "LMPQ",
        "KLMPQ",
        "MIN",
        "MAX",
        "VAR",
        "Q-1P",
        "OP INTENSITY 1",
        "KQ-1P",
        "NUM_ELEMENTS",
        "P%32",
        "LINED NEEDED K1",
        "LOAD DIFFERENCE",
        "K%32",
        "LOAD OP PER THREAD",
        "OP INTENSITY 2",
        "OP INTENSITY 3",
        "K-1P",
        "K-1Q",
        "PK-1Q",
        "K-1QLOPT",
        "LK-1P-1",
        "KP-1M-1",
        "Tensor Load K3",
        "PM-1",
        "Load Per Thread K3",
        "Percentage of active Load Threads K3"
    ])
    tmp1 = tmp1.sort_values(by="Kernel Int").copy()
    print(f"DATAFRAME {r}:")
    print("\n".join(tmp1))

    x = list()
    for i in tmp1["Kernel Int"]:
      x.append("(" + str(i) + ")")
    print(",\n".join(x))
    #raise Exception("UWU")

    pd_dataframes.append(tmp1.copy())

"""
s = ""
for i in pd_dataframes[0]["Kernel"]:
  s += "(" + i + "),\n"
raise Exception(s)
"""

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


#pd_avg.sort_values(by="KMP", inplace=True)
#pd_var.sort_values(by="KMP", inplace=True)

#pd_avg = pd_avg[["Kernel", "Operational Intensity", "Gemmforge GFLOP/s"]]
#pd_avg = pd_avg[["Kernel", "Operational Intensity", "cuTensor GFLOP/s"]]

#pd_var = pd_var[["Kernel", "Operational Intensity", "Gemmforge GFLOP/s"]]
#pd_var = pd_var[["Kernel", "Operational Intensity", "cuTensor GFLOP/s"]]

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
    fig, ax = plt.subplots(figsize=(9, 7))  # Adjust the width and height as needed

    def roof(val):
        return min(peak_floating_point_perf,
                   (peak_memory_bandwidth * val))

    # Set the width of each bar
    bar_width = 0.4
    # Calculate the positions for the bars
    kernel_strs = list()
    for kernel_str in pd_avg["Kernel"]:
      print(kernel_str)
      s = kernel_str.replace(" ", "")
      kernel_strs.append(s)

    #gf_points = pd_avg["Gemmforge GFLOP/s"]

    x_positions1 = np.arange(len(kernel_strs), dtype=float)
    pd_avg.sort_values(by='Operational Intensity 2', inplace=True)
    gf_points = pd_avg["Gemmforge GFLOP/s"]
  
    plt.bar(x_positions1 - 0.2,
            gf_points, color=dense_blue, 
            label="Gemmfogre", width = bar_width) #,
    plt.bar(x_positions1 + 0.2,
            pd_avg["cuTensor GFLOP/s"], color=nvidia_green, 
            label="cuTensor", width = bar_width) #width = bar_width,
    theo_roof_for_intensity = roof(max(pd_avg["Operational Intensity 2"]))
    #y = [roof(x) for x in pd_avg["Operational Intensity 2"]]
    #plt.bar(x_positions1, y, label="Roof Unfused", color="gray", linestyle="--")
    #y = [roof(x) for x in pd_avg["Operational Intensity 1"]]
    #plt.bar(x_positions1, y, label="Roof Fused", color="darkgray", linestyle="-.")

    for i, x_pos in enumerate(x_positions1):
        xpts = np.linspace(x_pos-1.2, x_pos+0.6)
        ypts = [roof(pd_avg["Operational Intensity 2"].iloc[i]) for x in xpts]
        plt.hlines(roof(pd_avg["Operational Intensity 2"].iloc[i]), x_pos-0.5, x_pos+0.5, color=["darkgray"], linestyles=["--"], label="Roofline" if i == 0 else "_nolabel_")
        #plt.hlines(roof(pd_avg["Operational Intensity 1"].iloc[i]), x_pos-0.4, x_pos+0.4, color=["gray"], linestyles=["solid"], label="Roofline" if i == 0 else "_nolabel_")


    std_dev_data = np.sqrt(pd_var["Gemmforge GFLOP/s"])
    yerr = 1.96 * (std_dev_data / np.sqrt(runs))
    plt.errorbar(x_positions1 -0.2, pd_avg["Gemmforge GFLOP/s"], 
                 yerr=yerr, fmt='none', ecolor='black', 
                 capsize=1, capthick=1, label='_nolegend_')
    std_dev_data = np.sqrt(pd_var["cuTensor GFLOP/s"])
    yerr = 1.96 * (std_dev_data / np.sqrt(runs))
    plt.errorbar(x_positions1 +0.2, pd_avg["cuTensor GFLOP/s"], 
                 yerr=yerr, fmt='none', ecolor='black', 
                 capsize=1, capthick=1,
                 label='_nolegend_')
    plt.ylim(0, 5100)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('Tensor Dimensions', fontsize=12)
    plt.ylabel('Performance (GFLOPs/s)', fontsize=12)
    plt.xticks(x_positions1, kernel_strs,  rotation=90, ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/kernel-2-bar.pdf")
    plt.clf()

def round_up_to_power_of_ten(n):
    if n == 0:
        return 1
    elif n < 0:
        return -round_up_to_power_of_ten(-n)
    else:
        power = math.ceil(math.log10(n))
        return 10 ** power


def plot_roofline2(peak_memory_bandwidth, peak_floating_point_perf, 
                  title):
    plt.clf()
    fig, ax = plt.subplots(figsize=(9, 7))  # Adjust the width and height as needed

    def roof(val):
        return min(peak_floating_point_perf,
                   (peak_memory_bandwidth * val))

    #kernel_strs = list()
    #for kernel_str in pd_avg["Kernel"]:
    #  s = kernel_str.replace(" ", "")
    #  kernel_strs.append(s)

    x_positions1 = np.arange(len(pd_avg["Kernel"]), dtype=float)
    #pd_avg.sort_values(by='Gemmforge Efficiency 2', inplace=True)
    pd_avg.sort_values(by='Kernel Int', inplace=True)

    plt.scatter(x_positions1,
            pd_avg["Gemmforge Efficiency K1"],  c="orangered", #linewidth=0.1, linestyle="--",
            s=15, label="Gemmfogre\nKernel 1", marker="o")
    plt.scatter(x_positions1,
            pd_avg["Gemmforge Efficiency K2"], c="darkorchid", #linewidth=0.1, linestyle="--",
            s=15, label="Gemmfogre\nKernel 2", marker="x")
    plt.scatter(x_positions1,
            pd_avg["Gemmforge Efficiency K3"],  c=sparse_rose, #linewidth=0.1, linestyle="--",
            s=15, label="Gemmfogre\nKernel 3", marker="s")
    plt.scatter(x_positions1,
            pd_avg["Gemmforge Efficiency 2"], c=dense_blue, #linewidth=0.1, linestyle="--",
            s=15, label="Gemmfogre\nKernel All", marker="v")
    plt.scatter(x_positions1,
            pd_avg["cuTensor Efficiency 2"], c=nvidia_green, #linewidth=0.1, linestyle="--",
            s=15, label="cuTensor\nKernel All", marker="*")
    
    """
    for key, color in [("Gemmforge Efficiency K1", "orangered"), 
                       ("Gemmforge Efficiency K2", "darkorchid"), 
                       ("Gemmforge Efficiency K3", sparse_rose), 
                       ("Gemmforge Efficiency 2", dense_blue), 
                       ("cuTensor Efficiency 2", nvidia_green)]:
      plt.plot(x_positions1, 
              [pd_avg[key].mean() for _ in x_positions1], c=color, linewidth=1,
              label ="_nolabel_", linestyle="--")
    """
    plt.legend(loc='center left', bbox_to_anchor=(0.10, -0.225), ncol=6,
                        fontsize=9)
    desired_y_ticks = [i for i in range(5, 105, 5)]  # Define desired tick positions
    plt.yticks(desired_y_ticks)
    plt.xticks(x_positions1)
    plt.tight_layout()
    #plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=12)
    plt.ylim((0, 101.0))
    plt.title(title, fontsize=14)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('\n\n\nTensor Dimensions', fontsize=12)
    plt.ylabel('Performance Relative to Roofline (%)', fontsize=12)
    plt.xticks(x_positions1, pd_avg["Kernel"],  rotation=90, ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/kernel-2-eff.pdf")
    plt.clf()



def plot_roofline4(peak_memory_bandwidth, peak_floating_point_perf, 
                  title):
    plt.clf()
    fig, ax = plt.subplots(figsize=(9, 7))  # Adjust the width and height as needed

    pd_avg.sort_values(by="Operational Intensity 2", inplace = True)
    def roof(val):
        return min(peak_floating_point_perf,
                   (peak_memory_bandwidth * val))

    kernel_strs = list()
    for kernel_str in pd_avg["Kernel"]:
      s = kernel_str.replace(" ", "")
      kernel_strs.append(s)

    x_positions1 = np.arange(len(kernel_strs), dtype=float)
    #pd_avg.sort_values(by='Gemmforge Efficiency 2', inplace=True)


    # Extract the 'x' and 'y' columns from the DataFrame
    x = pd_avg['Operational Intensity 2'].values.reshape(-1, 1)
    y = pd_avg['Gemmforge Efficiency 2'].values

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(x, y)

    # Make predictions using the model
    y_pred = model.predict(x)

    pd_avg2 = pd_avg.drop(len(pd_avg["Operational Intensity 2"])-1)
    #raise Exception(pd_avg)
    # Extract the 'x' and 'y' columns from the DataFrame
    x = np.array([x*x for x in pd_avg2['Operational Intensity 2'].values]).reshape(-1, 1)
    y = pd_avg2['Gemmforge Efficiency 2'].values

    # Create a linear regression model
    model2 = LinearRegression()

    # Fit the model to the data
    model2.fit(x, y)

    # Make predictions using the model
    y_pred_2 = model.predict(x)
    r_squared = r2_score(pd_avg["Gemmforge Efficiency 2"], y_pred)
    print("R_square:", r_squared)
    
    # Plot the original data and the regression line
    #plt.scatter(x, y, label='Data')
    #plt.scatter(x_positions1, 
    #            y_pred, color='lightgray', label='Predicted Efficency', s=15, marker="x")
    #plt.scatter(x_positions1[:-1], 
    #            y_pred_2, color='gray', label='Predicted Efficency 2', s=15, marker="+")
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.legend()
    #plt.show()

    # Get the coefficients of the linear regression model
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"Slope (Coefficient): {slope}")
    print(f"Intercept: {intercept}")


    plt.scatter(x_positions1,
            pd_avg["Gemmforge Efficiency 2"], c=dense_blue, #linewidth=0.1, linestyle="--",
            s=15, label="Gemmfogre Kernel All", marker="v")
    plt.scatter(x_positions1,
            pd_avg["cuTensor Efficiency 2"], c=nvidia_green, #linewidth=0.1, linestyle="--",
            s=15, label="cuTensor Kernel All", marker="*")

    degrees_of_freedom = 10 - 1
    alpha = 1 - 0.95
    t_score = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

    # Calculate the margin of error
    margin_of_error = [t_score * math.sqrt(x) / math.sqrt(10) for x in pd_var["Gemmforge Efficiency 2"]]


    plt.errorbar(x_positions1, pd_avg["Gemmforge Efficiency 2"], 
                 yerr=margin_of_error , fmt='none', ecolor=dense_blue, 
                 capsize=5, capthick=1,
                 label='_nolegend_')

    degrees_of_freedom = 10 - 1
    alpha = 1 - 0.95
    t_score = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

    # Calculate the margin of error
    margin_of_errorw = [t_score * math.sqrt(x) / math.sqrt(10) for x in pd_var["cuTensor Efficiency 2"]]


    plt.errorbar(x_positions1, pd_avg["cuTensor Efficiency 2"], 
                 yerr=margin_of_errorw , fmt='none', ecolor=nvidia_green, 
                 capsize=5, capthick=1,
                 label='_nolegend_')

    """
    for key, color in [("Gemmforge Efficiency K1", "orangered"), 
                       ("Gemmforge Efficiency K2", "darkorchid"), 
                       ("Gemmforge Efficiency K3", sparse_rose), 
                       ("Gemmforge Efficiency 2", dense_blue), 
                       ("cuTensor Efficiency 2", nvidia_green)]:
      plt.plot(x_positions1, 
              [pd_avg[key].mean() for _ in x_positions1], c=color, linewidth=1,
              label ="_nolabel_", linestyle="--")
    """
    plt.legend(loc='center left', bbox_to_anchor=(0.28, -0.225), ncol=6,
                        fontsize=9)
    desired_y_ticks = [i for i in range(5, 105, 5)]  # Define desired tick positions
    plt.yticks(desired_y_ticks)
    plt.xticks(x_positions1)
    plt.tight_layout()
    #plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=12)
    plt.ylim((0, 101.0))
    plt.title(title, fontsize=14)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('\n\n\nTensor Dimensions', fontsize=12)
    plt.ylabel('Performance Relative to Roofline (%)', fontsize=12)
    plt.xticks(x_positions1, kernel_strs,  rotation=90, ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/kernel-2-eff-2.pdf")
    plt.clf()


def plot_roofline3(peak_memory_bandwidth, peak_floating_point_perf, 
                  title, sort_key, access_key, save_key, color_key, marker_key):
    plt.clf()
    fig, ax = plt.subplots(figsize=(9, 7))  # Adjust the width and height as needed



    pd_avg.sort_values(by=sort_key, inplace = True)
    #pd_avg.sort_values(by="LINED NEEDED K1", inplace = True)
    #pd_avg.sort_values(by="LOAD DIFFERENCE", inplace = True)

    def roof(val):
        return min(peak_floating_point_perf,
                   (peak_memory_bandwidth * val))

    kernel_strs = list()
    for kernel_str in pd_avg["Kernel"]:
      s = kernel_str.replace(" ", "")
      kernel_strs.append(s)

    x_positions1 = np.arange(len(kernel_strs), dtype=float)
    #pd_avg.sort_values(by='Gemmforge Efficiency 2', inplace=True)


    plt.scatter(x_positions1,
            pd_avg[access_key],  c=color_key, #linewidth=0.1, linestyle="--",
            s=15, label=f"Gemmfogre Kernel {save_key[1:]}", marker=marker_key)

    degrees_of_freedom = 10 - 1
    alpha = 1 - 0.95
    t_score = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

    # Calculate the margin of error
    margin_of_error = [t_score * math.sqrt(x) / math.sqrt(10) for x in pd_var[access_key]]


    plt.errorbar(x_positions1, pd_avg[access_key], 
                 yerr=margin_of_error , fmt='none', ecolor=color_key, 
                 capsize=5, capthick=1,
                 label='_nolegend_')
    """
    for key, color in [(access_key, "orangered"), 
                       ("Gemmforge Efficiency K2", "darkorchid"), 
                       ("Gemmforge Efficiency K3", sparse_rose), 
                       ("Gemmforge Efficiency 2", dense_blue), 
                       ("cuTensor Efficiency 2", nvidia_green)]:
      plt.plot(x_positions1, 
              [pd_avg[key].mean() for _ in x_positions1], c=color, linewidth=1,
              label ="_nolabel_", linestyle="--")
    """
    plt.legend(loc='center left', bbox_to_anchor=(0.38, -0.225), ncol=6,
                        fontsize=9)
    desired_y_ticks = [i for i in range(5, 105, 5)]  # Define desired tick positions
    plt.yticks(desired_y_ticks)
    plt.xticks(x_positions1)
    plt.tight_layout()
    #plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=12)
    plt.ylim((0, 101.0))
    plt.title(title, fontsize=14)
    plt.grid(visible=True, which="both", axis="both", linestyle=':')
    plt.xlabel('\n\n\nTensor Dimensions', fontsize=12)
    plt.ylabel('Performance Relative to Roofline (%)', fontsize=12)
    plt.xticks(x_positions1, kernel_strs,  rotation=90, ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(
        f"{stdout_dir}/plots/kernel-2-{save_key}-eff.pdf")
    plt.clf()


if save_plots:
    plot_roofline(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "")
    plot_roofline2(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "")
    #raise Exception("AAAAAAAAAA")
    plot_roofline3(peakMemoryBandwidthTheo, peakFLOPTheo, "", "K-1Q", "Gemmforge Efficiency K1", "k1", "orangered", "o")
    plot_roofline3(peakMemoryBandwidthTheo, peakFLOPTheo, "", "Q-1P", "Gemmforge Efficiency K2", "k2", "darkorchid", "x")
    plot_roofline3(peakMemoryBandwidthTheo, peakFLOPTheo, "", "Load Per Thread K3", "Gemmforge Efficiency K3", "k3", sparse_rose, "s")
    plot_roofline4(peakMemoryBandwidthTheo, peakFLOPTheo,
                  "")


correlation_matrix = pd_avg.corr()

plt.clf()
plt.figure(figsize=(30, 24))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)

# Add labels and title
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(
    f"{stdout_dir}/plots/kernel-2-corr.pdf")


X = pd_avg[["Load Per Thread K3"]]
y = pd_avg["Gemmforge Efficiency K3"]

model = LinearRegression()
model.fit(X, y)

from sklearn.metrics import mean_squared_error
y_pred = model.predict(X)
r_squared = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
print(r_squared, mse)