import os

# row_a = 56
# col_a = 9
# row_b = 9
# col_b = 9
# row_c = 56
# col_c = 9

runs = 10

# Change if wanted
row_a = 32
col_a = 32
row_b = 32
col_b = 32
row_c = 32
col_c = 32

# Change if wanted
Alpha = 1.5
Beta = 0.0

# Change if wanted
non_zero_ratio = 0.15
sparsity = non_zero_ratio

# Do not change
adressingA = "strided"
adressingB = "strided"
adressingC = "strided"

# Always False while benchmarking
debug = False

# Change if wanted
b_matrix_types = ["band", "chequered", "full", "random"]
#b_matrix_types = ["full", "random"]
a_matrix_types = b_matrix_types
# b_matrix_types = ["random"]

# Change when GPU changes in GB/s and GFLOP/s
# peakMemoryBandwidthTheo = 176.032 #760.08 #176.032
# peakFLOPTheo = 4329.47 #29767.7 #4329.47
peakMemoryBandwidthTheo = 760.08 
peakFLOPTheo = 29767.7 
tensorPeakBandwidth =  760.08
tensorPeakFLOP = 29767.7 

# Dont change
nvidia_green = "#76b900"
my_blue = "#6ab2ca"
sparse_rose = "#c74375"
dense_blue = my_blue
#pre = "TensorKernel1-NewLoopHeuristics"
#pre = "TensorKernel2-NoLoopHeuristics"
#pre = "TensorKernel3-NewLoadStrategy"
pre = f"SparseDense-A{row_a}x{col_a}-B{row_b}-{col_b}-a{Alpha}-b{Beta}-Filtered"

# Change always
workaround = False
scripts_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = f"{scripts_dir}/../data"

stdout_dir = f"{data_dir}/{pre}"
save_plots = True
debug_print = True

ld_library_path = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/lib64"
nvcc = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/bin/nvcc"
gcc = "gcc-11"
cuda_incl = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/include"
mat_lib_incl = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/include"
cuda_lib = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/lib64"
mat_lib_lib = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/lib64"
# nvcc = "nvcc"
# cuda_incl = "."
# gcc = "clang"
# ld_library_path = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/lib64"
# mat_lib_incl ="."

#flow_generator_list = [f"{scripts_dir}/benchmark_dense_sparse_cuda.py"]
flow_generator_list = [f"{scripts_dir}/benchmark_sparse_dense_cuda.py"]
#flow_generator_list = [f"{scripts_dir}/benchmark_product.py"]
#flow_generator_list = [f"{scripts_dir}/benchmark_tensor1.py"]
#flow_generator_list = [f"{scripts_dir}/benchmark_product_2.py"]
#flow_generator_list = [f"{scripts_dir}/benchmark_loop.py"]
#flow_generator_list = [f"{scripts_dir}/benchmark_tensor_unfuse.py"]
#flow_generator_list = [f"{scripts_dir}/benchmark_tensor2.py"]
#flow_generator_list = [f"{scripts_dir}/benchmark_tensor3.py"]
write_output = True

sort = False