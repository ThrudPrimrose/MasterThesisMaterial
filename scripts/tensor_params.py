import os


runs = 10

Alpha = 1.0
Beta = 1.0

# Change when GPU changes in GB/s and GFLOP/s
# peakMemoryBandwidthTheo = 176.032
# peakFLOPTheo = 4329.47
peakMemoryBandwidthTheo = 760.08
peakFLOPTheo = 29767.7

# Dont change
nvidia_green = "#76b900"
my_blue = "#6ab2ca"
sparse_rose = "#c74375"
dense_blue = my_blue

# Change always
workaround = False
scripts_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{scripts_dir}/../data"
stdout_dir = f"{data_dir}/Tensor"

ld_library_path = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/lib64"
nvcc = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/bin/nvcc"
gcc = "gcc-11"
cuda_incl = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/include"
mat_lib_incl = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/include"
cuda_lib = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/lib64"
mat_lib_lib = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/lib64"

flow_generator_list = [f"{scripts_dir}/benchmark_tensor1.py"]
write_output = False
