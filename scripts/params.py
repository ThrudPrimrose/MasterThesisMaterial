#row_a = 56
#col_a = 9
#row_b = 9
#col_b = 9
#row_c = 56
#col_c = 9

# Change if wanted
row_a = 56
col_a = 9
row_b = 9
col_b = 9
row_c = 56
col_c = 9

# Change if wanted
Alpha = 1.0
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

# Change when GPU changes in GB/s and GFLOP/s
peakMemoryBandwidthTheo = 760.08 #176.032
peakFLOPTheo = 29767.7 #4329.47

# Dont change
nvidia_green = "#76b900"
my_blue = "#6ab2ca"
sparse_rose = "#c74375"
dense_blue = my_blue

# Change always
workaround = False
scripts_dir = "/home/primrose/Work/MasterThesisMaterial/scripts"
data_dir = f"{scripts_dir}/../data"
input_name = "stdout32-32-beta0.txt"

ld_library_path = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/lib64"
nvcc = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/bin/nvcc"
gcc = "gcc-11"
cuda_incl = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/include"
mat_lib_incl = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/include"
cuda_lib = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8/lib64"
mat_lib_lib = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/lib64"
#nvcc = "nvcc"
#cuda_incl = "."
#gcc = "clang"
#ld_library_path = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/lib64"
#mat_lib_incl ="."