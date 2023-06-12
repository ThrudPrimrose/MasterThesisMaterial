import os
import subprocess
from subprocess import Popen
import signal
import sys

# benchmark_x_y_cuda.py put them in folder ./cuda_code/*.py
code_path = "cuda_code"
exec_path = "cuda_executables"
report_path = "reports"
out_path = "stdout.txt"
err_path = "stderr.txt"

for path in [code_path, exec_path, report_path]:
    if not os.path.exists(path):
        os.mkdir(path)

out_file = open(out_path, "w")
err_file = open(err_path, "w")

stdout_as_str = ""
stderr_as_str = ""

def handler(signum, frame):
    out_file.write(stdout_as_str)
    err_file.write(stderr_as_str)

    out_file.close()
    err_file.close()
    exit(1)

signal.signal(signal.SIGINT, handler)

# Compile CUDA Kernels
for generator in ['benchmark_dense_sparse_cuda.py', 'benchmark_sparse_dense_cuda.py']:
    proc = subprocess.run(['python3', generator], stdout=subprocess.PIPE)
    stdout_as_str += proc.stdout.decode('utf-8')

for file in os.listdir(code_path):
    filename = os.fsdecode(file)
    filename_without_suffix = filename.split(".cu")[0]
    benchmark_identifier = filename.split(".cu")[0].split("benchmark_cuda_")[1]

    compile_command = f"nvcc --gpu-code=sm_86 --gpu-architecture=compute_86 --generate-line-info --resource-usage \
--ptxas-options=-v {code_path}/{filename} -o {exec_path}/{benchmark_identifier}"

    profile_command = f"ncu --export {report_path}/{benchmark_identifier}_rep --force-overwrite  --replay-mode kernel \
--kernel-name-base function --launch-skip-before-match 0  --sampling-interval auto --sampling-max-passes 5 \
--sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --set full \
--import-source no --check-exit-code yes --target-processes all {exec_path}/{benchmark_identifier}"

    print("Compile: ", filename)
    proc = Popen(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    proc.wait()
    stdout_as_str += proc.stdout.read().decode('utf-8')
    stderr_as_str += proc.stderr.read().decode('utf-8')

    print(f"Profile: {exec_path}/{benchmark_identifier}")
    proc = Popen(profile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    proc.wait()
    stdout_as_str += proc.stdout.read().decode('utf-8')
    stderr_as_str += proc.stderr.read().decode('utf-8')

out_file.write(stdout_as_str)
err_file.write(stderr_as_str)

out_file.close()
err_file.close()