import os
import subprocess
from subprocess import Popen
import signal
import sys
import shutil

# benchmark_x_y_cuda.py put them in folder ./cuda_code/*.py
cur_dir = os.getcwd()
code_path = f"{cur_dir}/cuda_code"
exec_path = f"{cur_dir}/cuda_executables"
report_path = f"{cur_dir}/reports"
out_path = f"{cur_dir}/stdout.txt"
err_path = f"{cur_dir}/stderr.txt"

mode = "Profile"
#mode = "Run"

for path in [code_path, exec_path, report_path]:
    if not os.path.exists(path):
        os.mkdir(path)

for path in [code_path, exec_path]:
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
        
if mode == "Profile":
    if os.path.exists(report_path):
        shutil.rmtree(report_path)
        os.mkdir(report_path)

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
for generator in [f'{cur_dir}/benchmark_dense_sparse_cuda.py', f'{cur_dir}/benchmark_sparse_dense_cuda.py']:
    proc = subprocess.run(['python3', generator], stdout=subprocess.PIPE)
    stdout_as_str += proc.stdout.decode('utf-8')
    print("Call: ", generator)

for file in os.listdir(code_path):
    filename = os.fsdecode(file)
    filename_without_suffix = filename.split(".cu")[0]
    benchmark_identifier = filename.split(".cu")[0].split("benchmark_cuda_")[1]

    compile_command = f"nvcc --gpu-code=sm_86 --gpu-architecture=compute_86 --generate-line-info --resource-usage \
--ptxas-options=-v --source-in-ptx {code_path}/{filename} -o {exec_path}/{benchmark_identifier}"

    profile_command = f'sudo env "PATH=$PATH" ncu -f -o {report_path}/{benchmark_identifier}_rep --set full \
--import-source yes {exec_path}/{benchmark_identifier}'

    run_command = f"{exec_path}/{benchmark_identifier}"

    command = profile_command if mode == "Profile" else run_command

    print("Compile: ", filename)
    proc = Popen(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    proc.wait()
    stdout_as_str += proc.stdout.read().decode('utf-8')
    stderr_as_str += proc.stderr.read().decode('utf-8')

    print(f"{mode}: {exec_path}/{benchmark_identifier}")
    proc = Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    proc.wait()
    stdout_as_str += proc.stdout.read().decode('utf-8')
    stderr_as_str += proc.stderr.read().decode('utf-8')

out_file.write(stdout_as_str)
err_file.write(stderr_as_str)

out_file.close()
err_file.close()
