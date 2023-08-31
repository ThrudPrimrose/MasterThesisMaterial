import os
import subprocess
from subprocess import Popen
import signal
import sys
import shutil

from params import *

# benchmark_x_y_cuda.py put them in folder ./cuda_code/*.py
code_path = f"{scripts_dir}/cuda_code"
exec_path = f"{scripts_dir}/cuda_executables"
report_path = f"{scripts_dir}/reports"
out_path = f"{scripts_dir}/stdout.txt"
err_path = f"{scripts_dir}/stderr.txt"

#mode = "Profile"
mode = "Run"

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
# [f'{scripts_dir}/benchmark_dense_sparse_cuda.py', f'{scripts_dir}/benchmark_sparse_dense_cuda.py']
#for generator in [f'{scripts_dir}/benchmark_tensor.py']:
#for generator in [f"{scripts_dir}/benchmark_sparse_dense_cuda.py"]:
for generator in [f"{scripts_dir}/benchmark_dense_sparse_cuda.py"]:
#for generator in [f'{scripts_dir}/benchmark_dense_sparse_cuda.py', f'{scripts_dir}/benchmark_sparse_dense_cuda.py']:
    proc = subprocess.run(['python3', generator], stdout=subprocess.PIPE)
    stdout_as_str += proc.stdout.decode('utf-8')
    print("Call: ", generator)

for file in os.listdir(code_path):
    filename = os.fsdecode(file)
    filename_without_suffix = filename.split(".cu")[0]
    benchmark_identifier = filename.split(".cu")[0].split("benchmark_cuda_")[1]

    advanced_compile_command = f"{nvcc} -ccbin={gcc} -I{cuda_incl} -I{mat_lib_incl} -L{cuda_lib} -L{mat_lib_lib} --gpu-code=sm_86 --gpu-architecture=compute_86 --generate-line-info --resource-usage \
--ptxas-options=-v --source-in-ptx -lcublas -lcusparse {code_path}/{filename} -o {exec_path}/{benchmark_identifier}"

    if workaround:
        compile_command = advanced_compile_command
    else:
        compile_command = f"nvcc --gpu-code=sm_86 --gpu-architecture=compute_86 --generate-line-info --resource-usage \
--ptxas-options=-v --source-in-ptx -lcublas -lcusparse {code_path}/{filename} -o {exec_path}/{benchmark_identifier}"

    profile_command = f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{ld_library_path}; sudo env "PATH=$PATH" ncu -f -o {report_path}/{benchmark_identifier}_rep --set full \
--import-source yes {exec_path}/{benchmark_identifier}'

    run_command = f"{exec_path}/{benchmark_identifier}"

    command = profile_command if mode == "Profile" else run_command

    print("Compile: ", filename)
    proc = Popen(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        out_file.write(line)
    for line in proc.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
        err_file.write(line)
    proc.wait()
    #stdout_as_str += proc.stdout.read().decode('utf-8')
    #stderr_as_str += proc.stderr.read().decode('utf-8')

    print(f"{mode}: {exec_path}/{benchmark_identifier}")
    proc = Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    for line in proc.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
        err_file.write(line)
    proc.wait()
    #stdout_as_str += proc.stdout.read().decode('utf-8')
    #stderr_as_str += proc.stderr.read().decode('utf-8')

#out_file.write(stdout_as_str)
#err_file.write(stderr_as_str)

out_file.close()
err_file.close()
