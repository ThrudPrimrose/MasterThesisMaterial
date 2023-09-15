import os
import subprocess

# Path to the CUDA file
cuda_file_path = "unroll_test.cu"

# Unroll counts
unroll_counts = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]


def bytes_to_human_readable(size):
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0


for unroll_count in unroll_counts:
    # Compile the CUDA program
    compile_command = ["nvcc", cuda_file_path, "-o", "unroll_test",
                       "-D", f"UNROLL_COUNT={unroll_count}",
                       "-D", f"ITERATIONS={100000}"]
    subprocess.run(compile_command, check=True)

    file_size = os.path.getsize("unroll_test")
    print(f"Size of executable (unroll={unroll_count}): {bytes_to_human_readable(file_size)}")

    # Run the compiled program
    run_command = ["./unroll_test"]
    subprocess.run(run_command, check=True)

# Unroll without definition
# Compile the CUDA program
compile_command = ["nvcc", cuda_file_path, "-o", "unroll_test"]
subprocess.run(compile_command, check=True)

file_size = os.path.getsize("unroll_test")
print(f"Size of executable (unroll={unroll_count}): {bytes_to_human_readable(file_size)}")

# Run the compiled program
run_command = ["./unroll_test"]
subprocess.run(run_command, check=True)
