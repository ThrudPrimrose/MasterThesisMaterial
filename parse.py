from tabulate import tabulate
import os
import pandas as pd
import seaborn as sns

cur_dir = os.getcwd()
path = f"{cur_dir}/gemmforge_remote/stdout_only_run.txt"
#path = f"{cur_dir}/stdout.txt"

# The simple state machine
# Initial ->(1) Kernel Region ->(2) Initial
# For change (1) read: ==PROF== Connected to process
# In Kernel region we need to read:
# Dense x Dense kernel took ...
# Then
# Dense x Sparse kernel took ..., or Sparse x Dense kernel took ...
# Then we return to the initial state with
# ==PROF== Disconnected from process

report_dense_sparse = []
report_sparse_dense = []


row_a = 56
col_a = 9
row_b = 9
col_b = 9

def calculate_ops_dense():
    # Row a x Row b and then add per row
    # 1 row x 1 row -> col_a mul, col_a add
    # do for row_b count
    ops = row_a * 2 * col_a * row_b
    load_store = row_a * col_a + row_b * col_b + row_a * col_a
    return (ops, load_store)

sparsity = 0.25

def calculate_ops_sparse_dense(typestr):
    if typestr == "full":
        return calculate_ops_dense()
    elif typestr == "random":
        el_count = int(sparsity * (row_a * col_a))
        ops = 2 * el_count
        load_store = row_b * col_b + (el_count) + (row_a * col_a)
        return (ops, load_store)
    else:
        raise Exception("TODO for type: " + typestr)

def calculate_ops_dense_sparse(typestr):
    if typestr == "band":
        ops = 0
        ops += row_a * 4 # first and last row
        ops += row_a * 6 * (row_b - 2) # for every other row
        load_store = row_a * col_a + (3*row_b - 2) + row_a * col_a
        return (ops, load_store)
    elif typestr == "full":
        return calculate_ops_dense()
    elif typestr == "single_row":
        # Every row of A will be multiplied with a single row of B
        ops = row_a * col_a * 2
        load_store = row_a * col_a + row_b + row_a * col_a
        return (ops, load_store)
    elif typestr == "single_column":
        # Every row of A will be multiple with 1 element
        ops = row_a * col_a * 2
        load_store = row_a * col_a + row_b + row_a * col_a
        return (ops, load_store)
    elif typestr == "random":
        el_count = int(sparsity * (row_b * col_b))
        ops = 2 * el_count
        load_store = row_a * col_a + (el_count) + row_a * col_a
        return (ops, load_store)
    elif typestr == "chequered":
        a = row_b // 2 + (row_b % 2)
        b = row_b // 2
        c = col_b // 2 + (col_b % 2)
        d = col_b // 2
        el_count = a*b + c*d
        ops = row_a * el_count * 2
        load_store = el_count + row_a * col_a * 2
        return (ops, load_store)
    else:
        raise Exception(typestr + " is undefined")

with open(path, "r") as file:
    identifier = ""
    dense_time = 0.0
    sparse_time = 0.0
    speed_up = 0.0
    op_diff = 0.0
    load_diff = 0.0
    flops_per_byte_dd = 0.0
    flops_per_byte_ds = 0.0
    dense_sparse_type = ""
    sparse_dense_type = ""
    state = "initial"

    i = 1
    for line in file:
        print(i)

        if state == "initial" and "Gemm-Type:" in line:
            l = line.split("Type:")
            identifier = l[-1]
            # inside the identifier there has to be smth like: dense_sparse_At_mul_B_full_compiler_time_value
            # A{T if transA else NT}_{a_type}_B{T if transB else NT}_DenseXDense
            # or
            # A{T if transA else NT}_B{T if transB else NT}_{b_type}_DenseXDense
            tokens = identifier.split("_")
            assert(len(tokens) >= 4)
            if "DenseXSparse" in identifier:
                dense_sparse_type = tokens[2]
                if dense_sparse_type == "single":
                    dense_sparse_type += "_" + tokens[3]
                assert(dense_sparse_type != "")
                print(i, "DS: ", dense_sparse_type)
            elif "SparseXDense" in identifier:
                sparse_dense_type = tokens[1]
                if sparse_dense_type == "single":
                    sparse_dense_type += "_" + tokens[2]
                assert(sparse_dense_type != "")
                print(i, "SD: ", sparse_dense_type)
            state = "kernel"
            print(i, " initial -> kernel")
            i += 1
            continue
        elif state == "kernel" and "Dense x Dense kernel took" in line:
            duration = line.split("Dense x Dense kernel took ")[1][:-3]
            dense_time = float(duration)
            state = "kernel-2"
            print(i, " kernel -> kernel-2")
            i += 1
            continue
        elif state == "kernel-2" and "Dense x Sparse kernel took" in line:
            duration = line.split("Dense x Sparse kernel took ")[1][:-3]
            sparse_time = float(duration)
            state = "write-ds"
            print(i, " kernel-2 -> write-ds")
            i += 1
            continue
        elif state == "kernel-2" and "Sparse x Dense kernel took" in line:
            duration = line.split("Sparse x Dense kernel took ")[1][:-3]
            sparse_time = float(duration)
            state = "write-sd"
            print(i, " kernel-2 -> write-sd")
            i += 1
            continue
        elif state == "write-ds" and "Freeing device memory" in line: 
            speed_up =  dense_time / sparse_time
            dd_ops, dd_load_store = calculate_ops_dense()
            print(dense_sparse_type)
            ds_ops, ds_load_store = calculate_ops_dense_sparse(dense_sparse_type)
            op_diff = dd_ops / ds_ops
            load_store_diff = dd_load_store / ds_load_store
            flops_per_byte_dd = dd_ops / (4*dd_load_store)
            flops_per_byte_ds = ds_ops / (4*ds_load_store)
            speed_up_per = 100 - 100 / speed_up 
            report_dense_sparse.append([identifier, 
                                        round(dense_time, 4), 
                                        round(sparse_time, 4), 
                                        round(speed_up, 4),
                                        round(speed_up_per, 2),
                                        round(op_diff, 4),
                                        round(load_store_diff, 4),
                                        round(flops_per_byte_dd, 4),
                                        round(flops_per_byte_ds, 4),])
            state = "return"
            i += 1
            print(i, " write-ds -> return")
            continue
        elif state == "write-sd" and "Freeing device memory" in line:
            speed_up =  dense_time / sparse_time
            dd_ops, dd_load_store = calculate_ops_dense()
            print(sparse_dense_type)
            sd_ops, sd_load_store = calculate_ops_sparse_dense(sparse_dense_type)
            op_diff = dd_ops / sd_ops
            load_store_diff = dd_load_store / sd_load_store
            flops_per_byte_dd = dd_ops / (4*dd_load_store)
            flops_per_byte_sd = sd_ops / (4*sd_load_store)
            speed_up_per = speed_up * 100.0 - 100.0
            report_sparse_dense.append([identifier, 
                                        round(dense_time, 4), 
                                        round(sparse_time, 4), 
                                        round(speed_up, 4),
                                        round(speed_up_per, 2),
                                        round(op_diff, 4),
                                        round(load_store_diff, 4),
                                        round(flops_per_byte_dd, 4),
                                        round(flops_per_byte_sd, 4),])
            state = "return"
            i += 1
            print(i, " write-sd -> return")
            continue
        elif state == "return":
            identifier = ""
            dense_time = 0.0
            sparse_time = 0.0
            speed_up = 0.0
            state = "initial"
            substate = ""
            op_diff = 0.0
            load_store_diff = 0.0
            flops_per_byte_dd = 0.0
            flops_per_byte_ds = 0.0
            sparse_dense_type = ""
            dense_sparse_type = ""
            print(i, " return -> initial")
            i += 1
            continue
        else:
            i += 1

report_dense_sparse = list(sorted(report_dense_sparse, key = lambda x: x[0]))
report_sparse_dense = list(sorted(report_sparse_dense, key = lambda x: x[0]))

print(
    tabulate(report_dense_sparse, 
             headers=["Identifier", 
                      "DD Time", 
                      "DS Time",
                      "Speed-up",
                      "%",
                      "Flop. Ceil SU",
                      "LS Ceil SU",
                      "DD Flop/byte",
                      "DS Flop/byte"],
             tablefmt="github"))
print(
    tabulate(report_sparse_dense,
             headers=["Identifier", 
                      "DD Time", 
                      "DS Time",
                      "Speed-up",
                      "%",
                      "Flop. Ceil SU",
                      "LS Ceil SU",
                      "DD Flop/byte",
                      "DS Flop/byte"],
             tablefmt="github"))

p_ds = pd.DataFrame(data=report_dense_sparse, columns=[
                      "Identifier",
                      "DD Time", 
                      "DS Time",
                      "Speed-up",
                      "%",
                      "Flop. Ceil SU",
                      "LS Ceil SU",
                      "DD Flop/byte",
                      "DS Flop/byte"
                      ])

p_sd = pd.DataFrame(data=report_sparse_dense, columns=[
                      "Identifier",
                      "DD Time", 
                      "DS Time",
                      "Speed-up",
                      "%",
                      "Flop. Ceil SU",
                      "LS Ceil SU",
                      "DD Flop/byte",
                      "DS Flop/byte"
                      ])

cov = p_ds[[
    "Speed-up",
    "Flop. Ceil SU",
    "LS Ceil SU",
    "DD Flop/byte",
    "DS Flop/byte"
    ]].cov()
print(cov)
heatmap = sns.heatmap(cov, annot=True, fmt=".2f")
fig = heatmap.get_figure()
fig.tight_layout()
fig.savefig(f"{cur_dir}/heatmap-cov-ds.png") 
fig.clear()

corr =  p_ds[[
    "Speed-up",
    "Flop. Ceil SU",
    "LS Ceil SU",
    "DD Flop/byte",
    "DS Flop/byte"
    ]].corr()
print(corr)
heatmap = sns.heatmap(corr, annot=True, fmt=".2f")
fig = heatmap.get_figure()
fig.tight_layout()
fig.savefig(f"{cur_dir}/heatmap-corr-ds.png") 
fig.clear()

cov = p_sd[[
    "Speed-up",
    "Flop. Ceil SU",
    "LS Ceil SU",
    "DD Flop/byte",
    "DS Flop/byte"
    ]].cov()
print(cov)
heatmap = sns.heatmap(cov, annot=True, fmt=".2f")
fig = heatmap.get_figure()
fig.tight_layout()
fig.savefig(f"{cur_dir}/heatmap-cov-sd.png") 
fig.clear()

corr =  p_sd[[
    "Speed-up",
    "Flop. Ceil SU",
    "LS Ceil SU",
    "DD Flop/byte",
    "DS Flop/byte"
    ]].corr()
print(corr)
heatmap = sns.heatmap(corr, annot=True, fmt=".2f")
fig = heatmap.get_figure()
fig.tight_layout()
fig.savefig(f"{cur_dir}/heatmap-corr-sd.png") 