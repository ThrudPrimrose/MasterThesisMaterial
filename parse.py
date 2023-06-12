from tabulate import tabulate

stdout = open("stdout.txt", "r")
lines = stdout.readlines()

# The simple state machine
# Initial ->(1) Kernel Region ->(2) Initial
# For change (1) read: ==PROF== Connected to process
# In Kernel region we need to read:
# Dense x Dense kernel took ...
# Then
# Dense x Sparse kernel took ..., or Sparse x Dense kernel took ...
# Then we return to the initial state with
# ==PROF== Disconnected from process

state = "initial"

report_dense_sparse = []
report_sparse_dense = []

with open("stdout.txt", "r") as file:
    identifier = ""
    dense_time = 0.0
    sparse_time = 0.0
    speed_up = 0.0
    substate = ""

    i = 0
    for line in file:
        if state == "initial" and "==PROF== Connected to process" in line:
            l = line.split("/")
            identifier = l[-1][:-2]
            state = "kernel"
        if state == "kernel" and "Dense x Dense kernel took" in line:
            duration = line.split("Dense x Dense kernel took ")[1][:-3]
            dense_time = float(duration)
        if state == "kernel" and "Dense x Sparse kernel took" in line:
            duration = line.split("Dense x Sparse kernel took ")[1][:-3]
            sparse_time = float(duration)
            substate = "ds"
        if state == "kernel" and "Sparse x Dense kernel took" in line:
            duration = line.split("Sparse x Dense kernel took ")[1][:-3]
            sparse_time = float(duration)
            substate = "sd"
        if state == "kernel" and "==PROF== Disconnected from process" in line:
            speed_up =  dense_time / sparse_time
            if substate == "ds":
                report_dense_sparse.append([identifier, 
                                            round(dense_time, 4), 
                                            round(sparse_time, 4), 
                                            round(speed_up, 4)])
            if substate == "sd":
                report_sparse_dense.append([identifier, 
                                            round(dense_time, 4), 
                                            round(sparse_time, 4), 
                                            round(speed_up, 4)])
            identifier = ""
            dense_time = 0.0
            sparse_time = 0.0
            speed_up = 0.0
            state == "initial"
            substate == ""
        i += 1

print(
    tabulate(report_dense_sparse, 
             headers=["Identifier", 
                      "Dense x Dense Kernel Time", 
                      "Dense x Sparse Kernel Time",
                      "Speed-up"],
             tablefmt="github"))
print(
    tabulate(report_sparse_dense,
             headers=["Identifier", 
                      "Dense x Dense Kernel Time", 
                      "Sparse x Dense Kernel Time",
                      "Speed-up"],
             tablefmt="github"))