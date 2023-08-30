grouped_list = [list(range(i, i + 32)) for i in range(0, 1024, 32)]

# Printing the result to confirm
s = "{\n"
for group in grouped_list:
    s += "{" + ", ".join([str(x) for x in group]) + "},\n"
s += "}\n"

print(s)
