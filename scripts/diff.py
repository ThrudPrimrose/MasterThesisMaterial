
def load_less_per(Asize, Bsize, Csize, perOfB, beta):
    load = Asize + Csize
    load_original = load
    load += int(perOfB*Bsize)
    load_original += Bsize
    if beta != 0.0:
        load += Csize
        load_original += Csize
    print(f"{round(float(load*100) / float(load_original), 4)}% of Original")
    return round(float(load*100) / float(load_original), 4)

l = list()
for i in range(100, 0, -1):
    load = load_less_per(32*32, 32*32, 32*32, float(i)/100.0, 0.0)
    if i < 100:
        print(f"Diff = {l[-1] - load}")
    l.append(load)

print("================================================================")

l = list()
for i in range(100, 0, -1):
    load = load_less_per(56*9, 9*9, 56*9, float(i)/100.0, 1.0)
    if i < 100:
        print(f"Diff = {l[-1] - load}")
    l.append(load)


load_less_per(32*32, 32*32, 32*32, 15.0/100.0, 0.0)
load_less_per(56*9, 9*9, 56*9, 15.0/100.0, 1.0)
load_less_per(56*9, 9*9, 56*9, 0.0/100.0, 1.0)

print(5.2/5.96)
print(5.5/5.66)
print(5.7/5.96)