
import random

fs = ""
for i in range(14):
    o1 = random_integer = random.randint(0, 3)
    o2 = random_integer = random.randint(0, 5)

    fs += f"\\filldraw [fill=LimeGreen!10!white,draw=LimeGreen!40!black] (0.8+{o1}*0.2,0+{o2}*0.2) rectangle (0.8+{o1}*0.2+0.2,0+{o2}*0.2+0.2);\n"

print(fs)
print("=============")

fs = set()
for i in range(9):
    o1 = random_integer = random.randint(0, 3)
    o2 = random_integer = random.randint(0, 3)

    fs.add(f"\\filldraw [fill=Cerulean!10!white,draw=Cerulean!40!black] (2+{o1}*0.2,1.6+{o2}*0.2) rectangle (2+0.2+{o1}*0.2,1.6+0.2+{o2}*0.2);\n")

print("".join(fs))