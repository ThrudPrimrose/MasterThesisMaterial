import pytaco as pt
import numpy as np
from functools import reduce
import operator
from tensor_params import *
from numba import cuda

N = 32
shapeA = (N, N)
shapeB = (N, N, N)
shapew = (N, )
shapeC = (N, N)
sizeA = reduce(operator.mul, shapeA, 1)
sizeB = reduce(operator.mul, shapeB, 1)
sizew = reduce(operator.mul, shapew, 1)
sizeC = reduce(operator.mul, shapeC, 1)
def get_available_mem_on_gpu():
    meminfo = cuda.current_context().get_memory_info()
    return meminfo[0]

def get_suggested_num_elements():
    #1 pointer extra needed per element
    per_el_size = (sizeA + sizeB + sizew + sizeC) * 4 + 16

    available_mem = get_available_mem_on_gpu()
    can_fit_els = available_mem // per_el_size
    lower = int(0.90 * can_fit_els)
    return lower

num_els = get_suggested_num_elements()
#C['ij'] <= C['ij'] + A['lj'] * B['ikl'] * w['k']
print(num_els)




# Generate two random vectors using NumPy and pass them into TACO
A = pt.from_array(np.random.uniform(size=sizeA))
B = pt.from_array(np.random.uniform(size=sizeB))
C = pt.from_array(np.random.uniform(size=sizeC))
w = pt.from_array(np.random.uniform(size=sizew))

# Declare index vars
i, j, k, l, b = pt.get_index_vars(5)

# Define the SpMV computation
C[i, j, b] = C[i, j, b] + A[l, j, b] * B[i, k, l, b] * w[k, b]

# Perform the SpMV computation and write the result to file
C.compile()