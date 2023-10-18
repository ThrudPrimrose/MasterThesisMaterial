import dace
import numpy
import cupy

from dace.dtypes import StorageType, ScheduleType

num_els = 200000

AliasedA = dace.data.Array(
    dace.float32, [num_els, 56, 9], may_alias=False, strides=(56*9, 1, 56))
AliasedB = dace.data.Array(
    dace.float32, [num_els, 9,  9], may_alias=False, strides=(9*9,  1, 9))
AliasedC = dace.data.Array(
    dace.float32, [num_els, 56, 9], may_alias=False, strides=(56*9, 1, 56))

customA = dace.data.Array(
    dace.float32, [num_els, 56, 9],
    storage=dace.StorageType.GPU_Global,
    strides=[56*9, 1, 56], start_offset=0)

customB = dace.data.Array(
    dace.float32, [num_els, 9, 9],
    storage=dace.StorageType.GPU_Global,
    strides=[9*9, 1, 9], start_offset=0)

customC = dace.data.Array(
    dace.float32, [num_els, 56, 9],
    storage=dace.StorageType.GPU_Global,
    strides=[56*9, 1, 9], start_offset=0)

"""
  for el in range(num_els):
    with dace.tasklet(dace.Language.CUDA):
      for i in range(56):
        acc = dace.data.Array(dace.float32, [9], strides=[1], start_offset=0,
                            storage=dace.StorageType.Register)

        for j in range(9):
          for k in range(9):
            acc[j] += A[el][i][j] * B[el][j][k]

        for j in range(9):
          C[el][i][j] += acc[j]
"""


@dace.program
def gemm(A, B, C):
  for i in dace.map[0:num_els]:
    C[i] += A[i] @ B[i]
  return C


sdfg = gemm.to_sdfg(customA, customB, customC)

# sdfg.add_transient('gA', [num_els, 56, 9], dace.float32, dace.StorageType.GPU_Global)
# sdfg.add_transient('gB', [num_els, 9,  9], dace.float32, dace.StorageType.GPU_Global)
# sdfg.add_transient('gC', [num_els, 56, 9], dace.float32, dace.StorageType.GPU_Global)

for _, desc in sdfg.arrays.items():
    if not desc.transient:
        desc.storage = dace.StorageType.GPU_Global

sdfg.apply_gpu_transformations()
# sdfg.save("gemm.sdfg")


# sdfg = gemm.to_sdfg(A, B, C)
# sdfg.save("gemm.sdfg")

sdfg.compile()
