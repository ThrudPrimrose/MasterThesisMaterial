import gemmforge
from params import *

mat_a = gemmforge.DenseMatrix(num_rows=row_a,
                              num_cols=col_a,
                              addressing="pointer_based",
                              bbox=[0, 0, row_a, col_a])

mat_b = gemmforge.DenseMatrix(num_rows=row_b,
                              num_cols=col_b,
                              addressing="none",
                              bbox=[0, 0, row_b, col_b])

mat_c = gemmforge.DenseMatrix(num_rows=row_c,
                              num_cols=col_c,
                              addressing="strided",
                              bbox=[0, 0, row_c, col_c])

vm = gemmforge.vm_factory(arch="sm_86", backend="cuda", fp_type="float")

dense_gen = gemmforge.GemmGenerator(
    vm=vm, kernel_type=gemmforge.GemmKernelType.AUTO)

dense_gen.set(False, False, mat_a, mat_b, mat_c, alpha=2.0,
              beta=3.0, base_name="adressing_test")

dense_gen.generate()
print(dense_gen.get_kernel())
print(dense_gen.get_launcher())
print(dense_gen.get_launcher_header())
