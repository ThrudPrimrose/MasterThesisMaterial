from yateto import *
import os

def add_weird_tensor(g):
    N = 8
    A = Tensor('A', (N, N, N))
    B = Tensor('B', (N,))
    C = Tensor('C', (N, N))
    D = Tensor('D', (N, N))
    E = Tensor('E', (N, N, N))
    F = Tensor('F', (N, N))

    kernel = A['kpm'] <= A['kpm'] + B['m'] * C['kq'] * D['qp'] + E['kpl'] * F['lm']
    g.add(name='kernel', ast=kernel, target="gpu")


arch = useArchitectureIdentifiedBy(
    host_arch="shsw", device_arch="ssm_86", device_backend="cuda")
generator = Generator(arch)
#add_simple_tensor(generator)
#add_complex_matrix(generator)
#add_matrix(generator)
add_weird_tensor(generator)

directory = os.path.dirname(os.path.abspath(__file__))
generator.generate(outputDir=directory,
                   gemm_cfg=GeneratorCollection([GemmForge(arch)]))


