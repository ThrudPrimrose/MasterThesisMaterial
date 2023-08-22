from yateto import *
import os


def add_simple_tensor(g):
    N = 8
    A = Tensor('A', (N, 2*N, N))
    B = Tensor('B', (N, 2*N, N))
    C = Tensor('C', (N, 2*N, N))

    kernel = C['ijl'] <= A['ijk'] * B['kjl']
    g.add(name='kernel', ast=kernel, target="gpu")

def add_matrix(g):
    N = 8
    A = Tensor("A", (N, N))
    B = Tensor("B", (N, N))
    C = Tensor("C", (N, N))
    kernel = C['ij'] <= A['ik'] * B['kj']
    g.add(name='kernel', ast=kernel, target="gpu")

def add_example_tensor(g):
    N = 8
    A = Tensor('A', (N, N))
    B = Tensor('B', (N, N, N))
    w = Tensor('w', (N,))
    C = Tensor('C', (N, N))

    kernel = C['ij'] <= C['ij'] + A['lj'] * B['ikl'] * w['k']
    g.add(name='kernel', ast=kernel, target="gpu")


arch = useArchitectureIdentifiedBy(
    host_arch="shsw", device_arch="ssm_86", device_backend="cuda")
generator = Generator(arch)
add_simple_tensor(generator)
#add_matrix(generator)

directory = os.path.dirname(os.path.abspath(__file__))
generator.generate(outputDir=directory,
                   gemm_cfg=GeneratorCollection([GemmForge(arch)]))
