from yateto import *
import os

def add_trace(g):
    N = 8
    A = Tensor("A", (N, N))
    c = Tensor("c", (1, ))

    kernel = c <= A["ii"]
    g.add(name='kernel', ast=kernel, target="c")

def add_simple_tensor(g, target):
    N = 8
    A = Tensor('A', (N, 2*N, N))
    B = Tensor('B', (N, 2*N, N))
    C = Tensor('C', (N, 2*N, N))

    kernel = C['zxa'] <= A['zxy'] * B['yxa']
    g.add(name='kernel', ast=kernel, target=target)

def add_weird_tensor(g):
    N = 8
    A = Tensor('A', (N, 2*N, N))
    B = Tensor('B', (N, 2*N, N))
    C = Tensor('C', (N, 2*N))

    kernel = C['ij'] <= A['ijk'] * B['kjl']
    g.add(name='kernel', ast=kernel, target="cpu")

def add_complex_matrix(g):
    N = 8
    A = Tensor("A", (N, N))
    B = Tensor("B", (N, N))
    C = Tensor("C", (1))
    kernel = C <= A['ik'] * B['ki']
    g.add(name='kernel', ast=kernel, target="gpu")

def add_matrix(g, target):
    N = 9
    A = Tensor("A", (N, N))
    B = Tensor("B", (N, N))
    C = Tensor("C", (N, N))
    kernel = C['ij'] <= A['ik'] * B['kj']
    g.add(name='kernel', ast=kernel, target=target)

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
add_simple_tensor(generator, "cpu")
add_simple_tensor(generator, "gpu")
#add_complex_matrix(generator)
add_matrix(generator, "cpu")
add_matrix(generator, "gpu")
#add_weird_tensor(generator)
#add_trace(generator)

directory = os.path.dirname(os.path.abspath(__file__))
generator.generate(outputDir=directory,
                   gemm_cfg=GeneratorCollection([GemmForge(arch), Eigen(arch)]))
