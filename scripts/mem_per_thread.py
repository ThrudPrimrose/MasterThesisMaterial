from params import *

FLOAT_SIZE = 4


for (row_a, col_a) in [(56,9), (32,32)]:
    def get_load_store_size(a_el_count, ctv):
        load_store = 0

        # If adressing is none then the matrix is loaded only 1 time in the whole batch
        # Read A
        if adressingA != "none" and not ctv:
            load_store += a_el_count

        # Read B
        if adressingB != "none":
            load_store += row_b * col_b

        # Write C
        if adressingC != "none":
            load_store += row_c * col_c

        # If Beta is not 0 then we need to read C
        if Beta != 0.0:
            load_store += row_c * col_c

        load_store *= FLOAT_SIZE
        return load_store


    def calculate_ops_dense(ctv):
        # Flops = (col_a + (row_a - 1)) * col_a * col_b;
        # FMA of 1 row = (2 * col_a * col_b)
        # Done for every row (row_a) *

        Flops = (col_b) * (2 * row_a * row_b)
        # Flops -= col_a * col_b # First row

        if Alpha != 1.0:
            Flops += row_c * col_c

        # Flops += row_a * col_a

        # Adding C to end result, row_c = row_a, col_c = col_b
        if Beta != 0.0:
            Flops += 2 * row_c * col_c

        load_store = get_load_store_size(row_a * col_a, ctv)

        return Flops, load_store


    """
    def calculate_ops_dense():
        # Matrix mul (ffma = 2 op)
        # This is Ravil's calculation of FLOP/s
        ops = (row_a) * (2 * (col_a - 1)) * col_b
        #2 * (k - 1) * m * n
        # I believe it should be this?
        # ops = (row_a) * (2 * col_a) * col_b
        # ops += row_c * col_c

        # Load A, B, C, store C, collective load of B, each thread then neesd a row of A
        # and a row of C
        load_store =  get_load_store_size(row_b * col_b)
        #return (ops, load_store)
        return ops, load_store
    """
    def calculate_ops_sparse_dense(typestr, ctv):
        if typestr == "band":
            block_count = int(row_a // col_a)
            elcount = 0
            for i in range(block_count):
                elcount += 2 * 2
                elcount += 3 * (col_a - 2)
            remainder = row_a - block_count*col_a
            if remainder > 2:
                elcount += 2 * 2
                elcount += 3 * (remainder - 2)
            else:
                elcount += 2 * remainder
            ops = col_b * elcount * 2
            if Alpha != 1.0:
                ops += row_c * col_c
            if Beta != 0.0:
                ops += 2 * row_c * col_c
            # ops += row_a * col_a
            # ops -= row_a * col_a
            load_store = get_load_store_size(elcount, ctv)
            return (ops, load_store)
        elif typestr == "full":
            return calculate_ops_dense(ctv)
        elif typestr == "random":
            el_count = int(sparsity * (row_a * col_a))
            ops = col_b * el_count * 2
            if Alpha != 1.0:
                ops += row_c * col_c
            if Beta != 0.0:
                ops += 2 * row_c * col_c
            # ops += row_a * col_a
            # ops -= row_a * col_a
            load_store = get_load_store_size(el_count, ctv)
            return (ops, load_store)
        elif typestr == "chequered":
            a = row_a // 2 + (row_a % 2)
            b = row_a // 2
            c = col_a // 2 + (col_a % 2)
            d = col_a // 2
            el_count = a * c + b * d
            ops = col_b * el_count * 2
            if Alpha != 1.0:
                ops += row_c * col_c
            if Beta != 0.0:
                ops += 2 * row_c * col_c
            # ops += row_a * col_a
            # ops -= row_a * col_a
            load_store = get_load_store_size(el_count, ctv)
            return (ops, load_store)
        else:
            raise Exception(typestr + " is undefined")


    for ctv in [True, False]:
        ctvstr = "(ctv)" if ctv else ""
        for a_type in a_matrix_types:
            flop, mem = calculate_ops_sparse_dense(a_type, ctv)
            tcount = col_b
            while(tcount % 32 != 0):
                tcount += 1
            print(f"Sparse-Dense {a_type}{ctvstr} {row_a}x{col_a}:\t{mem / tcount} bytes per thread")

    for ctv in [True, False]:
        ctvstr = "(ctv)" if ctv else ""
        flop, mem = calculate_ops_sparse_dense("full", ctv)
        tcount = row_a
        while(tcount % 32 != 0):
            tcount += 1
        print(f"Dense-Dense {ctvstr} {row_a}x{col_a}:\t{mem / tcount} bytes per thread")