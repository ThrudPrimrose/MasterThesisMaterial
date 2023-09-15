def print_matrix(matrix, rows, cols):
    for i in range(rows):
        for j in range(cols):
            print(matrix[i * cols + j], end=" ")
        print()


def transpose_matrix(matrix, rows, cols):
    # Initialize the transposed matrix AT with dimensions (cols x rows)
    transposed_matrix = [0] * (rows * cols)

    # Transpose the matrix
    for i in range(rows):
        for j in range(cols):
            # The index in the 1D array for the transposed matrix is calculated as follows:
            # Transpose formula: AT[j][i] = A[i][j]
            index_A = i * cols + j
            index_AT = j * rows + i
            transposed_matrix[index_AT] = matrix[index_A]

    return transposed_matrix


# Create a matrix A with dimensions 8x16 and store it in a 1D array (column-major format)
matrix_A = [i + j * 8 + 1 for j in range(16) for i in range(8)]

# Print the original matrix A
print("Matrix A:")
print_matrix(matrix_A, 8, 16)

# Transpose matrix A to get AT
rows_A = 8
cols_A = 16
matrix_AT = transpose_matrix(matrix_A, rows_A, cols_A)

# Print the transposed matrix AT
print("\nMatrix AT:")
print_matrix(matrix_AT, 16, 8)

"""
for (int i = 0; i < 8; ++i) {
    if (threadIdx.x < 16) {
        const int shrMemRow = i;
        const int shrMemCol = threadIdx.x;
        shrRegion0[shrMemRow + (shrMemCol * 16)] = glb_A[threadIdx.x + i * 16];
    }
}
"""

for i in range(0, 8):
    for Idx in range(0, 16):
        glbMemRow = Idx
        glbMemCol = i
        shrMemCol = glbMemRow
        shrMemRow = glbMemCol
        # shrOffset = (shrMemRow * 16) + shrMemCol
        glbOffset = Idx + i * 16
        # glbOffset = i * 8 + Idx
        # shrOffset = Idx * 16 + i
        shrOffset = shrMemRow + (shrMemCol * 8)
        print("shrOffset: ", shrOffset, "glbOffset: ", glbOffset, "val in transposed (shared): ", matrix_A[shrOffset],
              "val in not-transposed (glb)", matrix_AT[glbOffset])
