def count_sectors_needed(intervals, line_length):
    sectors_needed = 0
    for interval_begin, interval_end in intervals:
        sbegin = interval_begin
        while sbegin % line_length != 0:
            sbegin -= 1
        sectors_needed += ((line_length-1)+interval_end-sbegin)//line_length
    return sectors_needed



for N in [16, 31, 32]:
    tensor_size = 2*N*N*N*4
    memory_size = 2*N*N*N*4
    load_region1 = [(0, N*N*N*4)]
    #print(load_region1)

    load_region2 = []
    for i in range(1,2):
        begin = N*N*N*i*4
        #for loopJump, leadingDim in [(N*N,N), (N, N*N)]:
        for j in range(N):
            for k in range(N):
                load_region2.append((begin + j*N*4 + k*N*N*4, begin + j*N*4 + k*N*N*4 + N*4))
        #print(load_region2)

    s1 = count_sectors_needed(load_region1, 4)
    s2 = count_sectors_needed(load_region2, 4)
    l1 = count_sectors_needed(load_region1, 128)
    l2 = count_sectors_needed(load_region2, 128)
    #print(f"Case 1 sectors needed for N = {N}:", s1)
    #print(f"Case 2 sectors needed for N = {N}:", s2)
    print(f"Case 1 cache lines (128-byte) needed for N = {N}:", l1)
    print(f"Case 2 cache lines (128-byte) needed for N = {N}:", l2)
    print(f"Case 1 sectors (32-byte) needed for N = {N}:", s1)
    print(f"Case 2 secctor (32-byte) needed for N = {N}:", s2)
    


#(3, 1) (5, 7, 3) (5, 11) (7)
# B goes 5*7
# 3 * load B(5,7) + load w(7)
# load A(3,1) + load(5,11)
load_per_kernel = 3*4 + 5*7*3*4 + 7*4 + 5*11*4
print(680//32)
worst_case_line_sizes = 1 + ((5*7*4*3 + 127)//128) + 1 + (5*11*4 + 127)//128

print(load_per_kernel)
print(worst_case_line_sizes)
print(worst_case_line_sizes*128)

print(5*11*4)
print(128*((127+5*11*4)//128))
print(5*128)
print(640/220)
print(4/1.7)