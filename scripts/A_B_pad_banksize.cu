
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <iomanip>
#include <cublas_v2.h>
#include <cusparse.h>

#define CHECK_ERR checkErr(__FILE__, __LINE__)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

std::string PrevFile = "";
int PrevLine = 0;

void transpose_matrix(int M, int K, float *A, float *AT)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            AT[i * K + j] = A[j * M + i];
        }
    }
}

void checkErr(const std::string &File, int Line)
{
#ifndef NDEBUG
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
    {
        std::cout << std::endl
                  << File
                  << ", line " << Line
                  << ": " << cudaGetErrorString(Error)
                  << " (" << Error << ")"
                  << std::endl;

        if (PrevLine > 0)
            std::cout << "Previous CUDA call:" << std::endl
                      << PrevFile << ", line " << PrevLine << std::endl;
        throw;
    }
    PrevFile = File;
    PrevLine = Line;
#endif
}

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
        }                                                                  \
    }

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
        }                                                              \
    }

// Dense x Dense Kernel
__global__ void
    __launch_bounds__(64)
        kernel_A_B_DenseXDense(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
    unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
    if (batchID < numElements)
    {
        bool isFlagsProvided = (flags != nullptr);
        bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
        if (allowed)
        {
            const float *const __restrict__ glb_A = &A[batchID * 504 + 0 + A_extraOffset];
            const float *const __restrict__ glb_B = &B[batchID * 81 + 0 + B_extraOffset];
            float *const __restrict__ glb_C = &C[batchID * 504 + 0 + C_extraOffset];
            float reg0[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            __shared__ __align__(8) float totalShrMem[81];
            float *localShrMem0 = &totalShrMem[81 * threadIdx.y];

            float *shrRegion0 = &localShrMem0[0];
            // using ExtendedPatchLoader
            {
                shrRegion0[threadIdx.x + 0] = glb_B[threadIdx.x + 0];
                if (threadIdx.x < 17)
                {
                    shrRegion0[threadIdx.x + 64] = glb_B[threadIdx.x + 64];
                }
            }
            __syncthreads();
            if (threadIdx.x < 56)
            {
                float value;

#pragma unroll
                for (int k = 0; k < 9; ++k)
                {
                    value = glb_A[threadIdx.x + k * 56];

#pragma unroll
                    for (int n = 0; n < 9; ++n)
                    {
                        reg0[n] += value * shrRegion0[k + 9 * n];
                    }
                }
            }
            if (threadIdx.x < 56)
            {
#pragma unroll
                for (int n = 0; n < 9; ++n)
                {
                    glb_C[threadIdx.x + 56 * n] = 1.5f * reg0[n];
                }
            }
        }
    }
}

__global__ void
    __launch_bounds__(32)
        kernel_A_B_full_DenseXDense2(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
    unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
    if (batchID < numElements)
    {
        bool isFlagsProvided = (flags != nullptr);
        bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
        if (allowed)
        {
            const float *const __restrict__ glb_A = &A[batchID * 504 + 0 + A_extraOffset];
            const float *const __restrict__ glb_B = &B[batchID * 81 + 0 + B_extraOffset];
            float *const __restrict__ glb_C = &C[batchID * 504 + 0 + C_extraOffset];
            float reg0[56] = {0.0f};
            __shared__ __align__(128) float totalShrMem1[11 * 9];
            __shared__ __align__(128) float totalShrMem2[64 * 9];
            //__shared__ __align__(128) float totalShrMem3[64 * 9];
            float *localShrMem0 = &totalShrMem1[11 * 9 * threadIdx.y];
            float *localShrMem1 = &totalShrMem2[64 * 9 * threadIdx.y];
            //float *localShrMem2 = &totalShrMem3[56 * 9 * threadIdx.y];
            float *shrRegion2 = &localShrMem1[0];

            // Load Matrix B
            float *shrRegion0 = &localShrMem0[0];
            if (threadIdx.x < 9)
            {
#pragma unroll 9
                for (int i = 0; i < 9; i++)
                {
                    shrRegion0[threadIdx.x + 11 * i] = glb_B[threadIdx.x + 9 * i];
                }
            }
            // Load Matrix A
            float *shrRegion1 = &localShrMem1[0];
#pragma unroll 9
            for (int i = 0; i < 9; i++)
            {
                shrRegion1[threadIdx.x + 64*i] = glb_A[threadIdx.x + 56*i];
                if (threadIdx.x < 56 - 32)
                {
                    shrRegion1[threadIdx.x + 32 + 64*i] = glb_A[threadIdx.x + 32 + 56*i];
                }
            }

            __syncthreads();
            if (threadIdx.x < 9)
            {
                float value;
#pragma unroll 9
                for (int k = 0; k < 9; ++k)
                {
                    value = shrRegion0[threadIdx.x * 11 + k];

#pragma unroll 56
                    for (int m = 0; m < 56; m++)
                    {
                        reg0[m] += value * shrRegion1[k * 64 + m];
                    }
                }
            }
            if (threadIdx.x < 9)
            {
#pragma unroll 56
                for (int m = 0; m < 56; ++m)
                {
                    shrRegion2[threadIdx.x * 64 + m] = 1.5f * reg0[m];
                }
            }
            __syncthreads();
#pragma unroll 9
            for (int i = 0; i < 9; i++)
            {
                glb_C[threadIdx.x + 56 * i] = shrRegion2[threadIdx.x + 64 * i];
                if (threadIdx.x < 56 - 32){
                    glb_C[threadIdx.x + 32 + 56 * i] = shrRegion2[threadIdx.x + 32 + 64 * i];
                }
            }
        }
    }
}

// Dense x Dense Kernel Launcher
void A_B_DenseXDense(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags, void *streamPtr)
{
    dim3 block(64, 1, 1);
    dim3 grid((numElements + 1 - 1) / 1, 1, 1);
    cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
    kernel_A_B_DenseXDense<<<grid, block, 0, stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
    CHECK_ERR;
}

// Dense x Sparse Kernel Launcher
void A_B_full_DenseXDense2(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags, void *streamPtr)
{
    dim3 block(32, 1, 1);
    dim3 grid((numElements + 1 - 1) / 1, 1, 1);
    cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
    kernel_A_B_full_DenseXDense2<<<grid, block, 0, stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
    CHECK_ERR;
}

__global__ void printKernel(const float *devicePtr, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n)
    {
        printf("Value at index %d: %f\n", idx, devicePtr[idx]);
    }
}

int main()
{
    bool debug_print = true;
    float *alpha_dev = nullptr;
    float *beta_dev = nullptr;
    float *transpose_alpha_dev = nullptr;
    float *transpose_beta_dev = nullptr;
    float alpha = 1.5f;
    float beta = 0.0f;
    float talpha = 1.0f;
    float tbeta = 0.0f;
    cudaMalloc(&alpha_dev, sizeof(float));
    cudaMalloc(&beta_dev, sizeof(float));
    cudaMalloc(&transpose_alpha_dev, sizeof(float));
    cudaMalloc(&transpose_beta_dev, sizeof(float));
    cudaMemcpy((void *)alpha_dev, (void *)&alpha, sizeof(float), cudaMemcpyHostToDevice);
    CHECK_ERR;
    cudaMemcpy((void *)beta_dev, (void *)&beta, sizeof(float), cudaMemcpyHostToDevice);
    CHECK_ERR;
    cudaMemcpy((void *)transpose_alpha_dev, (void *)&talpha, sizeof(float), cudaMemcpyHostToDevice);
    CHECK_ERR;
    cudaMemcpy((void *)transpose_beta_dev, (void *)&tbeta, sizeof(float), cudaMemcpyHostToDevice);
    CHECK_ERR;

    std::cout.precision(10);
    std::cout << std::setw(10);
    cublasHandle_t handle;
    cublasStatus_t createStatus = cublasCreate(&handle);
    if (createStatus != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("UWU");
    }
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cusparseHandle_t cuSparseHandle;
    cusparseCreate(&cuSparseHandle);

    int num_els = 325878;
    if (325878 > std::numeric_limits<int>::max())
    {
        throw std::runtime_error("Batch size too huge for num_els");
    }
    constexpr int cuSparseBatchSize = 65500;
    constexpr int cudaStreamsNeeded = 5;
    cudaStream_t streams[cudaStreamsNeeded];
    for (int i = 0; i < cudaStreamsNeeded; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    std::cout << "Gemm-Type: "
              << "A_B_full_DenseXDense2" << std::endl;
    std::cout << "Number of elements: " << num_els << std::endl;
    std::cout << "Dense FLOP/s per element from gemmforge: "
              << "8064" << std::endl;
    // Element Matrices
    std::cout << "Instantiating core matrices" << std::endl;

    float CoreA[56 * 9] = {0., 2., 4., 6., 8., 10., 12., 14., 16., 18.,
                           20., 22., 24., 26., 28., 30., 32., 34., 36., 38.,
                           40., 42., 44., 46., 48., 50., 52., 54., 56., 58.,
                           60., 62., 64., 66., 68., 70., 72., 74., 76., 78.,
                           80., 82., 84., 86., 88., 90., 92., 94., 96., 98.,
                           100., 102., 104., 106., 108., 110., 112., 114., 116., 118.,
                           120., 122., 124., 126., 128., 130., 132., 134., 136., 138.,
                           140., 142., 144., 146., 148., 150., 152., 154., 156., 158.,
                           160., 162., 164., 166., 168., 170., 172., 174., 176., 178.,
                           180., 182., 184., 186., 188., 190., 192., 194., 196., 198.,
                           200., 202., 204., 206., 208., 210., 212., 214., 216., 218.,
                           220., 222., 224., 226., 228., 230., 232., 234., 236., 238.,
                           240., 242., 244., 246., 248., 250., 252., 254., 256., 258.,
                           260., 262., 264., 266., 268., 270., 272., 274., 276., 278.,
                           280., 282., 284., 286., 288., 290., 292., 294., 296., 298.,
                           300., 302., 304., 306., 308., 310., 312., 314., 316., 318.,
                           320., 322., 324., 326., 328., 330., 332., 334., 336., 338.,
                           340., 342., 344., 346., 348., 350., 352., 354., 356., 358.,
                           360., 362., 364., 366., 368., 370., 372., 374., 376., 378.,
                           380., 382., 384., 386., 388., 390., 392., 394., 396., 398.,
                           400., 402., 404., 406., 408., 410., 412., 414., 416., 418.,
                           420., 422., 424., 426., 428., 430., 432., 434., 436., 438.,
                           440., 442., 444., 446., 448., 450., 452., 454., 456., 458.,
                           460., 462., 464., 466., 468., 470., 472., 474., 476., 478.,
                           480., 482., 484., 486., 488., 490., 492., 494., 496., 498.,
                           500., 502., 504., 506., 508., 510., 512., 514., 516., 518.,
                           520., 522., 524., 526., 528., 530., 532., 534., 536., 538.,
                           540., 542., 544., 546., 548., 550., 552., 554., 556., 558.,
                           560., 562., 564., 566., 568., 570., 572., 574., 576., 578.,
                           580., 582., 584., 586., 588., 590., 592., 594., 596., 598.,
                           600., 602., 604., 606., 608., 610., 612., 614., 616., 618.,
                           620., 622., 624., 626., 628., 630., 632., 634., 636., 638.,
                           640., 642., 644., 646., 648., 650., 652., 654., 656., 658.,
                           660., 662., 664., 666., 668., 670., 672., 674., 676., 678.,
                           680., 682., 684., 686., 688., 690., 692., 694., 696., 698.,
                           700., 702., 704., 706., 708., 710., 712., 714., 716., 718.,
                           720., 722., 724., 726., 728., 730., 732., 734., 736., 738.,
                           740., 742., 744., 746., 748., 750., 752., 754., 756., 758.,
                           760., 762., 764., 766., 768., 770., 772., 774., 776., 778.,
                           780., 782., 784., 786., 788., 790., 792., 794., 796., 798.,
                           800., 802., 804., 806., 808., 810., 812., 814., 816., 818.,
                           820., 822., 824., 826., 828., 830., 832., 834., 836., 838.,
                           840., 842., 844., 846., 848., 850., 852., 854., 856., 858.,
                           860., 862., 864., 866., 868., 870., 872., 874., 876., 878.,
                           880., 882., 884., 886., 888., 890., 892., 894., 896., 898.,
                           900., 902., 904., 906., 908., 910., 912., 914., 916., 918.,
                           920., 922., 924., 926., 928., 930., 932., 934., 936., 938.,
                           940., 942., 944., 946., 948., 950., 952., 954., 956., 958.,
                           960., 962., 964., 966., 968., 970., 972., 974., 976., 978.,
                           980., 982., 984., 986., 988., 990., 992., 994., 996., 998.,
                           1000., 1002., 1004., 1006.};
    float CoreB_sparse[81] = {8., 7., 3., 9., 2., 3., 5., 9., 4., 3., 5., 4., 7., 9., 8., 5., 7., 8.,
                              6., 5., 7., 7., 1., 1., 2., 1., 7., 4., 7., 1., 4., 2., 2., 3., 7., 4.,
                              7., 1., 6., 2., 8., 9., 6., 1., 4., 3., 1., 3., 6., 8., 4., 4., 5., 2.,
                              4., 1., 2., 6., 2., 2., 8., 7., 2., 7., 7., 7., 1., 5., 8., 1., 8., 7.,
                              6., 2., 5., 7., 7., 9., 9., 8., 5.};
    float CoreB_dense[9 * 9] = {8., 7., 3., 9., 2., 3., 5., 9., 4., 3., 5., 4., 7., 9., 8., 5., 7., 8.,
                                6., 5., 7., 7., 1., 1., 2., 1., 7., 4., 7., 1., 4., 2., 2., 3., 7., 4.,
                                7., 1., 6., 2., 8., 9., 6., 1., 4., 3., 1., 3., 6., 8., 4., 4., 5., 2.,
                                4., 1., 2., 6., 2., 2., 8., 7., 2., 7., 7., 7., 1., 5., 8., 1., 8., 7.,
                                6., 2., 5., 7., 7., 9., 9., 8., 5.};
    float CoreC[56 * 9] = {0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1,
                           1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3,
                           2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5,
                           3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
                           4.8, 4.9, 5., 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9,
                           6., 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7., 7.1,
                           7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8., 8.1, 8.2, 8.3,
                           8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9., 9.1, 9.2, 9.3, 9.4, 9.5,
                           9.6, 9.7, 9.8, 9.9, 10., 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7,
                           10.8, 10.9, 11., 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9,
                           12., 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13., 13.1,
                           13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14., 14.1, 14.2, 14.3,
                           14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15., 15.1, 15.2, 15.3, 15.4, 15.5,
                           15.6, 15.7, 15.8, 15.9, 16., 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7,
                           16.8, 16.9, 17., 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9,
                           18., 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 19., 19.1,
                           19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20., 20.1, 20.2, 20.3,
                           20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21., 21.1, 21.2, 21.3, 21.4, 21.5,
                           21.6, 21.7, 21.8, 21.9, 22., 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7,
                           22.8, 22.9, 23., 23.1, 23.2, 23.3, 23.4, 23.5, 23.6, 23.7, 23.8, 23.9,
                           24., 24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7, 24.8, 24.9, 25., 25.1,
                           25.2, 25.3, 25.4, 25.5, 25.6, 25.7, 25.8, 25.9, 26., 26.1, 26.2, 26.3,
                           26.4, 26.5, 26.6, 26.7, 26.8, 26.9, 27., 27.1, 27.2, 27.3, 27.4, 27.5,
                           27.6, 27.7, 27.8, 27.9, 28., 28.1, 28.2, 28.3, 28.4, 28.5, 28.6, 28.7,
                           28.8, 28.9, 29., 29.1, 29.2, 29.3, 29.4, 29.5, 29.6, 29.7, 29.8, 29.9,
                           30., 30.1, 30.2, 30.3, 30.4, 30.5, 30.6, 30.7, 30.8, 30.9, 31., 31.1,
                           31.2, 31.3, 31.4, 31.5, 31.6, 31.7, 31.8, 31.9, 32., 32.1, 32.2, 32.3,
                           32.4, 32.5, 32.6, 32.7, 32.8, 32.9, 33., 33.1, 33.2, 33.3, 33.4, 33.5,
                           33.6, 33.7, 33.8, 33.9, 34., 34.1, 34.2, 34.3, 34.4, 34.5, 34.6, 34.7,
                           34.8, 34.9, 35., 35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 35.9,
                           36., 36.1, 36.2, 36.3, 36.4, 36.5, 36.6, 36.7, 36.8, 36.9, 37., 37.1,
                           37.2, 37.3, 37.4, 37.5, 37.6, 37.7, 37.8, 37.9, 38., 38.1, 38.2, 38.3,
                           38.4, 38.5, 38.6, 38.7, 38.8, 38.9, 39., 39.1, 39.2, 39.3, 39.4, 39.5,
                           39.6, 39.7, 39.8, 39.9, 40., 40.1, 40.2, 40.3, 40.4, 40.5, 40.6, 40.7,
                           40.8, 40.9, 41., 41.1, 41.2, 41.3, 41.4, 41.5, 41.6, 41.7, 41.8, 41.9,
                           42., 42.1, 42.2, 42.3, 42.4, 42.5, 42.6, 42.7, 42.8, 42.9, 43., 43.1,
                           43.2, 43.3, 43.4, 43.5, 43.6, 43.7, 43.8, 43.9, 44., 44.1, 44.2, 44.3,
                           44.4, 44.5, 44.6, 44.7, 44.8, 44.9, 45., 45.1, 45.2, 45.3, 45.4, 45.5,
                           45.6, 45.7, 45.8, 45.9, 46., 46.1, 46.2, 46.3, 46.4, 46.5, 46.6, 46.7,
                           46.8, 46.9, 47., 47.1, 47.2, 47.3, 47.4, 47.5, 47.6, 47.7, 47.8, 47.9,
                           48., 48.1, 48.2, 48.3, 48.4, 48.5, 48.6, 48.7, 48.8, 48.9, 49., 49.1,
                           49.2, 49.3, 49.4, 49.5, 49.6, 49.7, 49.8, 49.9, 50., 50.1, 50.2, 50.3};

    float CoreB_data[81] = {8.0, 3.0, 6.0, 4.0, 7.0, 3.0, 4.0, 7.0, 6.0, 7.0, 5.0, 5.0, 7.0, 1.0, 1.0, 1.0, 7.0, 2.0, 3.0, 4.0, 7.0, 1.0, 6.0, 3.0, 2.0, 7.0, 5.0, 9.0, 7.0, 7.0, 4.0, 2.0, 6.0, 6.0, 1.0, 7.0, 2.0, 9.0, 1.0, 2.0, 8.0, 8.0, 2.0, 5.0, 7.0, 3.0, 8.0, 1.0, 2.0, 9.0, 4.0, 2.0, 8.0, 9.0, 5.0, 5.0, 2.0, 3.0, 6.0, 4.0, 8.0, 1.0, 9.0, 9.0, 7.0, 1.0, 7.0, 1.0, 5.0, 7.0, 8.0, 8.0, 4.0, 8.0, 7.0, 4.0, 4.0, 2.0, 2.0, 7.0, 5.0};
    int CoreB_indices[81] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    int CoreB_indptr[10] = {0, 9, 18, 27, 36, 45, 54, 63, 72, 81};

    // Buffers
    std::cout << "Instantiating buffer matrices" << std::endl;
    float *A = new float[56 * 9 * num_els];
    float *B_dense = new float[9 * 9 * num_els];
    float *B_sparse = new float[81 * num_els];
    float *C = new float[56 * 9 * num_els];
    float *R1 = new float[56 * 9 * num_els];
    float *R2 = new float[56 * 9 * num_els];

    // Copy the Element Matrices N times into Element Buffers
    std::cout << "Copying core matrices to buffers" << std::endl;
    for (int i = 0; i < num_els; i++)
    {
        std::memcpy(&A[56 * 9 * i], &CoreA[0], 56 * 9 * sizeof(float));
        std::memcpy(&B_dense[9 * 9 * i], &CoreB_dense[0], 9 * 9 * sizeof(float));
        std::memcpy(&B_sparse[81 * i], &CoreB_sparse[0], 81 * sizeof(float));
        std::memcpy(&C[56 * 9 * i], &CoreC[0], 56 * 9 * sizeof(float));
    }

    float *A_dev = nullptr;
    float *B_sparse_dev = nullptr;
    float *B_dense_dev = nullptr;
    float *C1_dev = nullptr;
    float *C2_dev = nullptr;

    std::cout << "Allocating device memory" << std::endl;
    cudaMalloc((void **)&A_dev, sizeof(float) * 56 * 9 * num_els);
    CHECK_ERR;
    cudaMalloc((void **)&B_sparse_dev, sizeof(float) * 81 * num_els);
    CHECK_ERR;
    cudaMalloc((void **)&B_dense_dev, sizeof(float) * 9 * 9 * num_els);
    CHECK_ERR;
    cudaMalloc((void **)&C1_dev, sizeof(float) * 56 * 9 * num_els);
    CHECK_ERR;
    cudaMalloc((void **)&C2_dev, sizeof(float) * 56 * 9 * num_els);
    CHECK_ERR;

    std::cout << "Copying buffers to device" << std::endl;
    cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 56 * 9 * num_els, cudaMemcpyHostToDevice);
    CHECK_ERR;
    cudaMemcpy((void *)B_sparse_dev, (void *)B_sparse, sizeof(float) * 81 * num_els, cudaMemcpyHostToDevice);
    CHECK_ERR;
    cudaMemcpy((void *)B_dense_dev, (void *)B_dense, sizeof(float) * 9 * 9 * num_els, cudaMemcpyHostToDevice);
    CHECK_ERR;
    cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * 56 * 9 * num_els, cudaMemcpyHostToDevice);
    CHECK_ERR;
    cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * 56 * 9 * num_els, cudaMemcpyHostToDevice);
    CHECK_ERR;

    // Dense x Dense Matrix Mult
    A_B_DenseXDense(A_dev, 0, B_dense_dev, 0, C1_dev, 0, num_els, nullptr, nullptr);
    CHECK_ERR;
    cudaDeviceSynchronize();
    CHECK_ERR;
    cudaMemcpy(C1_dev, C, sizeof(float) * 56 * 9 * num_els, cudaMemcpyHostToDevice);
    CHECK_ERR;

    std::cout << "Calling Dense x Dense kernel" << std::endl;
    float elapsedTime1 = 0.0;
    cudaEvent_t startDD, stopDD;
    cudaEventCreate(&startDD);
    CHECK_ERR;
    cudaEventCreate(&stopDD);
    CHECK_ERR;
    cudaEventRecord(startDD);
    CHECK_ERR;
    // callFirstGemm(DeviceA, 0, DeviceB, 0, DeviceTmp, 0, NumElements, nullptr, FirstDriver.getTestStream());
    A_B_DenseXDense(A_dev, 0, B_dense_dev, 0, C1_dev, 0, num_els, nullptr, nullptr);
    CHECK_ERR;
    cudaEventRecord(stopDD);
    CHECK_ERR;
    cudaEventSynchronize(stopDD);
    CHECK_ERR;
    cudaDeviceSynchronize();
    CHECK_ERR;
    cudaEventElapsedTime(&elapsedTime1, startDD, stopDD);
    CHECK_ERR;
    std::cout << "Dense x Dense kernel took " << elapsedTime1 << " ms" << std::endl;
    cudaEventDestroy(startDD);
    CHECK_ERR;
    cudaEventDestroy(stopDD);
    CHECK_ERR;
    cudaMemcpy(R1, C1_dev, sizeof(float) * 56 * 9 * num_els, cudaMemcpyDeviceToHost);
    CHECK_ERR;

    // Dense x Sparse Matrix Mult
    A_B_full_DenseXDense2(A_dev, 0, B_sparse_dev, 0, C2_dev, 0, num_els, nullptr, nullptr);
    CHECK_ERR;
    cudaDeviceSynchronize();
    CHECK_ERR;
    cudaMemcpy(C2_dev, C, sizeof(float) * 56 * 9 * num_els, cudaMemcpyHostToDevice);
    CHECK_ERR;

    std::cout << "Calling Dense x Sparse kernel" << std::endl;
    float elapsedTime2 = 0.0;
    cudaEvent_t startDS, stopDS;
    cudaEventCreate(&startDS);
    CHECK_ERR;
    cudaEventCreate(&stopDS);
    CHECK_ERR;
    cudaEventRecord(startDS);
    CHECK_ERR;
    A_B_full_DenseXDense2(A_dev, 0, B_sparse_dev, 0, C2_dev, 0, num_els, nullptr, nullptr);
    CHECK_ERR;
    cudaEventRecord(stopDS);
    CHECK_ERR;
    cudaEventSynchronize(stopDS);
    CHECK_ERR;
    cudaDeviceSynchronize();
    CHECK_ERR;
    cudaEventElapsedTime(&elapsedTime2, startDS, stopDS);
    CHECK_ERR;
    std::cout << "Dense x Sparse kernel took " << elapsedTime2 << " ms" << std::endl;
    cudaEventDestroy(startDS);
    CHECK_ERR;
    cudaEventDestroy(stopDS);
    CHECK_ERR;
    cudaMemcpy(R2, C2_dev, sizeof(float) * 56 * 9 * num_els, cudaMemcpyDeviceToHost);
    CHECK_ERR;

    for (int i = 0; i < 56 * 9 * num_els; i++)
    {
        if (std::abs(R1[i] - R2[i]) >= 0.1)
        {
            std::string s = " Dense x  Dense and  Dense x  Sparse Matrix Mismatch in Multiplication at " + std::to_string(i) + "!";
            // throw std::runtime_error(s);
            std::cout << " " << std::to_string(R1[i]) + " and " + std::to_string(R2[i]) << std::endl;
            std::cout << "Dense x Dense Gemmforge and Dense x Sparse Gemmforge results don't match!" << std::endl;
            std::cout << s << std::endl;
            break;
        }
    }
    cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * 56 * 9 * num_els, cudaMemcpyHostToDevice);
    CHECK_ERR;

    std::cout << "Freeing device memory" << std::endl;

    cudaFree(B_dense_dev);
    cudaFree(C1_dev);
    cudaFree(A_dev);

    delete[] A;
    delete[] B_dense;
    delete[] R2;
    delete[] R1;
    delete[] C;
    cudaFree(B_sparse_dev);
    CHECK_ERR;

    cudaDeviceReset();
}
