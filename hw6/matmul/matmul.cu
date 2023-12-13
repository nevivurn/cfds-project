#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

#define T 128
#define W 8
#define BS (T/W)

#define A(row, col) A[(row)*K + (col)]
#define B(row, col) B[(row)*N + (col)]
#define C(row, col) C[(row)*N + (col)]

static int mpi_rank, mpi_world_size;

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  // 0..DIM
  const int g_row = blockIdx.y * T;
  const int g_col = blockIdx.x * T;

  // 0..16 = BS
  const int l_row = threadIdx.y;
  const int l_col = threadIdx.x;

  // shared tiles, 8x128x2x2 = 16KiB
  __shared__ float atile[2][W][T];
  __shared__ float btile[2][W][T];
  
  // registers, ~8x8 = 64
  float acc[W][W];
  //float arow[W], brow[W];
  float arow[W], brow[W];

  // init accumulator
  for (int i = 0; i < W; i++)
    for (int j = 0; j < W; j++)
      acc[i][j] = 0.0f;

  for (int h = 0; h < W/2; h++)
    atile[0][(l_col&7)][h*32 + 2*l_row + (l_col>>3)] = A(g_row + h*32 + 2*l_row + (l_col>>3), (l_col&7));
  for (int h = 0; h < W/2; h++)
    btile[0][2*h + (l_row>>3)][(l_row&7)*BS + l_col] = B(2*h + (l_row>>3), g_col + (l_row&7)*BS + l_col);
  __syncthreads();

  // main loop
  for (int t = 0; t < K/W; t++) {
    if (t+1 < K/W)
      for (int h = 0; h < W/2; h++)
        atile[~t&1][(l_col&7)][h*32 + 2*l_row + (l_col>>3)] = A(g_row + h*32 + 2*l_row + (l_col>>3), (t+1)*W + (l_col&7));
      for (int h = 0; h < W/2; h++)
        btile[~t&1][2*h + (l_row>>3)][(l_row&7)*BS + l_col] = B((t+1)*W + 2*h + (l_row>>3), g_col + (l_row&7)*BS + l_col);

    // compute 8x8
    for (int k = 0; k < W; k++) {

      //for (int w = 0; w < W; w++) {
      //  arow[w] = atile[t&1][k][W*l_row+w];
      //  brow[w] = btile[t&1][k][W*l_col+w];
      //}

      reinterpret_cast<float4 *>(arow)[0] = reinterpret_cast<float4 *>(&atile[t&1][k][W*l_row])[0];
      reinterpret_cast<float4 *>(arow)[1] = reinterpret_cast<float4 *>(&atile[t&1][k][W*l_row])[1];
      reinterpret_cast<float4 *>(brow)[0] = reinterpret_cast<float4 *>(&btile[t&1][k][W*l_col])[0];
      reinterpret_cast<float4 *>(brow)[1] = reinterpret_cast<float4 *>(&btile[t&1][k][W*l_col])[1];

      for (int i = 0; i < W; i++)
        for (int j = 0; j < W; j++)
          //acc[i][j] += arow[i] * brow[j];
          acc[i][j] = fmaf(arow[i], brow[j], acc[i][j]);
    }
    __syncthreads();
  }

  for (int i = 0; i < W; i++)
    for (int j = 0; j < W; j++)
      C(g_row + l_row*W+i, g_col + l_col*W+j) = acc[i][j];
}

#define NGPU 4

static int Mbegin[4*NGPU], Mend[4*NGPU];
static int ngpu;
static cudaStream_t streams[4*NGPU];
static float *A_gpu[4*NGPU], *B_gpu[4*NGPU], *C_gpu[4*NGPU];


void matmul(float *A, float *B, float *C, int M, int N, int K) {

  if (mpi_rank != 0) return; // FIXME

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int part = 0; part < 4; part ++) {
      // Async memcpy H->D on each GPU
      CHECK_CUDA(cudaMemcpyAsync(A_gpu[4*i+part], &A[Mbegin[4*i+part] * K],
            (Mend[4*i+part] - Mbegin[4*i+part]) * K * sizeof(float),
            cudaMemcpyHostToDevice, streams[4*i+part]));
      CHECK_CUDA(cudaMemcpyAsync(B_gpu[4*i+part], B, K * N * sizeof(float),
			      cudaMemcpyHostToDevice, streams[4*i]));

      dim3 blockDim(BS, BS);
      dim3 gridDim((N + T - 1) / T, (Mend[4*i+part] - Mbegin[4*i+part] + T - 1) / T);
      matmul_kernel<<<gridDim, blockDim, 0, streams[4*i+part]>>>(
          A_gpu[4*i+part], B_gpu[4*i+part], C_gpu[4*i+part], Mend[4*i+part] - Mbegin[4*i+part], N, K);
      CHECK_CUDA(cudaGetLastError());

      CHECK_CUDA(cudaMemcpyAsync(&C[Mbegin[4*i+part] * N], C_gpu[4*i+part],
            (Mend[4*i+part] - Mbegin[4*i+part]) * N * sizeof(float),
            cudaMemcpyDeviceToHost, streams[4*i+part]));
    }
  }

  // Wait for all async jobs to finish
  for (int i = 0; i < ngpu; i++) {
    cudaSetDevice(i);
    for (int part = 0; part < 4; part ++) {
      cudaStreamSynchronize(streams[4*i+part]);
    }
  }
}


void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  CHECK_CUDA(cudaGetDeviceCount(&ngpu));

  printf("[rank %d] Number of devices: %d\n", mpi_rank, ngpu);
  cudaDeviceProp props[4];
  for (int i = 0; i < ngpu; ++i) {
    CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));
    printf("[rank %d] device %d: %s\n", mpi_rank, i, props[i].name);
  }

  for (int i = 0; i < 4*ngpu; i++) {
    Mbegin[i] = M / (4*ngpu) * i;
    Mend[i] = M / (4*ngpu) * (i + 1);
    if (i == (4*ngpu) - 1) Mend[i] = M;
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int part = 0; part < 4; part++)
      CHECK_CUDA(cudaStreamCreate(&streams[4*i+part]));
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int part = 0; part < 4; part++) {
      CHECK_CUDA(
          cudaMalloc(&A_gpu[4*i+part], (Mend[4*i+part] - Mbegin[4*i+part]) * K * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&B_gpu[4*i+part], K * N * sizeof(float)));
      CHECK_CUDA(
          cudaMalloc(&C_gpu[4*i+part], (Mend[4*i+part] - Mbegin[4*i+part]) * N * sizeof(float)));
    }
  }
}


void matmul_finalize() {
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int part = 0; part < 4; part++) {
      CHECK_CUDA(cudaFree(A_gpu[4*i+part]));
      CHECK_CUDA(cudaFree(B_gpu[4*i+part]));
      CHECK_CUDA(cudaFree(C_gpu[4*i+part]));
      CHECK_CUDA(cudaStreamDestroy(streams[4*i+part]));
    }
  }
}
