#define T 32
#define W 8

__kernel void sgemm(const __global float *A, const __global float *B, __global float *C, const int M, const int N, const int K) {
  // 0..DIM/8 -> 0..DIM
  const int g_row = get_group_id(0) * T;
  const int g_col = get_group_id(1) * T;

  // 0..4 -> 0..32 (TILE)
  const int l_row = get_local_id(0);
  const int l_col = get_local_id(1);

  // local tiles, 32 x 32/8
  __local float8 atile[T][T/W];
  __local float8 btile[T][T/W];

  float8 acc[W];
  float8 arow, brow;

  // init accumulator
  for (size_t i = 0; i < W; i++)
    for (size_t j = 0; j < W; j++)
      acc[i][j] = 0.0f;

  for (size_t t = 0; t < K/T; t++) {
    // copy b
    for (size_t i = 0; i < W; i++)
      btile[l_row*W+i][l_col] = vload8(g_col/W+l_col, B + (t*T + l_row*W+i)*N);
    // transpose a
    for (size_t i = 0; i < W; i++)
      for (size_t w = 0; w < W; w++)
        atile[l_row*W+i][l_col][w] = A[(g_row+l_col*W+w)*K + t*T + l_row*W + i];

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t k = 0; k < T; k++) {
      // load rows
      arow = atile[k][l_row];
      brow = btile[k][l_col];

      for (size_t i = 0; i < W; i++)
        acc[i] = fma(arow[i], brow, acc[i]);
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int i = 0; i < W; i++)
    vstore8(acc[i], g_col/W+l_col, C + (g_row+l_row*W + i)*N);
}

// naive sgemm for dim % 32 != 0
__kernel void sgemm_slow(const __global float *A, const __global float *B, __global float *C, const int M, const int N, const int K) {
  const int g_row = get_global_id(0);
  const int g_col = get_global_id(1);

  float acc = 0.0f;
  for (size_t k = 0; k < K; k++)
    acc += A[g_row*K + k] * B[k*N + g_col];
  C[g_row*N + g_col] = acc;
}
