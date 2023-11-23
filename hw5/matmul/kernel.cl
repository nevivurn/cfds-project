#define T 128
#define W 8

#define A(row, col) A[(row)*K + (col)]
#define B(row, col) B[(row)*N + (col)]
#define C(row, col) C[(row)*N + (col)]

__kernel void sgemm(const __global float *A, const __global float *B, __global float *C, const int M, const int N, const int K) {
  // 0..DIM/8 -> 0..DIM
  const int g_row = get_group_id(0) * T;
  const int g_col = get_group_id(1) * T;

  // 0..16 -> 0..128 (TILE)
  const int l_row = get_local_id(0);
  const int l_col = get_local_id(1);

  // tile offsets
  const int t_row = l_row/2;
  const int t_col = (l_col*W) + (l_row&1) * W/2;

  // local tiles, 8 x 128
  __local float atile[W][T];
  __local float btile[W][T];

  float8 acc[W];
  float8 arow, brow;

  // init accumulator
  for (int i = 0; i < W; i++)
    acc[i] = (float8) 0.0f;

  for (int t = 0; t < K/W; t++) {
    for (int h = 0; h < W/2; h++)
      atile[t_row][t_col+h] = A(g_row + t_col+h, t*W + t_row);
    for (int h = 0; h < W/2; h++)
      btile[t_row][t_col+h] = B(t*W + t_row, g_col + t_col+h);
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // compute
    for (int k = 0; k < W; k++) {
      arow = vload8(l_row, atile[k]);
      brow = vload8(l_col, btile[k]);
      for (int i = 0; i < W; i++)
        acc[i] = fma(arow[i], brow, acc[i]);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int i = 0; i < W; i++)
    for (int w = 0; w < W; w++)
      C(g_row + l_row*W+i, g_col + l_col*W+w) = acc[i][w];
}

// naive sgemm
__kernel void sgemm_slow(const __global float *A, const __global float *B, __global float *C, const int M, const int N, const int K) {
  const int g_row = get_global_id(0);
  const int g_col = get_global_id(1);

  float acc = 0.0f;
  for (int k = 0; k < K; k++)
    acc += A[g_row*K + k] * B[k*N + g_col];
  C[g_row*N + g_col] = acc;
}
