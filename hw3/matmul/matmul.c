#define _GNU_SOURCE
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <emmintrin.h>

// Naive CPU matrix multiplication
void matmul_singlethread(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K; k++) { c += A[i * K + k] * B[k * N + j]; }
      C[i * N + j] = c;
    }
  }
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
    int num_threads) {

  size_t i, ii, j, jj, k, kk;

#pragma omp parallel for num_threads(num_threads) private(i, ii, j, jj, k, kk) schedule(dynamic)
  for (i = 0; i < M; i += 16) {
    for (k = 0; k < K; k += 16) {
      for (j = 0; j < N-15; j += 16) {
        for (ii = i; ii < i+16 && ii < M; ii++) {
          __m512 cv = _mm512_loadu_ps(&C[ii*N + j]);
          for (kk = k; kk < k+16 && kk < K; kk++) {
            __m512 av = _mm512_set1_ps(A[ii*K + kk]);
            __m512 bv = _mm512_loadu_ps(&B[kk*N + j]);
            cv = _mm512_fmadd_ps(av, bv, cv);
          }
          _mm512_storeu_ps(&C[ii*N + j], cv);
        }
      }
    }
  }

  if (N % 16 == 0) {
    return;
  }

#pragma omp parallel for num_threads(num_threads) private(i, ii, j, jj, k, kk) schedule(dynamic)
  for (i = 0; i < M; i += 16) {
    for (k = 0; k < K; k += 16) {
      for (j = N - N%16; j < N; j += 16) {
        for (ii = i; ii < i+16 && ii < M; ii++) {
          for (kk = k; kk < k+16 && kk < K; kk++) {
            float aa = A[ii*K + kk];
            for (jj = j; jj < N; jj++) {
              C[ii*N + jj] += aa * B[kk*N + jj];
            }
          }
        }
      }
    }
  }
}
