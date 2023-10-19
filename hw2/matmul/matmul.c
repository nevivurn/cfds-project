#define _GNU_SOURCE
#include "util.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>
#include <emmintrin.h>

#define MAX_THREADS 256

struct thread_arg {
  const float *A;
  const float *B;
  float *C;
  int M;
  int N;
  int K;
  int num_threads;
  int rank; /* id of this thread */
} args[MAX_THREADS];
static pthread_t threads[MAX_THREADS];


// Naive CPU matrix multiplication
void matmul_singlethread(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K; k++) { c += A[i * K + k] * B[k * N + j]; }
      C[i * N + j] = c;
    }
  }
}


static void *matmul_thread(void *arg) {
  struct thread_arg *input = (struct thread_arg *)arg;

  const float *restrict A = (*input).A;
  const float *restrict B = (*input).B;
  float *restrict C = (*input).C;
  int M = (*input).M;
  int N = (*input).N;
  int K = (*input).K;
  int num_threads = (*input).num_threads;
  int rank = (*input).rank;

  size_t start = rank * M / num_threads;
  size_t end = (rank+1) * M / num_threads;

  size_t i, ii, j, k, kk;
  for (i = start; i < end; i += 16) {
    for (k = 0; k < K; k += 16) {
      for (j = 0; j < N-15; j += 16) {
        for (ii = i; ii < i+16 && ii < end; ii++) {
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

  if (N % 16 != 0) { // j remainder
    for (i = start; i < end; i += 16) {
      for (k = 0; k < K; k += 16) {
        for (j = N - N%16; j < N; j += 16) {
          for (ii = i; ii < i+16 && ii < end; ii++) {
            for (kk = k; kk < k+16 && kk < K; kk++) {
              float aa = A[ii*K + kk];
              for (size_t jj = j; jj < N; jj++) {
                C[ii*N + jj] += aa * B[kk*N + jj];
              }
            }
          }
        }
      }
    }
  }

  // Handle remainders

  return NULL;
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {
  if (num_threads > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }

  // Spawn num_thread CPU threads
  int err;
  for (int t = 0; t < num_threads; ++t) {
    args[t].A = A, args[t].B = B, args[t].C = C, args[t].M = M, args[t].N = N,
    args[t].K = K, args[t].num_threads = num_threads, args[t].rank = t;
    err = pthread_create(&threads[t], NULL, matmul_thread, (void *)&args[t]);
    if (err) {
      fprintf(stderr, "pthread_create(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }


  // Wait for spawned threads to terminate
  for (int t = 0; t < num_threads; ++t) {
    err = pthread_join(threads[t], NULL);
    if (err) {
      //free(BT);
      fprintf(stderr, "pthread_join(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }
  //free(BT);
}
