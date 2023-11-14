#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(float *A, float *B, float *C, int M, int N, int K,
    int threads_per_process, int mpi_rank, int mpi_world_size) {
  // Distribute rows
  int rowcnts[mpi_world_size];
  int sendcnts[mpi_world_size], displs[mpi_world_size];
  for (int i = 0; i < mpi_world_size; i++)
    rowcnts[i] = M/mpi_world_size + (i < M % (M/mpi_world_size));

  // B must be broadcast to everyone
  MPI_Bcast(B, N*K, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Scatter relevant parts of A
  for (int i = 0, displ = 0; i < mpi_world_size; i++) {
    sendcnts[i] = rowcnts[i] * N;
    displs[i] = displ;
    displ += sendcnts[i];
  }
  MPI_Scatterv(
      A, sendcnts, displs, MPI_FLOAT,
      A, sendcnts[mpi_rank], MPI_FLOAT,
      0, MPI_COMM_WORLD);

  size_t i, ii, j, k, kk;
#pragma omp parallel for num_threads(threads_per_process)  schedule(dynamic)
  for (i = 0; i < rowcnts[mpi_rank]; i += 16) {
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

  // Gather results
  for (int i = 0, displ = 0; i < mpi_world_size; i++) {
    sendcnts[i] = rowcnts[i] * K;
    displs[i] = displ;
    displ += sendcnts[i];
  }

  MPI_Barrier(MPI_COMM_WORLD); // ?
  MPI_Gatherv(
      C, sendcnts[mpi_rank], MPI_FLOAT,
      C, sendcnts, displs, MPI_FLOAT,
      0, MPI_COMM_WORLD);
}
