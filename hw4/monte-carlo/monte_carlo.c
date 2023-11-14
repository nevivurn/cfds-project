#include <mpi.h>
#include <stdio.h>

#include "monte_carlo.h"
#include "util.h"

double monte_carlo(double *xs, double *ys, int num_points, int mpi_rank, int mpi_world_size, int threads_per_process) {
  int share = num_points / mpi_world_size;
  int lo = mpi_rank*share;
  int hi = mpi_rank+1 == mpi_world_size ? num_points : lo+share;
  int count = 0;

  int sendcounts[mpi_world_size], displs[mpi_world_size];
  for (int i = 0; i < mpi_world_size; i++) {
    sendcounts[i] = i+1 != mpi_world_size ? share : share + num_points%share;
    displs[i] = i*share;
  }

  MPI_Bcast(xs, num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(ys, num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#pragma omp parallel for reduction(+:count) num_threads(threads_per_process)
  for (int i = lo; i < hi; i++) {
    double x = xs[i];
    double y = ys[i];
    if (x*x + y*y <= 1)
      count++;
  }

  int allcnt;
  MPI_Reduce(&count, &allcnt, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (mpi_rank != 0) {
    return 0;
  }

  return (double) 4 * allcnt / num_points;
}
