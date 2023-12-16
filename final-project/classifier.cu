#include <math.h>
#include <mpi.h>
#include <cassert>

#include "classifier.h"
#include "util.h"

#define BATCH 512
#define NGPU 4

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static int mpi_rank, mpi_world_size;
static cudaStream_t streams[NGPU];

// Multi-dimensional matrix containing fp32 elements
struct Tensor {
  Tensor(std::vector<int> shape_);
  Tensor(std::vector<int> shape_, float *buf_);
  ~Tensor();
  __host__ __device__ int num_elem() const;
  void fill_zeros();

  void allocate_gpu(int gpu);
  void copy_to_gpu(int gpu);
  void copy_to_cpu(int gpu);

  float *buf = nullptr;
  float *gbuf[NGPU] = { nullptr };
  int ndim = 0;
  int shape[4];
  int datashape[4];
};

Tensor::Tensor(std::vector<int> shape_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = datashape[i] = shape_[i]; }
  int N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
}

Tensor::Tensor(std::vector<int> shape_, float *buf_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = datashape[i] = shape_[i]; }
  int N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
  for (int n = 0; n < N_; ++n) { buf[n] = buf_[n]; }
}

Tensor::~Tensor() {
  // TODO(nevi): properly free memory
  //if (buf != nullptr) free(buf);
  //if (gbuf != nullptr) cudaFree(gbuf);
}

int Tensor::num_elem() const {
  int sz = 1;
  for (int i = 0; i < ndim; ++i) { sz *= shape[i]; }
  return sz;
}

void Tensor::fill_zeros() {
  int N_ = num_elem();
  for (int n = 0; n < N_; ++n) { buf[n] = 0.0; }
}

void Tensor::allocate_gpu(int gpu) {
  CHECK_CUDA(cudaMalloc(&gbuf[gpu], num_elem() * sizeof(float)));
}

void Tensor::copy_to_gpu(int gpu) {
  if (gbuf[gpu] == nullptr)
    allocate_gpu(gpu);
  CHECK_CUDA(cudaMemcpyAsync(gbuf[gpu], buf, num_elem() * sizeof(float), cudaMemcpyHostToDevice, streams[gpu]));
}

void Tensor::copy_to_cpu(int gpu) {
  CHECK_CUDA(cudaMemcpyAsync(buf, gbuf[gpu], num_elem() * sizeof(float), cudaMemcpyDeviceToHost, streams[gpu]));
}

void print_tensor(Tensor *t, int gpu, int n = 4) {
  t->copy_to_cpu(gpu);
  CHECK_CUDA(cudaStreamSynchronize(streams[gpu]));

  for (int i = 0; i < n; ++i) {
    printf("%.4f ", t->buf[i]);
  }
  printf("\n");
}

// Parameters
Tensor *w_conv1, *w_conv2, *w_conv3, *w_conv4, *w_conv5, *w_conv6, *b_conv1,
    *b_conv2, *b_conv3, *b_conv4, *b_conv5, *b_conv6, *w_fc1, *w_fc2, *w_fc3,
    *b_fc1, *b_fc2, *b_fc3, *gamma_conv1, *beta_conv1, *gamma_conv6, *beta_conv6;

// Activations
Tensor *a_conv1, *a_layernorm1, *a_relu1, *a_pool1;
Tensor *a_conv1_sum_mid, *a_conv1_sum, *a_conv1_sum_sq; // for layernorm
Tensor *a_conv2, *a_relu2, *a_pool2;
Tensor *a_conv3, *a_relu3;
Tensor *a_conv4, *a_relu4;
Tensor *a_conv5, *a_relu5;
Tensor *a_conv6, *a_layernorm6, *a_relu6, *a_pool6;
Tensor *a_conv6_sum_mid, *a_conv6_sum, *a_conv6_sum_sq; // for layernorm
Tensor *a_collapse;
Tensor *a_linear1, *a_relu7;
Tensor *a_linear2, *a_relu8;
Tensor *a_linear3;

// Operations
void conv1d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int padding, int dilation, bool has_bias);
void relu(Tensor *input, Tensor *output);
void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride);
void collapse(Tensor *input, Tensor *output);
void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias);
void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output);

// Cuda layers
__global__ void cuda_conv1d(int gpu, Tensor input, Tensor weight, Tensor bias, Tensor output);
__global__ void cuda_relu(int gpu, Tensor input, Tensor output);
__global__ void cuda_maxpool1d(int gpu, Tensor input, Tensor output); // always 3x3
__global__ void cuda_collapse(int gpu, Tensor input, Tensor output);
__global__ void cuda_linear(int gpu, Tensor input, Tensor weight, Tensor bias, Tensor output);
__global__ void cuda_layernorm(int gpu, Tensor input, Tensor gamma, Tensor beta, Tensor output,
                               Tensor sum, Tensor sum_sq);

// Cuda operations
__global__ void cuda_reduce_sum(int gpu, Tensor input, Tensor output, int N);
__global__ void cuda_reduce_sum_sq(int gpu, Tensor input, Tensor output, int N);

// Parallelization method is totally up to you, but you should gather
// the output at rank 0
void classifier(float *input_, float *output_, int N) {
  assert(N / mpi_world_size / NGPU == BATCH);

  Tensor *input = new Tensor({ N / mpi_world_size, VOCAB_SIZE, MAX_LENGTH });
  Tensor *output = new Tensor({ N / mpi_world_size });
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatter(
    input_, input->num_elem(), MPI_FLOAT,
    input->buf, input->num_elem(), MPI_FLOAT,
    0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);

    // Select input slice
    Tensor *input_slice = new Tensor({ BATCH, VOCAB_SIZE, MAX_LENGTH },
                                     input->buf + gpu * BATCH * VOCAB_SIZE * MAX_LENGTH);
    input_slice->copy_to_gpu(gpu);
    // Ready!

    // Conv block 1 : Conv1d + LayerNorm + ReLU + MaxPool1d
    cuda_conv1d<<<dim3(1008, BATCH), 256, 0, streams[gpu]>>>(gpu, *input_slice, *w_conv1, *b_conv1, *a_conv1);
    cuda_reduce_sum<<<dim3(1008, BATCH), 256, 256 * sizeof(float), streams[gpu]>>>(gpu, *a_conv1, *a_conv1_sum_mid, 256);
    cuda_reduce_sum<<<dim3(BATCH, 1), 1024, 1024 * sizeof(float), streams[gpu]>>>(gpu, *a_conv1_sum_mid, *a_conv1_sum, 1008);
    cuda_reduce_sum_sq<<<dim3(1008, BATCH), 256, 256 * sizeof(float), streams[gpu]>>>(gpu, *a_conv1, *a_conv1_sum_mid, 256);
    cuda_reduce_sum<<<dim3(BATCH, 1), 1024, 1024 * sizeof(float), streams[gpu]>>>(gpu, *a_conv1_sum_mid, *a_conv1_sum_sq, 1008);
    cuda_layernorm<<<dim3(1008, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_conv1, *gamma_conv1, *beta_conv1, *a_layernorm1,
                                               *a_conv1_sum, *a_conv1_sum_sq);
    cuda_relu<<<BATCH*1008, 256, 0, streams[gpu]>>>(gpu, *a_layernorm1, *a_relu1);
    cuda_maxpool1d<<<dim3(336, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_relu1, *a_pool1);

    // Conv block 2 : Conv1d + ReLU + MaxPool1d
    cuda_conv1d<<<dim3(330, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_pool1, *w_conv2, *b_conv2, *a_conv2);
    cuda_relu<<<BATCH*330, 256, 0, streams[gpu]>>>(gpu, *a_conv2, *a_relu2);
    cuda_maxpool1d<<<dim3(110, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_relu2, *a_pool2);

    // Conv block 3 : Conv1d + ReLU
    cuda_conv1d<<<dim3(108, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_pool2, *w_conv3, *b_conv3, *a_conv3);
    cuda_relu<<<BATCH*108, 256, 0, streams[gpu]>>>(gpu, *a_conv3, *a_relu3);

    // Conv block 4 : Conv1d + ReLU
    cuda_conv1d<<<dim3(106, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_relu3, *w_conv4, *b_conv4, *a_conv4);
    cuda_relu<<<BATCH*106, 256, 0, streams[gpu]>>>(gpu, *a_conv4, *a_relu4);

    // Conv block 5 : Conv1d + ReLU
    cuda_conv1d<<<dim3(104, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_relu4, *w_conv5, *b_conv5, *a_conv5);
    cuda_relu<<<BATCH*104, 256, 0, streams[gpu]>>>(gpu, *a_conv5, *a_relu5);

    // Conv block 6 : Conv1d + LayerNorm + ReLU + MaxPool1d
    cuda_conv1d<<<dim3(102, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_relu5, *w_conv6, *b_conv6, *a_conv6);
    cuda_reduce_sum<<<dim3(102, BATCH), 256, 256 * sizeof(float), streams[gpu]>>>(gpu, *a_conv6, *a_conv6_sum_mid, 256);
    cuda_reduce_sum<<<dim3(BATCH, 1), 128, 128 * sizeof(float), streams[gpu]>>>(gpu, *a_conv6_sum_mid, *a_conv6_sum, 102);
    cuda_reduce_sum_sq<<<dim3(102, BATCH), 256, 256 * sizeof(float), streams[gpu]>>>(gpu, *a_conv6, *a_conv6_sum_mid, 256);
    cuda_reduce_sum<<<dim3(BATCH, 1), 128, 128 * sizeof(float), streams[gpu]>>>(gpu, *a_conv6_sum_mid, *a_conv6_sum_sq, 102);
    cuda_layernorm<<<dim3(102, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_conv6, *gamma_conv6, *beta_conv6, *a_layernorm6,
                                              *a_conv6_sum, *a_conv6_sum_sq);
    cuda_relu<<<BATCH*102, 256, 0, streams[gpu]>>>(gpu, *a_layernorm6, *a_relu6);
    cuda_maxpool1d<<<dim3(34, BATCH), 256, 0, streams[gpu]>>>(gpu, *a_relu6, *a_pool6);

    // Collapse
    cuda_collapse<<<BATCH*68, 128, 0, streams[gpu]>>>(gpu, *a_pool6, *a_collapse);

    // FC block 1 : Linear + ReLU
    cuda_linear<<<BATCH, 1024, 0, streams[gpu]>>>(gpu, *a_collapse, *w_fc1, *b_fc1, *a_linear1);
    cuda_relu<<<BATCH, 1024, 0, streams[gpu]>>>(gpu, *a_linear1, *a_relu7);

    // FC block 2 : Linear + ReLU
    cuda_linear<<<BATCH, 1024, 0, streams[gpu]>>>(gpu, *a_relu7, *w_fc2, *b_fc2, *a_linear2);
    cuda_relu<<<BATCH, 1024, 0, streams[gpu]>>>(gpu, *a_linear2, *a_relu8);

    // FC block 3 : Linear
    cuda_linear<<<BATCH, 4, 0, streams[gpu]>>>(gpu, *a_relu8, *w_fc3, *b_fc3, *a_linear3);
  }

  // Wait for each GPU, copy to output
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);

    a_linear3->copy_to_cpu(gpu);
    CHECK_CUDA(cudaStreamSynchronize(streams[gpu]));

    for (int i = 0; i < BATCH; i++) {
      float max_val = -1e99f;
      int max_idx = 0;
      for (int j = 0; j < a_linear3->shape[1]; j++) {
        if (a_linear3->buf[j + i * a_linear3->shape[1]] > max_val) {
          max_val = a_linear3->buf[j + i * a_linear3->shape[1]];
          max_idx = j;
        }
      }
      output->buf[i + gpu * BATCH] = max_idx;
    }
  }

  // Gather
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gather(
    output->buf, output->num_elem(), MPI_FLOAT,
    output_, output->num_elem(), MPI_FLOAT,
    0, MPI_COMM_WORLD);
}

__global__ void cuda_conv1d(int gpu, Tensor input, Tensor weight, Tensor bias, Tensor output) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int IC = input.shape[1];
  int IL = input.shape[2];
  int IS = IC * IL;
  int OL = output.shape[2];
  int OS = output.shape[1] * output.shape[2];
  int KS = weight.shape[2];

  int oc = i / OL;
  int ol = i % OL;

  float val = bias.gbuf[gpu][oc];
  for (int ic = 0; ic < IC; ++ic)
    for (int ks = 0; ks < KS; ++ks)
      val += input.gbuf[gpu][ks + ol + ic * IL + j * IS] *
        weight.gbuf[gpu][ks + ic * KS + oc * IC * KS];
  output.gbuf[gpu][i + j * OS] = val;
}

void conv1d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride = 1, int padding = 0, int dilation = 1,
            bool has_bias = true) {
  int out_channels = weight->shape[0];
  int in_channels = weight->shape[1];
  int kernel_size = weight->shape[2];
  int input_length = input->shape[2];
  int output_length =
      (input->shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

  for (int oc = 0; oc < out_channels; ++oc) {
    for (int ol = 0; ol < output_length; ++ol) {
      float val = 0.0f;
      int offset = ol;
      for (int ic = 0; ic < in_channels; ++ic) {
        for (int ks = 0; ks < kernel_size; ++ks) {
          val += weight->buf[oc * in_channels * kernel_size + ic * kernel_size + ks] *
                 input->buf[ic * input_length + ks + offset];
        }
      }
      if (has_bias) val += bias->buf[oc];
      output->buf[oc * output_length + ol] = val;
    }
  }
}


__global__ void cuda_relu(int gpu, Tensor input, Tensor output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (input.gbuf[gpu][i] > 0.0f)
    output.gbuf[gpu][i] = input.gbuf[gpu][i];
  else
    output.gbuf[gpu][i] = 0.0f;
}

void relu(Tensor *input, Tensor *output) {
  for (int i = 0; i < input->num_elem(); ++i) {
    if (input->buf[i] > 0.0f)
      output->buf[i] = input->buf[i];
    else
      output->buf[i] = 0.0f;
  }
}

__global__ void cuda_maxpool1d(int gpu, Tensor input, Tensor output) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int IS = input.shape[1] * input.shape[2];
  int OS = output.shape[1] * output.shape[2];

  float mx = -1e99f;
  for (int ks = 0; ks < 3; ++ks) {
    float val = input.gbuf[gpu][ks + i * 3 + j * IS];
    if (val > mx) mx = val;
  }
  output.gbuf[gpu][i + j * OS] = mx;
}

void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride) {
  int IL = input->shape[2];
  int OC = output->shape[1];
  int OL = output->shape[2];

  for (int oc = 0; oc < OC; ++oc) {
    for (int ol = 0; ol < OL; ++ol) {
      float mx = -1e99;
      for (int ks = 0; ks < kernel_size; ++ks) {
        float val = input->buf[oc * IL + ks + ol * stride];
        if (val > mx) mx = val;
      }
      output->buf[oc * OL + ol] = mx;
    }
  }
}

__global__ void cuda_collapse(int gpu, Tensor input, Tensor output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  output.gbuf[gpu][i] = input.gbuf[gpu][i];
}

void collapse(Tensor *input, Tensor *output) {
  for (int i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = input->buf[i];
  }
}

__global__ void cuda_linear(int gpu, Tensor input, Tensor weight, Tensor bias, Tensor output) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int IC = input.shape[1];
  int OC = output.shape[1];

  float val = bias.gbuf[gpu][tid];
  for (int ic = 0; ic < IC; ++ic)
    val += input.gbuf[gpu][ic + bid * IC] * weight.gbuf[gpu][ic + tid * IC];
  output.gbuf[gpu][tid + bid * OC] = val;
}

void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias) {
  int IC = input->shape[1];
  int OC = output->shape[1];

  for (int oc = 0; oc < OC; ++oc) {
    float val = 0.0;
    for (int ic = 0; ic < IC; ++ic) {
      val += input->buf[ic] * weight->buf[oc * IC + ic];
    }
    if (has_bias) val += bias->buf[oc];
    output->buf[oc] = val;
  }
}

__global__ void cuda_reduce_sum(int gpu, Tensor input, Tensor output, int N) {
  extern __shared__ float sdata[];

  int IS = input.shape[1] * input.shape[2];
  int OS = output.shape[1];

  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int i = bid * blockDim.x + tid;

  if (tid < N)
    sdata[tid] = input.gbuf[gpu][i + j * IS];
  else
    sdata[tid] = 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    output.gbuf[gpu][bid + j * OS] = sdata[0];
}

__global__ void cuda_reduce_sum_sq(int gpu, Tensor input, Tensor output, int N) {
  extern __shared__ float sdata[];

  int IS = input.shape[1] * input.shape[2];
  int OS = output.shape[1];

  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int i = bid * blockDim.x + tid;

  if (tid < N)
    sdata[tid] = input.gbuf[gpu][i + j * IS];
  else
    sdata[tid] = 0.0f;
  sdata[tid] *= sdata[tid];
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    output.gbuf[gpu][bid + j * OS] = sdata[0];
}

__global__ void cuda_layernorm(int gpu, Tensor input, Tensor gamma, Tensor beta, Tensor output,
                               Tensor sum, Tensor sum_sq) {
  int IS = input.shape[1] * input.shape[2];

  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int off = i + j * IS;

  // E[X], E[X^2]
  float mean1 = sum.gbuf[gpu][j] / IS;
  float mean2 = sum_sq.gbuf[gpu][j] / IS;
  // V[X]
  float var = mean2 - mean1 * mean1;

  output.gbuf[gpu][off] =
    (input.gbuf[gpu][off] - mean1) / sqrtf(var + 1e-5) * gamma.gbuf[gpu][i] + beta.gbuf[gpu][i];
}

void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output) {
  // E[X], E[X^2]
  float sum1 = 0.0f, sum2 = 0.0f;
  for (int i = 0; i < input->num_elem(); ++i) {
      sum1 += input->buf[i];
      sum2 += input->buf[i] * input->buf[i];
  }
  float mean1 = sum1 / (float)input->num_elem();
  float mean2 = sum2 / (float)input->num_elem();

  // V[X]
  float var = mean2 - mean1 * mean1;

  // Normalization
  for (int i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = (input->buf[i] - mean1) / sqrtf(var + 1e-5) * gamma->buf[i] + beta->buf[i];
  }
}

// load the parameter binary file and store parameters into Tensors
// Only the first process (root, mpi_rank == 0) has the parameter
// You must broadcast it to the others
void initialize_classifier(float *parameter, int N) {
  vector<Tensor *> params, activations;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (mpi_rank == 0) {
    // Read params from disk
    w_conv1 = new Tensor({256, 70, 7}, parameter + OFFSET0);
    b_conv1 = new Tensor({256}, parameter + OFFSET1);
    gamma_conv1 = new Tensor({256, 1008}, parameter + OFFSET2);
    beta_conv1 = new Tensor({256, 1008}, parameter + OFFSET3);

    w_conv2 = new Tensor({256, 256, 7}, parameter + OFFSET4);
    b_conv2 = new Tensor({256}, parameter + OFFSET5);
    w_conv3 = new Tensor({256, 256, 3}, parameter + OFFSET6);
    b_conv3 = new Tensor({256}, parameter + OFFSET7);
    w_conv4 = new Tensor({256, 256, 3}, parameter + OFFSET8);
    b_conv4 = new Tensor({256}, parameter + OFFSET9);
    w_conv5 = new Tensor({256, 256, 3}, parameter + OFFSET10);
    b_conv5 = new Tensor({256}, parameter + OFFSET11);

    w_conv6 = new Tensor({256, 256, 3}, parameter + OFFSET12);
    b_conv6 = new Tensor({256}, parameter + OFFSET13);
    gamma_conv6 = new Tensor({256, 102}, parameter + OFFSET14);
    beta_conv6 = new Tensor({256, 102}, parameter + OFFSET15);

    w_fc1 = new Tensor({1024, 8704}, parameter + OFFSET16);
    b_fc1 = new Tensor({1024}, parameter + OFFSET17);
    w_fc2 = new Tensor({1024, 1024}, parameter + OFFSET18);
    b_fc2 = new Tensor({1024}, parameter + OFFSET19);
    w_fc3 = new Tensor({4, 1024}, parameter + OFFSET20);
    b_fc3 = new Tensor({4}, parameter + OFFSET21);
  } else {
    // Allocate memory only, will be broadcasted later
    w_conv1 = new Tensor({256, 70, 7});
    b_conv1 = new Tensor({256});
    gamma_conv1 = new Tensor({256, 1008});
    beta_conv1 = new Tensor({256, 1008});

    w_conv2 = new Tensor({256, 256, 7});
    b_conv2 = new Tensor({256});
    w_conv3 = new Tensor({256, 256, 3});
    b_conv3 = new Tensor({256});
    w_conv4 = new Tensor({256, 256, 3});
    b_conv4 = new Tensor({256});
    w_conv5 = new Tensor({256, 256, 3});
    b_conv5 = new Tensor({256});

    w_conv6 = new Tensor({256, 256, 3});
    b_conv6 = new Tensor({256});
    gamma_conv6 = new Tensor({256, 102});
    beta_conv6 = new Tensor({256, 102});

    w_fc1 = new Tensor({1024, 8704});
    b_fc1 = new Tensor({1024});
    w_fc2 = new Tensor({1024, 1024});
    b_fc2 = new Tensor({1024});
    w_fc3 = new Tensor({4, 1024});
    b_fc3 = new Tensor({4});
  }

  // Prepare to broadcast
  params.push_back(w_conv1); params.push_back(b_conv1);
  params.push_back(gamma_conv1); params.push_back(beta_conv1);

  params.push_back(w_conv2); params.push_back(b_conv2);
  params.push_back(w_conv3); params.push_back(b_conv3);
  params.push_back(w_conv4); params.push_back(b_conv4);
  params.push_back(w_conv5); params.push_back(b_conv5);

  params.push_back(w_conv6); params.push_back(b_conv6);
  params.push_back(gamma_conv6); params.push_back(beta_conv6);

  params.push_back(w_fc1); params.push_back(b_fc1);
  params.push_back(w_fc2); params.push_back(b_fc2);
  params.push_back(w_fc3); params.push_back(b_fc3);

  // Activations are allocated everywhere
  a_conv1 = new Tensor({BATCH, 256, 1008});
  a_conv1_sum_mid = new Tensor({BATCH, 1024}); // rounded up
  a_conv1_sum = new Tensor({BATCH, 1});
  a_conv1_sum_sq = new Tensor({BATCH, 1});
  a_layernorm1 = new Tensor({BATCH, 256, 1008});
  a_relu1 = new Tensor({BATCH, 256, 1008});
  a_pool1 = new Tensor({BATCH, 256, 336});

  a_conv2 = new Tensor({BATCH, 256, 330});
  a_relu2 = new Tensor({BATCH, 256, 330});
  a_pool2 = new Tensor({BATCH, 256, 110});

  a_conv3 = new Tensor({BATCH, 256, 108});
  a_relu3 = new Tensor({BATCH, 256, 108});

  a_conv4 = new Tensor({BATCH, 256, 106});
  a_relu4 = new Tensor({BATCH, 256, 106});

  a_conv5 = new Tensor({BATCH, 256, 104});
  a_relu5 = new Tensor({BATCH, 256, 104});

  a_conv6 = new Tensor({BATCH, 256, 102});
  a_conv6_sum_mid = new Tensor({BATCH, 128}); // rounded up
  a_conv6_sum = new Tensor({BATCH, 1});
  a_conv6_sum_sq = new Tensor({BATCH, 1});
  a_layernorm6 = new Tensor({BATCH, 256, 102});
  a_relu6 = new Tensor({BATCH, 256, 102});
  a_pool6 = new Tensor({BATCH, 256, 34});
  a_collapse = new Tensor({BATCH, 8704});

  a_linear1 = new Tensor({BATCH, 1024});
  a_relu7 = new Tensor({BATCH, 1024});
  a_linear2 = new Tensor({BATCH, 1024});
  a_relu8 = new Tensor({BATCH, 1024});
  a_linear3 = new Tensor({BATCH, 4});

  // Prepare activations
  activations.push_back(a_conv1);
  activations.push_back(a_conv1_sum_mid);
  activations.push_back(a_conv1_sum);
  activations.push_back(a_conv1_sum_sq);
  activations.push_back(a_layernorm1);
  activations.push_back(a_relu1);
  activations.push_back(a_pool1);

  activations.push_back(a_conv2); activations.push_back(a_relu2);
  activations.push_back(a_pool2);
  activations.push_back(a_conv3); activations.push_back(a_relu3);
  activations.push_back(a_conv4); activations.push_back(a_relu4);
  activations.push_back(a_conv5); activations.push_back(a_relu5);

  activations.push_back(a_conv6);
  activations.push_back(a_conv6_sum_mid);
  activations.push_back(a_conv6_sum);
  activations.push_back(a_conv6_sum_sq);
  activations.push_back(a_layernorm6);
  activations.push_back(a_relu6);
  activations.push_back(a_pool6);
  activations.push_back(a_collapse);

  activations.push_back(a_linear1); activations.push_back(a_relu7);
  activations.push_back(a_linear2); activations.push_back(a_relu8);
  activations.push_back(a_linear3);

  // Broacast, allocate, copy
  for (auto t : params) {
    MPI_Bcast(t->buf, t->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    cudaStreamCreate(&streams[gpu]);
    for (auto t : params) {
      t->copy_to_gpu(gpu);
    }
    for (auto t : activations) {
      t->allocate_gpu(gpu);
    }
    CHECK_CUDA(cudaStreamSynchronize(streams[gpu]));
  }
}

// Free all dynamically allocated variables
void finalize_classifier() {
  if (mpi_rank == 0) {
    delete w_conv1;
    delete b_conv1;
    delete w_conv2;
    delete b_conv2;
    delete w_conv3;
    delete b_conv3;
    delete w_conv4;
    delete b_conv4;
    delete w_conv5;
    delete b_conv5;
    delete w_conv6;
    delete b_conv6;
    delete w_fc1;
    delete b_fc1;
    delete w_fc2;
    delete b_fc2;
    delete w_fc3;
    delete b_fc3;
    delete gamma_conv1;
    delete gamma_conv6;
    delete beta_conv1;
    delete beta_conv6;
    delete a_conv1;
    delete a_conv1_sum_mid;
    delete a_conv1_sum;
    delete a_conv1_sum_sq;
    delete a_layernorm1;
    delete a_relu1;
    delete a_pool1;
    delete a_conv2;
    delete a_relu2;
    delete a_pool2;
    delete a_conv3;
    delete a_relu3;
    delete a_conv4;
    delete a_relu4;
    delete a_conv5;
    delete a_relu5;
    delete a_conv6;
    delete a_conv6_sum_mid;
    delete a_conv6_sum;
    delete a_conv6_sum_sq;
    delete a_layernorm6;
    delete a_relu6;
    delete a_pool6;
    delete a_collapse;
    delete a_linear1;
    delete a_relu7;
    delete a_linear2;
    delete a_relu8;
    delete a_linear3;
  }
}
