#include <math.h>
#include <mpi.h>
#include <cassert>

#include "classifier.h"
#include "util.h"

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static int mpi_rank;

// Multi-dimensional matrix containing fp32 elements
struct Tensor {
  Tensor(std::vector<int> shape_);
  Tensor(std::vector<int> shape_, float *buf_);
  ~Tensor();
  __host__ __device__ int num_elem() const;
  void fill_zeros();

  void allocate_gpu();
  void copy_to_gpu(cudaStream_t stream = 0);
  void copy_to_cpu(cudaStream_t stream = 0);

  float *buf = nullptr;
  float *gbuf = nullptr;
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

void Tensor::allocate_gpu() {
  CHECK_CUDA(cudaMalloc(&gbuf, num_elem() * sizeof(float)));
}

void Tensor::copy_to_gpu(cudaStream_t stream) {
  if (gbuf == nullptr)
    allocate_gpu();
  CHECK_CUDA(cudaMemcpyAsync(gbuf, buf, num_elem() * sizeof(float), cudaMemcpyHostToDevice, stream));
}

void Tensor::copy_to_cpu(cudaStream_t stream) {
  CHECK_CUDA(cudaMemcpyAsync(buf, gbuf, num_elem() * sizeof(float), cudaMemcpyDeviceToHost, stream));
}

// Parameters
Tensor *w_conv1, *w_conv2, *w_conv3, *w_conv4, *w_conv5, *w_conv6, *b_conv1,
    *b_conv2, *b_conv3, *b_conv4, *b_conv5, *b_conv6, *w_fc1, *w_fc2, *w_fc3,
    *b_fc1, *b_fc2, *b_fc3, *gamma_conv1, *beta_conv1, *gamma_conv6, *beta_conv6;

// Activations
Tensor *a_conv1, *a_layernorm1, *a_relu1, *a_pool1;
Tensor *a_conv1_sum, *a_conv1_sum_sq; // for layernorm
Tensor *a_conv2, *a_relu2, *a_pool2;
Tensor *a_conv3, *a_relu3;
Tensor *a_conv4, *a_relu4;
Tensor *a_conv5, *a_relu5;
Tensor *a_conv6, *a_layernorm6, *a_relu6, *a_pool6;
Tensor *a_conv6_sum, *a_conv6_sum_sq; // for layernorm
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
__global__ void cuda_conv1d(Tensor input, Tensor weight, Tensor bias, Tensor output);
__global__ void cuda_relu(Tensor input, Tensor output);
__global__ void cuda_maxpool1d(Tensor input, Tensor output); // always 3x3
__global__ void cuda_collapse(Tensor input, Tensor output);
__global__ void cuda_linear(Tensor input, Tensor weight, Tensor bias, Tensor output);
__global__ void cuda_layernorm(Tensor input, Tensor gamma, Tensor beta, Tensor output,
                               Tensor sum, Tensor sum_sq);

// Cuda operations
__global__ void cuda_reduce_sum(Tensor input, Tensor output);
__global__ void cuda_reduce_sum_sq(Tensor input, Tensor output);

// Parallelization method is totally up to you, but you should gather
// the output at rank 0
void classifier(float *input_, float *output_, int N) {
  if (mpi_rank == 0) {
    for (int n = 0; n < N; ++n) {  // N input sentences

      // Load one input sentence from input
      Tensor *one_input = new Tensor({1, VOCAB_SIZE, MAX_LENGTH}, input_ + n * VOCAB_SIZE * MAX_LENGTH);

      // Conv block 1 : Conv1d + LayerNorm + ReLU + MaxPool1d
      one_input->copy_to_gpu();
      cuda_conv1d<<<256,1008>>>(*one_input, *w_conv1, *b_conv1, *a_conv1);
      cuda_reduce_sum<<<1008, 256, 256 * sizeof(float)>>>(*a_conv1, *a_conv1_sum);
      cuda_reduce_sum<<<1, 1024, 1024 * sizeof(float)>>>(*a_conv1_sum, *a_conv1_sum);
      cuda_reduce_sum_sq<<<1008, 256, 256 * sizeof(float)>>>(*a_conv1, *a_conv1_sum_sq);
      cuda_reduce_sum<<<1, 1024, 1024 * sizeof(float)>>>(*a_conv1_sum_sq, *a_conv1_sum_sq);
      cuda_layernorm<<<256, 1008>>>(*a_conv1, *gamma_conv1, *beta_conv1, *a_layernorm1,
                                    *a_conv1_sum, *a_conv1_sum_sq);
      cuda_relu<<<256, 1008>>>(*a_layernorm1, *a_relu1);
      cuda_maxpool1d<<<256, 336>>>(*a_relu1, *a_pool1);

      // Conv block 2 : Conv1d + ReLU + MaxPool1d
      cuda_conv1d<<<256, 330>>>(*a_pool1, *w_conv2, *b_conv2, *a_conv2);
      cuda_relu<<<256, 330>>>(*a_conv2, *a_relu2);
      cuda_maxpool1d<<<256, 110>>>(*a_relu2, *a_pool2);

      // Conv block 3 : Conv1d + ReLU
      cuda_conv1d<<<256, 108>>>(*a_pool2, *w_conv3, *b_conv3, *a_conv3);
      cuda_relu<<<256, 108>>>(*a_conv3, *a_relu3);

      // Conv block 4 : Conv1d + ReLU
      cuda_conv1d<<<256, 106>>>(*a_relu3, *w_conv4, *b_conv4, *a_conv4);
      cuda_relu<<<256, 106>>>(*a_conv4, *a_relu4);

      // Conv block 5 : Conv1d + ReLU
      cuda_conv1d<<<256, 104>>>(*a_relu4, *w_conv5, *b_conv5, *a_conv5);
      cuda_relu<<<256, 104>>>(*a_conv5, *a_relu5);

      // Conv block 6 : Conv1d + LayerNorm + ReLU + MaxPool1d
      cuda_conv1d<<<256, 102>>>(*a_relu5, *w_conv6, *b_conv6, *a_conv6);
      cuda_reduce_sum<<<102, 256, 256 * sizeof(float)>>>(*a_conv6, *a_conv6_sum);
      cuda_reduce_sum<<<1, 128, 128 * sizeof(float)>>>(*a_conv6_sum, *a_conv6_sum);
      cuda_reduce_sum_sq<<<102, 256, 256 * sizeof(float)>>>(*a_conv6, *a_conv6_sum_sq);
      cuda_reduce_sum<<<1, 128, 128 * sizeof(float)>>>(*a_conv6_sum_sq, *a_conv6_sum_sq);
      cuda_layernorm<<<256, 102>>>(*a_conv6, *gamma_conv6, *beta_conv6, *a_layernorm6,
                                    *a_conv6_sum, *a_conv6_sum_sq);
      cuda_relu<<<256, 102>>>(*a_layernorm6, *a_relu6);
      cuda_maxpool1d<<<256, 34>>>(*a_relu6, *a_pool6);

      // Collapse
      cuda_collapse<<<68, 128>>>(*a_pool6, *a_collapse);

      // FC block 1 : Linear + ReLU
      cuda_linear<<<1, 1024>>>(*a_collapse, *w_fc1, *b_fc1, *a_linear1);
      cuda_relu<<<1, 1024>>>(*a_linear1, *a_relu7);

      // FC block 2 : Linear + ReLU
      cuda_linear<<<1, 1024>>>(*a_relu7, *w_fc2, *b_fc2, *a_linear2);
      cuda_relu<<<1, 1024>>>(*a_linear2, *a_relu8);

      // FC block 3 : Linear
      cuda_linear<<<1, 4>>>(*a_relu8, *w_fc3, *b_fc3, *a_linear3);

      a_linear3->copy_to_cpu();
      CHECK_CUDA(cudaStreamSynchronize(0));

      float max_val = -1e99f;
      int max_idx = 0;
      for (int i = 0; i < a_linear3->num_elem(); ++i) {
        if (a_linear3->buf[i] > max_val) {
          max_val = a_linear3->buf[i];
          max_idx = i;
        }
      }

      output_[n] = max_idx;
    }  // end N input sentences loop
  }    // if mpi_rank == 0
}

__global__ void cuda_conv1d(Tensor input, Tensor weight, Tensor bias, Tensor output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int in_channels = weight.shape[1];
  int kernel_size = weight.shape[2];
  int input_length = input.shape[2];
  int output_length = output.shape[2];

  int oc = i / output_length;
  int ol = i % output_length;

  float val = 0.0f;
  for (int ic = 0; ic < in_channels; ++ic)
    for (int ks = 0; ks < kernel_size; ++ks)
      val += weight.gbuf[oc * in_channels * kernel_size + ic * kernel_size + ks] *
        input.gbuf[ic * input_length + ks + ol];
  val += bias.gbuf[oc];
  output.gbuf[i] = val;
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


__global__ void cuda_relu(Tensor input, Tensor output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (input.gbuf[i] > 0.0f)
    output.gbuf[i] = input.gbuf[i];
  else
    output.gbuf[i] = 0.0f;
}

void relu(Tensor *input, Tensor *output) {
  for (int i = 0; i < input->num_elem(); ++i) {
    if (input->buf[i] > 0.0f)
      output->buf[i] = input->buf[i];
    else
      output->buf[i] = 0.0f;
  }
}

__global__ void cuda_maxpool1d(Tensor input, Tensor output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int input_length = input.shape[2];
  int output_length = output.shape[2];

  int oc = i / output_length;
  int ol = i % output_length;

  float mx = -1e99f;
  for (int ks = 0; ks < 3; ++ks) {
    float val = input.gbuf[oc * input_length + ks + ol * 3];
    if (val > mx) mx = val;
  }
  output.gbuf[i] = mx;
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

__global__ void cuda_collapse(Tensor input, Tensor output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  output.gbuf[i] = input.gbuf[i];
}

void collapse(Tensor *input, Tensor *output) {
  for (int i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = input->buf[i];
  }
}

__global__ void cuda_linear(Tensor input, Tensor weight, Tensor bias, Tensor output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int IC = input.shape[1];
  int OC = output.shape[1];

  float val = bias.gbuf[i];
  for (int ic = 0; ic < IC; ++ic)
    val += input.gbuf[ic] * weight.gbuf[i * IC + ic];
  output.gbuf[i] = val;
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

__global__ void cuda_reduce_sum(Tensor input, Tensor output) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  sdata[tid] = input.gbuf[blockIdx.x * blockDim.x + tid];
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    output.gbuf[blockIdx.x] = sdata[0];
}

__global__ void cuda_reduce_sum_sq(Tensor input, Tensor output) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  sdata[tid] = input.gbuf[blockIdx.x * blockDim.x + tid];
  sdata[tid] *= sdata[tid];
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    output.gbuf[blockIdx.x] = sdata[0];
}

__global__ void cuda_layernorm(Tensor input, Tensor gamma, Tensor beta, Tensor output,
                               Tensor sum, Tensor sum_sq) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // E[X], E[X^2]
  float mean1 = sum.gbuf[0] / input.num_elem();
  float mean2 = sum_sq.gbuf[0] / (float)input.num_elem();

  // V[X]
  float var = mean2 - mean1 * mean1;

  output.gbuf[idx] = (input.gbuf[idx] - mean1) / sqrtf(var + 1e-5) * gamma.gbuf[idx] + beta.gbuf[idx];
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
  CHECK_CUDA(cudaSetDevice(0));

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    w_conv1 = new Tensor({256, 70, 7}, parameter + OFFSET0);
    w_conv1->copy_to_gpu();
    b_conv1 = new Tensor({256}, parameter + OFFSET1);
    b_conv1->copy_to_gpu();
    gamma_conv1 = new Tensor({256, 1008}, parameter + OFFSET2);
    gamma_conv1->copy_to_gpu();
    beta_conv1 = new Tensor({256, 1008}, parameter + OFFSET3);
    beta_conv1->copy_to_gpu();

    w_conv2 = new Tensor({256, 256, 7}, parameter + OFFSET4);
    w_conv2->copy_to_gpu();
    b_conv2 = new Tensor({256}, parameter + OFFSET5);
    b_conv2->copy_to_gpu();
    w_conv3 = new Tensor({256, 256, 3}, parameter + OFFSET6);
    w_conv3->copy_to_gpu();
    b_conv3 = new Tensor({256}, parameter + OFFSET7);
    b_conv3->copy_to_gpu();
    w_conv4 = new Tensor({256, 256, 3}, parameter + OFFSET8);
    w_conv4->copy_to_gpu();
    b_conv4 = new Tensor({256}, parameter + OFFSET9);
    b_conv4->copy_to_gpu();
    w_conv5 = new Tensor({256, 256, 3}, parameter + OFFSET10);
    w_conv5->copy_to_gpu();
    b_conv5 = new Tensor({256}, parameter + OFFSET11);
    b_conv5->copy_to_gpu();

    w_conv6 = new Tensor({256, 256, 3}, parameter + OFFSET12);
    w_conv6->copy_to_gpu();
    b_conv6 = new Tensor({256}, parameter + OFFSET13);
    b_conv6->copy_to_gpu();
    gamma_conv6 = new Tensor({256, 102}, parameter + OFFSET14);
    gamma_conv6->copy_to_gpu();
    beta_conv6 = new Tensor({256, 102}, parameter + OFFSET15);
    beta_conv6->copy_to_gpu();
    w_fc1 = new Tensor({1024, 8704}, parameter + OFFSET16);
    w_fc1->copy_to_gpu();
    b_fc1 = new Tensor({1024}, parameter + OFFSET17);
    b_fc1->copy_to_gpu();
    w_fc2 = new Tensor({1024, 1024}, parameter + OFFSET18);
    w_fc2->copy_to_gpu();
    b_fc2 = new Tensor({1024}, parameter + OFFSET19);
    b_fc2->copy_to_gpu();
    w_fc3 = new Tensor({4, 1024}, parameter + OFFSET20);
    w_fc3->copy_to_gpu();
    b_fc3 = new Tensor({4}, parameter + OFFSET21);
    b_fc3->copy_to_gpu();

    a_conv1 = new Tensor({1, 256, 1008});
    a_conv1->allocate_gpu();
    a_conv1_sum = new Tensor({1, 1024}); // rounded up
    a_conv1_sum->allocate_gpu();
    a_conv1_sum_sq = new Tensor({1, 1024});
    a_conv1_sum_sq->allocate_gpu();
    a_layernorm1 = new Tensor({1, 256, 1008});
    a_layernorm1->allocate_gpu();
    a_relu1 = new Tensor({1, 256, 1008});
    a_relu1->allocate_gpu();
    a_pool1 = new Tensor({1, 256, 336});
    a_pool1->allocate_gpu();

    a_conv2 = new Tensor({1, 256, 330});
    a_conv2->allocate_gpu();
    a_relu2 = new Tensor({1, 256, 330});
    a_relu2->allocate_gpu();
    a_pool2 = new Tensor({1, 256, 110});
    a_pool2->allocate_gpu();

    a_conv3 = new Tensor({1, 256, 108});
    a_conv3->allocate_gpu();
    a_relu3 = new Tensor({1, 256, 108});
    a_relu3->allocate_gpu();

    a_conv4 = new Tensor({1, 256, 106});
    a_conv4->allocate_gpu();
    a_relu4 = new Tensor({1, 256, 106});
    a_relu4->allocate_gpu();

    a_conv5 = new Tensor({1, 256, 104});
    a_conv5->allocate_gpu();
    a_relu5 = new Tensor({1, 256, 104});
    a_relu5->allocate_gpu();

    a_conv6 = new Tensor({1, 256, 102});
    a_conv6->allocate_gpu();
    a_conv6_sum = new Tensor({1, 128}); // rounded up
    a_conv6_sum->allocate_gpu();
    a_conv6_sum_sq = new Tensor({1, 128});
    a_conv6_sum_sq->allocate_gpu();
    a_layernorm6 = new Tensor({1, 256, 102});
    a_layernorm6->allocate_gpu();
    a_relu6 = new Tensor({1, 256, 102});
    a_relu6->allocate_gpu();
    a_pool6 = new Tensor({1, 256, 34});
    a_pool6->allocate_gpu();

    a_collapse = new Tensor({1, 8704});
    a_collapse->allocate_gpu();

    a_linear1 = new Tensor({1, 1024});
    a_linear1->allocate_gpu();
    a_relu7 = new Tensor({1, 1024});
    a_relu7->allocate_gpu();

    a_linear2 = new Tensor({1, 1024});
    a_linear2->allocate_gpu();
    a_relu8 = new Tensor({1, 1024});
    a_relu8->allocate_gpu();

    a_linear3 = new Tensor({1, 4});
    a_linear3->allocate_gpu();
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
