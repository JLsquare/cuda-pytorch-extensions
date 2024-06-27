#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void sparse_linear_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const int* __restrict__ connections,
    int batch_size,
    int in_features,
    int out_features,
    int weight_per_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_features * batch_size) {
        int out_idx = idx / batch_size;
        int batch_idx = idx % batch_size;

        float sum = 0.0f;
        for (int i = 0; i < weight_per_out; ++i) {
            int in_idx = connections[out_idx * weight_per_out + i];
            sum += input[batch_idx * in_features + in_idx] * weights[out_idx * weight_per_out + i];
        }
        output[batch_idx * out_features + out_idx] = sum;
    }
}

torch::Tensor positional_sparse_linear_forward(
    torch::Tensor input,
    torch::Tensor connections,
    torch::Tensor weights
) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(connections.device().is_cuda(), "connections must be a CUDA tensor");
    TORCH_CHECK(weights.device().is_cuda(), "weights must be a CUDA tensor");
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = connections.size(0);
    int weight_per_out = connections.size(1);

    auto output = torch::empty({batch_size, out_features}, input.options());

    int threads = 256;
    int blocks = (out_features * batch_size + threads - 1) / threads;

    sparse_linear_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        connections.data_ptr<int>(),
        batch_size,
        in_features,
        out_features,
        weight_per_out
    );

    return output;
}

__global__ void sparse_linear_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weights,
    const int* __restrict__ connections,
    int batch_size,
    int in_features,
    int out_features,
    int weight_per_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_features * batch_size) {
        int out_idx = idx / batch_size;
        int batch_idx = idx % batch_size;
        
        float grad = grad_output[batch_idx * out_features + out_idx];
        
        for (int i = 0; i < weight_per_out; ++i) {
            int in_idx = connections[out_idx * weight_per_out + i];
            atomicAdd(&grad_input[batch_idx * in_features + in_idx], grad * weights[out_idx * weight_per_out + i]);
            atomicAdd(&grad_weights[out_idx * weight_per_out + i], grad * input[batch_idx * in_features + in_idx]);
        }
    }
}

std::vector<torch::Tensor> positional_sparse_linear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor connections
) {
    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weights.device().is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(connections.device().is_cuda(), "connections must be a CUDA tensor");

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = connections.size(0);
    int weight_per_out = connections.size(1);

    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);

    int threads = 256;
    int blocks = (out_features * batch_size + threads - 1) / threads;

    sparse_linear_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        connections.data_ptr<int>(),
        batch_size,
        in_features,
        out_features,
        weight_per_out
    );

    return {grad_input, grad_weights};
}