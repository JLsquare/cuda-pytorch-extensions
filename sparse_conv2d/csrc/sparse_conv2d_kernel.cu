#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void sparse_conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const int* __restrict__ connections,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int channels_per_filter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = height - kernel_size + 1;
    int out_w = width - kernel_size + 1;
    int total_elements = batch_size * out_channels * out_h * out_w;

    if (idx < total_elements) {
        int b = idx / (out_channels * out_h * out_w);
        int oc = (idx / (out_h * out_w)) % out_channels;
        int h = (idx / out_w) % out_h;
        int w = idx % out_w;

        float sum = 0.0f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                for (int c = 0; c < channels_per_filter; ++c) {
                    int in_c = connections[oc * channels_per_filter + c];
                    int weight_idx = ((oc * channels_per_filter + c) * kernel_size + kh) * kernel_size + kw;
                    int input_idx = ((b * in_channels + in_c) * height + h + kh) * width + w + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
        output[idx] = sum;
    }
}

__global__ void sparse_conv2d_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weights,
    const int* __restrict__ connections,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int channels_per_filter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = height - kernel_size + 1;
    int out_w = width - kernel_size + 1;
    int total_elements = batch_size * out_channels * out_h * out_w;

    if (idx < total_elements) {
        int b = idx / (out_channels * out_h * out_w);
        int oc = (idx / (out_h * out_w)) % out_channels;
        int h = (idx / out_w) % out_h;
        int w = idx % out_w;

        float grad = grad_output[idx];

        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                for (int c = 0; c < channels_per_filter; ++c) {
                    int in_c = connections[oc * channels_per_filter + c];
                    int weight_idx = ((oc * channels_per_filter + c) * kernel_size + kh) * kernel_size + kw;
                    int input_idx = ((b * in_channels + in_c) * height + h + kh) * width + w + kw;
                    
                    atomicAdd(&grad_input[input_idx], grad * weights[weight_idx]);
                    atomicAdd(&grad_weights[weight_idx], grad * input[input_idx]);
                }
            }
        }
    }
}

torch::Tensor sparse_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor connections,
    int out_channels,
    int kernel_size,
    int channels_per_filter
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto out_height = height - kernel_size + 1;
    const auto out_width = width - kernel_size + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    const int threads = 1024;
    const int blocks = (batch_size * out_channels * out_height * out_width + threads - 1) / threads;

    sparse_conv2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        connections.data_ptr<int>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        channels_per_filter
    );

    return output;
}

std::vector<torch::Tensor> sparse_conv2d_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor connections,
    int out_channels,
    int kernel_size,
    int channels_per_filter
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto out_height = height - kernel_size + 1;
    const auto out_width = width - kernel_size + 1;

    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);

    const int threads = 1024;
    const int blocks = (batch_size * out_channels * out_height * out_width + threads - 1) / threads;

    sparse_conv2d_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        connections.data_ptr<int>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        channels_per_filter
    );

    return {grad_input, grad_weights};
}