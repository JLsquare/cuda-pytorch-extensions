#include <torch/extension.h>

torch::Tensor sparse_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor connections,
    int out_channels,
    int kernel_size,
    int channels_per_filter
);

std::vector<torch::Tensor> sparse_conv2d_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor connections,
    int out_channels,
    int kernel_size,
    int channels_per_filter
);

PYBIND11_MODULE(sparse_conv2d_csrc, m) {
    m.def("sparse_conv2d_forward", &sparse_conv2d_forward, "Sparse Conv2d forward",
        py::arg("input"),
        py::arg("weights"),
        py::arg("connections"),
        py::arg("out_channels"),
        py::arg("kernel_size"),
        py::arg("channels_per_filter")
    );

    m.def("sparse_conv2d_backward", &sparse_conv2d_backward, "Sparse Conv2d backward",
        py::arg("grad_output"),
        py::arg("input"),
        py::arg("weights"),
        py::arg("connections"),
        py::arg("out_channels"),
        py::arg("kernel_size"),
        py::arg("channels_per_filter")
    );
}