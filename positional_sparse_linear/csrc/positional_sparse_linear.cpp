#include <torch/extension.h>

torch::Tensor positional_sparse_linear_forward(
    torch::Tensor input,
    torch::Tensor connections,
    torch::Tensor weights);
    

std::vector<torch::Tensor> positional_sparse_linear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor connections);

PYBIND11_MODULE(positional_sparse_linear_csrc, m) {
    m.def("positional_sparse_linear_forward", &positional_sparse_linear_forward, "Positional Sparse Linear forward",
        py::arg("input"),
        py::arg("connections"),
        py::arg("weights"));

    m.def("positional_sparse_linear_backward", &positional_sparse_linear_backward, "Positional Sparse Linear backward",
        py::arg("grad_output"),
        py::arg("input"),
        py::arg("weights"),
        py::arg("connections"));
}