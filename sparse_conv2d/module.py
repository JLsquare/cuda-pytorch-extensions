import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load

sparse_conv2d = load(name="sparse_conv2d_csrc", sources=[
    "sparse_conv2d/csrc/sparse_conv2d.cpp",
    "sparse_conv2d/csrc/sparse_conv2d_kernel.cu"
], verbose=True)

class SparseConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, connections: torch.Tensor,
                out_channels: int, kernel_size: int, channels_per_filter: int) -> torch.Tensor:
        """
        Forward pass of the sparse convolution.

        params: input: torch.Tensor, shape: (batch_size, in_channels, height, width)
        params: weight: torch.Tensor, shape: (out_channels, channels_per_filter, kernel_size, kernel_size)
        params: connections: torch.Tensor, shape: (out_channels, channels_per_filter)
        params: out_channels: int
        params: kernel_size: int
        params: channels_per_filter: int
        returns: output: torch.Tensor, shape: (batch_size, out_channels, height, width)
        """
        ctx.save_for_backward(input, weight, connections)
        ctx.out_channels = out_channels
        ctx.kernel_size = kernel_size
        ctx.channels_per_filter = channels_per_filter
        return sparse_conv2d.sparse_conv2d_forward(input, weight, connections, out_channels,
                                     kernel_size, channels_per_filter)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the sparse convolution.

        params: grad_output: torch.Tensor, shape: (batch_size, out_channels, height, width)
        returns: grad_input: torch.Tensor, shape: (batch_size, in_channels, height, width)
        """
        input, weight, connections = ctx.saved_tensors
        grad_input, grad_weight = sparse_conv2d.sparse_conv2d_backward(
            grad_output, input, weight, connections,
            ctx.out_channels, ctx.kernel_size, ctx.channels_per_filter)
        return grad_input, grad_weight, None, None, None, None

class SparseConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, channels_per_filter: int,
                position_influence: float = 0.1, no_position_influence: bool = False):
        """
        Sparse Convolutional Layer.

        params: in_channels: int
        params: out_channels: int
        params: kernel_size: int
        params: channels_per_filter: int
        params: position_influence: float
        params: no_position_influence: bool
        """
        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.channels_per_filter = channels_per_filter

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, channels_per_filter, kernel_size, kernel_size))

        if no_position_influence:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.register_buffer('connections', torch.randint(
                0, in_channels, (out_channels, channels_per_filter), dtype=torch.int32))
        else:
            device = self.weight.device

            in_pos = torch.arange(in_channels, device=device) / in_channels
            out_pos = torch.arange(out_channels, device=device) / out_channels

            pos_diff = torch.abs(out_pos.unsqueeze(1) - in_pos.unsqueeze(0))
            pos_prob = torch.exp(-pos_diff / position_influence)

            self.register_buffer('connections', torch.multinomial(pos_prob, channels_per_filter, replacement=True).to(torch.int32))

            fan_in = channels_per_filter * kernel_size ** 2
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.weight, -bound, bound)

            pos_scale = torch.exp(-pos_diff / position_influence)
            pos_scale = pos_scale[:, :channels_per_filter].unsqueeze(-1).unsqueeze(-1)
            self.weight.data *= pos_scale.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sparse convolution.

        params: input: torch.Tensor, shape: (batch_size, in_channels, height, width)
        returns: output: torch.Tensor, shape: (batch_size, out_channels, height
        """
        return SparseConv2dFunction.apply(input, self.weight, self.connections,
                                          self.out_channels, self.kernel_size,
                                          self.channels_per_filter)