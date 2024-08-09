import torch
import math
from torch.utils.cpp_extension import load

positional_sparse_linear = load(name='positional_sparse_linear_csrc', sources=[
    'positional_sparse_linear/csrc/positional_sparse_linear.cpp',
    'positional_sparse_linear/csrc/positional_sparse_linear_kernel.cu',
], verbose=True)

class PositionalSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, connections: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional sparse linear function

        params: input: torch.Tensor: input tensor , shape: (batch_size, in_features)
        params: connections: torch.Tensor: connections tensor, shape: (out_features, weight_per_out)
        params: weights: torch.Tensor: weights tensor, shape: (out_features, weight_per_out)
        return: torch.Tensor: output tensor, shape: (batch_size, out_features)
        """
        output = positional_sparse_linear.positional_sparse_linear_forward(input, connections, weights)
        ctx.save_for_backward(input, connections, weights)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the positional sparse linear function

        params: grad_output: torch.Tensor: gradient of the output tensor, shape: (batch_size, out_features)
        return: torch.Tensor: gradient of the input tensor, shape: (batch_size, in_features)
        """
        input, connections, weights = ctx.saved_tensors
        grad_input, grad_weight = positional_sparse_linear.positional_sparse_linear_backward(
            grad_output, input, weights, connections)
        return grad_input, None, grad_weight

class PositionalSparseLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, weight_per_out: int = 5, position_influence: int = 0.1, no_position_influence: bool = False):
        """
        Initialize the PositionalSparseLinear module

        params: in_features: int: number of input features
        params: out_features: int: number of output features
        params: weight_per_out: int: number of weights per output
        params: position_influence: int: influence of the position
        params: no_position_influence: bool: whether to use position influence
        """
        super(PositionalSparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_per_out = weight_per_out

        assert torch.cuda.is_available(), "PositionalSparseLinear2d is only available on CUDA"

        if no_position_influence:
            # Sample the connections randomly
            connections = torch.randint(in_features, (out_features, weight_per_out), device='cuda')
            self.connections = connections.to(torch.int32)

            # Initialize the weights
            bound = 1 / math.sqrt(weight_per_out)
            self.weights = torch.nn.Parameter(torch.empty(out_features, weight_per_out, device='cuda').uniform_(-bound, bound))
        else:
            # [0 / n, 1 / n, ..., (n - 1) / n]
            i_pos = torch.arange(in_features, device='cuda') / in_features
            o_pos = torch.arange(out_features, device='cuda') / out_features

            # Calculate the positional influence
            pos_diff = torch.abs(o_pos.unsqueeze(1) - i_pos.unsqueeze(0))
            pos_prob = torch.exp(-pos_diff / position_influence)

            # Sample the connections
            self.connections = torch.multinomial(pos_prob, weight_per_out, replacement=True)
            self.connections = self.connections.to(torch.int32)

            # Initialize the weights with xavier uniform
            bound = 1 / math.sqrt(weight_per_out)
            self.weights = torch.nn.Parameter(torch.empty(out_features, weight_per_out, device='cuda').uniform_(-bound, bound))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalSparseLinear module

        params: input: torch.Tensor: input tensor, shape: (batch_size, in_features)
        return: torch.Tensor: output tensor, shape: (batch_size, out_features)
        """
        assert input.dim() == 2 and input.size(1) == self.in_features, \
            f"Input shape must be (batch_size, {self.in_features}), but got {input.shape}"
        assert input.is_cuda, "Input must be a CUDA tensor"
        
        return PositionalSparseLinearFunction.apply(input, self.connections, self.weights)

class PositionalSparseLinear2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, connections: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional sparse linear 2D function

        params: input: torch.Tensor: input tensor, shape: (batch_size, in_height, in_width)
        params: connections: torch.Tensor: connections tensor, shape: (out_height * out_width, weight_per_out)
        params: weights: torch.Tensor: weights tensor, shape: (out_height * out_width, weight_per_out)
        return: torch.Tensor: output tensor, shape: (batch_size, out_height, out_width)
        """
        input_flat = input.view(input.size(0), -1)
        output = positional_sparse_linear.positional_sparse_linear_forward(input_flat, connections, weights)
        ctx.save_for_backward(input, connections, weights)
        return output.view(input.size(0), -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the positional sparse linear 2D function

        params: grad_output: torch.Tensor: gradient of the output tensor, shape: (batch_size, out_height, out_width)
        return: torch.Tensor: gradient of the input tensor, shape: (batch_size, in_height, in_width)
        """
        input, connections, weights = ctx.saved_tensors
        input_flat = input.view(input.size(0), -1)
        grad_output_flat = grad_output.view(grad_output.size(0), -1)
        grad_input_flat, grad_weight = positional_sparse_linear.positional_sparse_linear_backward(
            grad_output_flat, input_flat, weights, connections)
        grad_input = grad_input_flat.view_as(input)
        return grad_input, None, grad_weight

class PositionalSparseLinear2d(torch.nn.Module):
    def __init__(self, in_height: int, in_width: int, out_height: int, out_width: int, weight_per_out: int = 5, 
                 position_influence: int = 0.1, no_position_influence: bool = False):
        """
        Initialize the PositionalSparseLinear2d module

        params: in_height: int: number of input height
        params: in_width: int: number of input width
        params: out_height: int: number of output height
        params: out_width: int: number of output width
        params: weight_per_out: int: number of weights per output
        params: position_influence: int: influence of the position
        params: no_position_influence: bool: whether to use position influence
        """
        super(PositionalSparseLinear2d, self).__init__()
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = out_height
        self.out_width = out_width
        self.weight_per_out = weight_per_out

        assert torch.cuda.is_available(), "PositionalSparseLinear2d is only available on CUDA"

        if no_position_influence:
            # Sample the connections randomly
            connections_2d = torch.randint(in_height * in_width, (out_height * out_width, weight_per_out), device='cuda')
            self.connections = connections_2d.to(torch.int32)

            # Initialize the weights
            bound = 1 / math.sqrt(weight_per_out)
            self.weights = torch.nn.Parameter(torch.empty(out_height * out_width, weight_per_out, device='cuda').uniform_(-bound, bound))
        else:
            # [0 / n, 1 / n, ..., (n - 1) / n]
            w_i_pos = torch.arange(in_width, device='cuda') / in_width
            h_i_pos = torch.arange(in_height, device='cuda') / in_height
            w_o_pos = torch.arange(out_width, device='cuda') / out_width
            h_o_pos = torch.arange(out_height, device='cuda') / out_height

            # Calculate the positional influence
            w_pos_diff = torch.abs(w_o_pos.unsqueeze(1) - w_i_pos.unsqueeze(0))
            h_pos_diff = torch.abs(h_o_pos.unsqueeze(1) - h_i_pos.unsqueeze(0))
            pos_diff = torch.sqrt(w_pos_diff.unsqueeze(2).unsqueeze(3) ** 2 +
                                h_pos_diff.unsqueeze(0).unsqueeze(1) ** 2)
            pos_prob = torch.exp(-pos_diff / position_influence)

            # Sample the connections
            pos_prob_2d = pos_prob.view(out_height * out_width, -1)
            connections_2d = torch.multinomial(pos_prob_2d, weight_per_out, replacement=True)
            self.connections = connections_2d.to(torch.int32)

            # Initialize the weights with xavier uniform
            bound = 1 / math.sqrt(weight_per_out)
            self.weights = torch.nn.Parameter(torch.empty(out_height * out_width, weight_per_out, device='cuda').uniform_(-bound, bound))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalSparseLinear2d module

        params: input: torch.Tensor: input tensor, shape: (batch_size, in_height, in_width)
        return: torch.Tensor: output tensor, shape: (batch_size, out_height, out_width)
        """
        assert input.dim() == 3 and input.size(1) == self.in_height and input.size(2) == self.in_width, \
            f"Input shape must be (batch_size, {self.in_height}, {self.in_width}), but got {input.shape}"
        assert input.is_cuda, "Input must be a CUDA tensor"
        
        output = PositionalSparseLinear2dFunction.apply(input, self.connections, self.weights)
        return output.view(input.size(0), self.out_height, self.out_width)
    
class PositionalSparseLinear3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, connections: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional sparse linear 3D function

        params: input: torch.Tensor: input tensor, shape: (batch_size, in_depth, in_height, in_width)
        params: connections: torch.Tensor: connections tensor, shape: (out_depth * out_height * out_width, weight_per_out)
        params: weights: torch.Tensor: weights tensor, shape: (out_depth * out_height * out_width, weight_per_out)
        return: torch.Tensor: output tensor, shape: (batch_size, out_depth, out_height, out_width)
        """
        input_flat = input.view(input.size(0), -1)
        output = positional_sparse_linear.positional_sparse_linear_forward(input_flat, connections, weights)
        ctx.save_for_backward(input, connections, weights)
        return output.view(input.size(0), -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the positional sparse linear 3D function

        params: grad_output: torch.Tensor: gradient of the output tensor, shape: (batch_size, out_depth, out_height, out_width)
        return: torch.Tensor: gradient of the input tensor, shape: (batch_size, in_depth, in_height, in_width)
        """
        input, connections, weights = ctx.saved_tensors
        input_flat = input.view(input.size(0), -1)
        grad_output_flat = grad_output.view(grad_output.size(0), -1)
        grad_input_flat, grad_weight = positional_sparse_linear.positional_sparse_linear_backward(
            grad_output_flat, input_flat, weights, connections)
        grad_input = grad_input_flat.view_as(input)
        return grad_input, None, grad_weight

class PositionalSparseLinear3d(torch.nn.Module):
    def __init__(self, in_depth: int, in_height: int, in_width: int, 
                 out_depth: int, out_height: int, out_width: int, 
                 weight_per_out: int = 5, position_influence: float = 0.1, 
                 no_position_influence: bool = False):
        """
        Initialize the PositionalSparseLinear3d module

        params: in_depth: int: number of input depth
        params: in_height: int: number of input height
        params: in_width: int: number of input width
        params: out_depth: int: number of output depth
        params: out_height: int: number of output height
        params: out_width: int: number of output width
        params: weight_per_out: int: number of weights per output
        params: position_influence: float: influence of the position
        params: no_position_influence: bool: whether to use position influence
        """
        super(PositionalSparseLinear3d, self).__init__()
        self.in_depth = in_depth
        self.in_height = in_height
        self.in_width = in_width
        self.out_depth = out_depth
        self.out_height = out_height
        self.out_width = out_width
        self.weight_per_out = weight_per_out

        assert torch.cuda.is_available(), "PositionalSparseLinear3d is only available on CUDA"

        if no_position_influence:
            connections_3d = torch.randint(in_depth * in_height * in_width, 
                                           (out_depth * out_height * out_width, weight_per_out), 
                                           device='cuda')
            self.connections = connections_3d.to(torch.int32)

            bound = 1 / math.sqrt(weight_per_out)
            self.weights = torch.nn.Parameter(torch.empty(out_depth * out_height * out_width, 
                                                          weight_per_out, device='cuda').uniform_(-bound, bound))
        else:
            d_i_pos = torch.arange(in_depth, device='cuda') / in_depth
            h_i_pos = torch.arange(in_height, device='cuda') / in_height
            w_i_pos = torch.arange(in_width, device='cuda') / in_width
            d_o_pos = torch.arange(out_depth, device='cuda') / out_depth
            h_o_pos = torch.arange(out_height, device='cuda') / out_height
            w_o_pos = torch.arange(out_width, device='cuda') / out_width

            d_pos_diff = torch.abs(d_o_pos.unsqueeze(1) - d_i_pos.unsqueeze(0))
            h_pos_diff = torch.abs(h_o_pos.unsqueeze(1) - h_i_pos.unsqueeze(0))
            w_pos_diff = torch.abs(w_o_pos.unsqueeze(1) - w_i_pos.unsqueeze(0))

            pos_diff = torch.sqrt(
                d_pos_diff.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5) ** 2 +
                h_pos_diff.unsqueeze(0).unsqueeze(1).unsqueeze(4).unsqueeze(5) ** 2 +
                w_pos_diff.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3) ** 2
            )
            pos_prob = torch.exp(-pos_diff / position_influence)

            pos_prob_3d = pos_prob.view(out_depth * out_height * out_width, -1)
            connections_3d = torch.multinomial(pos_prob_3d, weight_per_out, replacement=True)
            self.connections = connections_3d.to(torch.int32)

            bound = 1 / math.sqrt(weight_per_out)
            self.weights = torch.nn.Parameter(torch.empty(out_depth * out_height * out_width, 
                                                          weight_per_out, device='cuda').uniform_(-bound, bound))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalSparseLinear3d module

        params: input: torch.Tensor: input tensor, shape: (batch_size, in_depth, in_height, in_width)
        return: torch.Tensor: output tensor, shape: (batch_size, out_depth, out_height, out_width)
        """
        assert input.dim() == 4 and input.size(1) == self.in_depth and input.size(2) == self.in_height and input.size(3) == self.in_width, \
            f"Input shape must be (batch_size, {self.in_depth}, {self.in_height}, {self.in_width}), but got {input.shape}"
        assert input.is_cuda, "Input must be a CUDA tensor"
        
        output = PositionalSparseLinear3dFunction.apply(input, self.connections, self.weights)
        return output.view(input.size(0), self.out_depth, self.out_height, self.out_width)