import torch
from torchsummary import summary
import torch.nn as nn

from UNetOG import UNet_2
from AttentionUNet import AttentionUNet
from AttentionUNet3D import AttentionUNet3D


def print_layer_ip_op(model):
    # Dictionary to store the shapes
    layer_io_shapes = {}

    def register_hooks(model):
        def hook_fn(module, input, output):
            layer_name = f"{module.__class__.__name__}_{id(module)}"
            input_shape = (
                input[0].shape if isinstance(input, (list, tuple)) else input.shape
            )
            output_shape = (
                output.shape
                if isinstance(output, torch.Tensor)
                else [o.shape for o in output]
            )
            layer_io_shapes[layer_name] = {
                "input": tuple(input_shape),
                "output": (
                    tuple(output_shape)
                    if isinstance(output_shape, torch.Size)
                    else output_shape
                ),
            }

        for name, module in model.named_modules():
            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not module == model
            ):
                module.register_forward_hook(hook_fn)

    # Instantiate your model and register hooks
    # model = AttentionUNet(c_in=3, c_out=1)  # Example input/output channels
    register_hooks(model)

    # Run a forward pass with dummy input
    dummy_input = torch.randn(1, 3, 256, 256)  # Adjust size as needed
    _ = model(dummy_input)

    # Print layer input and output shapes
    for name, shapes in layer_io_shapes.items():
        print(f"{name}:")
        print(f"  Input: {shapes['input']}")
        print(f"  Output: {shapes['output']}")


if __name__ == "__main__":
    model = AttentionUNet(c_in=3, c_out=1)
    ip = torch.randn(1, 3, 32, 32)

    print(ip.size())

    op = model(ip)

    print(op.size())
