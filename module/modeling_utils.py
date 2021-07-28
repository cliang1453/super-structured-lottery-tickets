from math import sqrt
from torch.nn import Linear

def mask_grad_linear_layer(layer, mask, dim=-1):
    import ipdb; ipdb.set_trace()
    """zeros gradient of certain rows/columns of a linear layer"""
    sizes = [1, 1]
    sizes[dim] = layer.weight.size(dim)
    weight_mask = mask.unsqueeze(dim).repeat(*sizes)
    # Weight
    if layer.weight.grad is not None:
        layer.weight.grad.data.masked_fill_(weight_mask, 0)
        # print(weight_mask)
        # print(layer.weight.grad)
    # Bias
    if layer.bias is not None and layer.bias.grad is not None:
        if dim == 0:
            layer.bias.grad.data.zero_()
        else:
            layer.bias.grad.data.masked_fill_(mask, 0)
