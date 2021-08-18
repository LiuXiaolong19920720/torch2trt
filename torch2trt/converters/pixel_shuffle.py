import tensorrt as trt
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.pixel_shuffle')
def convert_pixel_shuffle(ctx):
    input = ctx.method_args[0]
    upscale_factor = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    # inshape = tuple(input.shape)[1:]  # exclude batch

    print("【convert_pixel_shuffle - input shape】: ", tuple(input.shape))

    in_channels, in_height, in_width = tuple(input.shape)[1:]
    out_channels = int(1.0 * in_channels / (1.0 * upscale_factor * upscale_factor))
    out_height = int(in_height * upscale_factor)
    out_width = int(in_width * upscale_factor)

    dims1 = [out_channels, upscale_factor, upscale_factor, in_height, in_width]

    ps1 = ctx.network.add_shuffle(input_trt)
    ps1.reshape_dims = dims1

    # ps1 = add_reshape(network, input, dims1)

    ps2 = ctx.network.add_shuffle(ps1.get_output(0))
    ps2.first_transpose = trt.Permutation([0, 3, 1, 4, 2])
    ps2.reshape_dims = [out_channels, out_height, out_width]

    output._trt = ps2.get_output(0)
