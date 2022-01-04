"""
A script to search for Conv parameters based on the given size of input and output
"""
import math
import itertools


def calc_output_size(input_size, padding, dilation, kernel_size, stride):
    return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def calc_output_size_convTranspose(input_size, padding, dilation, kernel_size, stride):
    output_padding = 1
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


input_size = 256 * 3
output_size = 256 * 3
padding_list = range(0, 8)
dilation_list = range(1, 8)
kernel_size_list = [3, 5, 7]
stride = range(1, 8)

for padding, dilation, kernel_size, stride in itertools.product(padding_list, dilation_list, kernel_size_list, stride):
    if output_size == calc_output_size(input_size, padding, dilation, kernel_size, stride):
        print(f"[Conv] Padding: {padding}, Dilation: {dilation}, Kernel Size: {kernel_size}, Stride: {stride}")
    if output_size == calc_output_size_convTranspose(input_size, padding, dilation, kernel_size, stride):
        print(f"[ConvTranspose] Padding: {padding}, Dilation: {dilation}, Kernel Size: {kernel_size}, Stride: {stride}")
