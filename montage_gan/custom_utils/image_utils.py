import math
import os
import copy

import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid

"""
Naming convention
=================
img_tensor: PyTorch tensor with shape [C,H,W]
img_np: Numpy ndarray with shape [H,W,C]
img_pil: Pillow image

By size, for example:
    blchw: PyTorch tensor with shape [B,L,C,H,W]
    lchw: PyTorch tensor with shape [L,C,H,W]

=================
Notes when testing the Pytorch's affine_grid and gird_sample
    [[1, 0, 1], [0, 1, 1]] translate the image to top-left
    [[1, 0, -1], [0, 1, -1]] translate the image to bottom-right
    affine_grid / grid_sample assume that the input image is [0,1], will cause bug if [-1,1] is given
        See also: https://issueexplorer.com/issue/pytorch/pytorch/66366
    scale_x and scale_y, the scaling is actually inverted, means that the larger the number, the smaller the content
    scale_x = -1 means x_flip, scale_y = -1 means y_flip
"""


def assert_range_zero1(tensor):
    _min, _max = torch.min(tensor), torch.max(tensor)
    assert _min >= 0 and _max <= 1


def load_test_image():
    """
    Load a nextgen image in format: shape CHW, range [0.,1.].
    """
    PATH = "test_data/all.png"
    # Load with Pillow
    img_pil = Image.open(PATH)
    # Convert to tensor [0.,1.]
    img_tensor = ToTensor()(img_pil)
    return img_tensor


def load_test_layers():
    """
    Load a set of image layers in format: shape LCHW, range [0.,1.].
    Layers are sorted according to their filename.
    """
    DIR = "test_data/layers"
    layers = []
    for filename in sorted(os.listdir(DIR)):
        file_path = os.path.join(DIR, filename)
        # Load with Pillow
        img_pil = Image.open(file_path)
        # Convert to tensor [0.,1.]
        img_tensor = ToTensor()(img_pil)
        layers.append(img_tensor)
    layers = torch.stack(layers)
    return layers


def make_dummy_batch(tensor, batch_size=8):
    """
    Make a dummy batch by repeating the given tensor.
    """
    return torch.stack([tensor] * batch_size)


def alpha_composite(blchw_lchw):
    """
    Alpha composite using Pillow.
    Limitation: Not differentiable.
    """
    is_unbatched: bool = blchw_lchw.ndimension() == 4
    device = blchw_lchw.device

    def process(lchw):
        # Convert to PIL Image, ToPILImage will preserve the value range
        img_pil_list = [ToPILImage()(chw) for chw in lchw]
        canvas = img_pil_list[0]
        for img_pil in img_pil_list[1:]:
            canvas.alpha_composite(img_pil)
        # Convert back to Tensor
        return ToTensor()(canvas).to(device)

    if is_unbatched:
        # Output in format: shape CHW, range [0.,1.]
        return process(blchw_lchw)
    else:
        # Output in format: shape BCHW, range [0.,1.]
        return torch.stack([process(lchw) for lchw in blchw_lchw])


try:
    nan_to_num = torch.nan_to_num  # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):  # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


def alpha_composite_pytorch(blchw_lchw, use_premultiplied=False):
    """
    Alpha composite implemented in PyTorch, range [0,1]
    """
    is_unbatched: bool = blchw_lchw.ndimension() == 4

    def premultiplied_rgba(chw):
        color, alpha = chw[:3], chw[3:]
        chw[:3] = color * alpha
        return chw

    def reverse_premultiplied_rgba(chw):
        color, alpha = chw[:3], chw[3:]
        chw[:3] = nan_to_num(color / alpha)
        return chw

    def a_over_b(chw1, chw2):
        color1, alpha1 = chw1[:3], chw1[3:]
        color2, alpha2 = chw2[:3], chw2[3:]
        alpha_out = alpha1 + alpha2 * (1 - alpha1)
        color_out = nan_to_num((color1 * alpha1 + color2 * alpha2 * (1 - alpha1)) / alpha_out)
        return torch.cat([color_out, alpha_out])

    def a_over_b_premultiplied(chw1, chw2):
        color1, alpha1 = chw1[:3], chw1[3:]
        color2, alpha2 = chw2[:3], chw2[3:]
        color_out = color1 + color2 * (1 - alpha1)
        alpha_out = alpha1 + alpha2 * (1 - alpha1)
        return torch.cat([color_out, alpha_out])

    def process(lchw):
        canvas = lchw[0]
        for chw in lchw[1:]:
            canvas = a_over_b(chw, canvas)
        return canvas

    def process_premultiplied(lchw):
        canvas = premultiplied_rgba(lchw[0])
        for chw in lchw[1:]:
            canvas = a_over_b_premultiplied(premultiplied_rgba(chw), canvas)
        return reverse_premultiplied_rgba(canvas)

    if is_unbatched:
        # Output in format: shape CHW, range [0.,1.]
        if use_premultiplied:
            return process_premultiplied(blchw_lchw)
        return process(blchw_lchw)
    else:
        # Output in format: shape BCHW, range [0.,1.]
        if use_premultiplied:
            torch.stack([process_premultiplied(lchw) for lchw in blchw_lchw])
        return torch.stack([process(lchw) for lchw in blchw_lchw])


def show_image(bchw_chw):
    """
    Convert to PIL Image and show.
    Limitation: Cannot show multiple images at once.
    """
    is_unbatched: bool = bchw_chw.ndimension() == 3

    def process(img_tensor):
        img_pil = ToPILImage()(img_tensor)
        img_pil.show()

    if is_unbatched:
        process(bchw_chw)
    else:
        grid = make_grid(bchw_chw, padding=0)
        process(grid)


def normalize_minus11(tensor):
    """
    Shift the image from the range [0,1] to [-1,1].
    """
    return tensor * 2. - 1.


def normalize_zero1(tensor):
    """
    Shift the image from the range [0,1] to [-1,1].
    """
    return (tensor + 1.) / 2.


def bounding_box(mask):
    """
    Similar to openCV boundingRect, implemented with PyTorch.
    :param mask: 2D tensor (mask)
    :return: Rect bounding box (x1, y1, x2, y2)
    """
    y_min, x_min = mask.min(dim=0)[0]
    y_max, x_max = mask.max(dim=0)[0]
    return x_min, y_min, x_max, y_max


def crop_to_content(img_tensor):
    alpha = img_tensor[3]  # RGBA
    nonzero = torch.nonzero(alpha)
    x1, y1, x2, y2 = bounding_box(nonzero) if len(nonzero) != 0 else (0, 0, 0, 0)
    return img_tensor[:, y1:y2, x1:x2]


def pad_256(img_tensor: torch.Tensor, pad_value=0):
    """
    Add constant-padding to image. Content of img_tensor will be placed at the middle.
    Result dimension is 256*256 pixels. Default constant used in padding is 0 (zero-padding).
    If the img_tensor is normalized to [-1,1], the pad_value should be set to -1 for the correct result.
    """
    _, h, w = img_tensor.shape
    pad_x, pad_y = 256 - w, 256 - h
    pad_x1, pad_y1 = pad_x // 2, pad_y // 2
    pad_x2, pad_y2 = pad_x - pad_x1, pad_y - pad_y1
    return F.pad(img_tensor, (pad_x1, pad_x2, pad_y1, pad_y2), mode="constant", value=pad_value)


def make_batch_for_pos_estimator(list_of_bchw, pad_value=0):
    """
    Input: a list of bchw, Output: blchw
    Add a constant-padding to the bchw (using pad_256), then reshape it in format blchw.
    If the input is normalized to [-1,1], the pad_value should be set to -1 for the correct result.
    """

    def process(bchw):
        b, c, h, w = bchw.shape
        if h == 256 and w == 256:
            return bchw
        return torch.stack([pad_256(chw, pad_value) for chw in bchw])

    lbchw = torch.stack([process(bchw) for bchw in list_of_bchw])
    return torch.transpose(lbchw, 0, 1).contiguous()  # Swap l and b dimension


def make_batch_for_local_d(blchw, layer_size_list, to_minus11=False):
    """
    Input: blchw, Output: list of bchw
    Make a batch of real layer images for each local discriminator.
    The content in the layer will be moved to the center,
    and each layer will be cropped into the respective size specified in the layer_size_list.
    The input must be in range: [0,1].
    """
    assert_range_zero1(blchw)
    b, l, c, h, w = blchw.shape
    # TODO debug
    # centered_blchw = generate_pseudo_fake(blchw).contiguous()
    # lbchw = torch.transpose(centered_blchw, 0, 1)
    centered_blchw = generate_pseudo_fake(blchw)
    lbchw = torch.transpose(centered_blchw, 0, 1).contiguous()
    list_of_bchw = []
    for bchw, (base_height, base_width) in zip(lbchw, layer_size_list):
        x1, y1 = (w - base_width) // 2, (h - base_height) // 2
        x2, y2 = (w + base_width) // 2, (h + base_height) // 2
        if to_minus11:
            list_of_bchw.append(normalize_minus11(bchw[..., y1:y2, x1:x2]))
        else:
            list_of_bchw.append(bchw[..., y1:y2, x1:x2])
    return list_of_bchw


def generate_pseudo_fake(blchw):
    """
    Move all contents in each layer to center. Input must be in range: [0,1].
    """
    assert_range_zero1(blchw)
    b, l, c, h, w = blchw.shape
    pseudo_fake = []
    for img_tensor in blchw.view(-1, c, h, w):
        pseudo_fake.append(pad_256(crop_to_content(img_tensor)))
    return torch.stack(pseudo_fake).view(b, l, c, h, w)


def random_position(blchw):
    """
    Move each layer to random position. Input must be in range: [0,1].
    """
    assert_range_zero1(blchw)
    b, l, c, h, w = blchw.shape
    device = blchw.device
    translation_bl2 = torch.empty((b, l, 2), device=device).uniform_(-1., 1.)
    theta = convert_translate_to_2x3(translation_bl2)
    theta = theta.view(-1, 2, 3)  # [B*L,2,3]
    blchw = blchw.view(-1, c, h, w)  # [B*L,C,H,W]
    grid = F.affine_grid(theta, blchw.size(), align_corners=False)
    blchw = F.grid_sample(blchw, grid, align_corners=False)
    return blchw.view(b, l, c, h, w)


def convert_2x3_to_3x3(A):
    # Note: Identical to Kornia's Impl convert_affinematrix_to_homography
    t = F.pad(A, [0, 0, 0, 1], "constant", value=0.)
    t[..., -1, -1] += 1.0
    return t


def combine_transformation(trans):
    # Note: The order of transformations matter
    assert len(trans) >= 2
    trans = [convert_2x3_to_3x3(t) for t in trans]
    A = trans[0]
    for B in trans[1:]:
        A = torch.matmul(A, B)
    # Convert back to 2x3 representation
    A = A[:2, :]
    return A


def convert_translate_to_2x3(translation_bl2_l2):
    """
    Convert translate tensor to affine transformation matrix.
    Translation tensor, format: dx, dy in range [-1, 1].
    """
    is_unbatched: bool = translation_bl2_l2.ndimension() == 2
    device = translation_bl2_l2.device

    def process(translation_l2):
        theta = []
        for trans in translation_l2:
            t = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device)
            t[..., -1] += trans
            theta.append(t)
        return torch.stack(theta)

    if is_unbatched:
        return process(translation_bl2_l2)
    else:
        return torch.stack([process(translation_l2) for translation_l2 in translation_bl2_l2])


def stack_layer_to_channel(tensor_in):
    """
    Convert tensor format blchw -> b[l*c]hw
    """
    b, l, c, h, w = tensor_in.shape
    return torch.reshape(tensor_in, (b, l * c, h, w))


def unstack_layer_to_channel(tensor_in, num_channels=4):
    """
    Convert tensor format blchw <- b[l*c]hw
    """
    b, lc, h, w = tensor_in.shape
    return torch.reshape(tensor_in, (b, lc // num_channels, num_channels, h, w))


def calc_psnr(x, y, data_range=1):
    mse = torch.nn.MSELoss()(x, y)
    return 10 * math.log10(data_range ** 2 / mse.item())


def blend_white_bg(images):
    """
    Blend RGBA images with white background to RGB images
    images: bchw [0,1]
    output: bchw [0,1]
    """
    device = images.device
    white_bg = torch.ones((images.shape[1:]), device=device)  # [4,H,W]
    temp = torch.unsqueeze(images, dim=1)  # [B,1,4,H,W]
    temp = temp.repeat(1, 2, 1, 1, 1)  # [B,2,4,H,W]
    temp[:, 0, :, :, :] = white_bg  # Replace first layer with white bg
    output = alpha_composite(temp)[:, :3, :, :]  # [B,3,H,W] [0,1]
    return output


def test():
    # t = make_dummy_batch(load_test_image().to("cuda"))
    # print(t.shape)
    #
    # t = make_dummy_batch(load_test_layers().to("cuda"))
    # blended = alpha_composite(t)
    # # show_image(blended)
    # print(t.shape)
    #
    # eye = load_test_layers()[4].to("cuda")
    # cropped = crop_to_content(eye)
    # # show_image(pad_256(cropped))
    #
    # pseudo_fake = generate_pseudo_fake(make_dummy_batch(load_test_layers().to("cuda")))
    # blended = alpha_composite(pseudo_fake)
    # # show_image(blended)
    #
    # random = random_position(pseudo_fake)
    # blended = alpha_composite(random)
    # show_image(blended)
    #
    # # print(convert_translate_to_2x3(torch.empty((8, 2), device="cuda").uniform_(-1., 1.)))
    # # print(convert_translate_to_2x3(torch.rand((8, 9, 2), device="cuda").uniform_(-1., 1.)))

    t = make_dummy_batch(load_test_layers())
    # t = []
    # for i in range(1,4):
    #     img = Image.open(f"layer{i}.png")
    #     t.append(ToTensor()(img))
    # # for i in range(1,3):
    # #     img = Image.open(f"test{i}.png")
    # #     t.append(ToTensor()(img))
    # t = torch.stack(t)
    # print(t[:,:, 120, 120])
    blended = alpha_composite_pytorch(t)
    # print(blended[:,120,120])
    from torchvision.utils import save_image
    save_image(blended, "test-blend.png")

    # t = torch.tensor([[0.0000, 0.0000, 1.0000, 0.5020],
    #                   [0.0000, 1.0000, 0.0000, 0.5020],
    #                   [1.0000, 0.0000, 0.0000, 0.5020]])
    # t = torch.unsqueeze(t, dim=2)
    # t = torch.unsqueeze(t, dim=3)
    # print(t.shape)
    # print(t)
    # print(alpha_composite_pytorch(t))


if __name__ == "__main__":
    test()
