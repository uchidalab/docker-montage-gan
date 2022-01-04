"""
Keep for achieve purpose. Not longer used.
"""

# from typing import Tuple
#
# import torch
# from kornia.geometry.transform import warp_affine
#
# """
# Modified from Kornia. Added dsize to translate and affine.
# """
#
#
# def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
#     """Computes affine matrix for translation."""
#     matrix: torch.Tensor = torch.eye(
#         3, device=translation.device, dtype=translation.dtype)
#     matrix = matrix.repeat(translation.shape[0], 1, 1)
#
#     dx, dy = torch.chunk(translation, chunks=2, dim=-1)
#     matrix[..., 0, 2:3] += dx
#     matrix[..., 1, 2:3] += dy
#     return matrix
#
#
# def kornia_affine(tensor: torch.Tensor, matrix: torch.Tensor, dsize: Tuple[int, int], mode: str = 'bilinear',
#                   align_corners: bool = False) -> torch.Tensor:
#     r"""Apply an affine transformation to the image.
#
#     Args:
#         tensor (torch.Tensor): The image tensor to be warped in shapes of
#             :math:`(H, W)`, :math:`(D, H, W)` and :math:`(B, C, H, W)`.
#         matrix (torch.Tensor): The 2x3 affine transformation matrix.
#         mode (str): 'bilinear' | 'nearest'
#         align_corners(bool): interpolation flag. Default: False. See
#         https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
#
#     Returns:
#         torch.Tensor: The warped image.
#     """
#     # warping needs data in the shape of BCHW
#     is_unbatched: bool = tensor.ndimension() == 3
#     if is_unbatched:
#         tensor = torch.unsqueeze(tensor, dim=0)
#
#     # we enforce broadcasting since by default grid_sample it does not
#     # give support for that
#     matrix = matrix.expand(tensor.shape[0], -1, -1)
#
#     # warp the input tensor
#     warped: torch.Tensor = warp_affine(tensor, matrix, dsize, mode,
#                                        align_corners=align_corners)
#
#     # return in the original shape
#     if is_unbatched:
#         warped = torch.squeeze(warped, dim=0)
#
#     return warped
#
#
# def kornia_translate(tensor: torch.Tensor, translation: torch.Tensor, dsize: Tuple[int, int],
#                      align_corners: bool = False) -> torch.Tensor:
#     r"""Translate the tensor in pixel units.
#
#     See :class:`~kornia.Translate` for details.
#     """
#     if not torch.is_tensor(tensor):
#         raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
#                         .format(type(tensor)))
#     if not torch.is_tensor(translation):
#         raise TypeError("Input translation type is not a torch.Tensor. Got {}"
#                         .format(type(translation)))
#     if len(tensor.shape) not in (3, 4,):
#         raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
#                          "Got: {}".format(tensor.shape))
#
#     # compute the translation matrix
#     translation_matrix: torch.Tensor = _compute_translation_matrix(translation)
#
#     # warp using the affine transform
#     return kornia_affine(tensor, translation_matrix[..., :2, :3], dsize=dsize, align_corners=align_corners)
