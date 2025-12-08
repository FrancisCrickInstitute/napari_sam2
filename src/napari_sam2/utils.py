import textwrap

import torch

from napari.layers import Image

def format_tooltip(text: str, width: int = 70) -> str:
    """
    Function to wrap text in a tooltip to the specified width. Ensures better-looking tooltips.

    Necessary because Qt only automatically wordwraps rich text, which has it's own issues.
    """
    return textwrap.fill(text.strip(), width=width, drop_whitespace=True)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def configure_cuda(device):
    if device.type == "cuda":
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def get_active_dim_size(viewer, image_layer: Image=None):
    # Get the size of the currently active non-displayed dimension (e.g. z or t, based on slider last used) for a particular layer
    # Returns
    # - idx: index of relevant dimension in data array (None if dimension doesn't exist)
    # - size: the size of the relavant dimension in input layer
    last_slider_idx = viewer.dims.last_used
    if image_layer == None:
        # Get currently selected and active layer
        if not len(viewer.layers):
            raise RuntimeError("Cannot get frame dimensions. No layers loaded.")
        image_layer = viewer.layers.selection.active
    # Note that ndim counts spatial dimensions, so and RGB image has ndim=2 but an RGB stack has ndim=3
    if image_layer.ndim < 3:
        return (None, 1)
    
    offset = viewer.dims.ndim - image_layer.ndim
    if (idx := last_slider_idx - offset) < 0:
        # This is an unused (effectively singleton) dimension for this layer
        return (None, 1)
    
    return (idx, image_layer.data.shape[idx])