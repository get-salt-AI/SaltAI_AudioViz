import torch
import numpy as np
import os
from PIL import Image
import subprocess
import shutil

from .. import logger

def tensor2pil(x):
    return Image.fromarray(np.clip(255. * x.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
def pil2tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)

def mask2pil(x):
    x = 1. - x
    if x.ndim != 3:
        logger.warning(f"Expected a 3D tensor ([N, H, W]). Got {x.ndim} dimensions.")
        x = x.unsqueeze(0) 
    x_np = x.cpu().numpy()
    if x_np.ndim != 3:
        x_np = np.expand_dims(x_np, axis=0) 
    return Image.fromarray(np.clip(255. * x_np[0, :, :], 0, 255).astype(np.uint8), 'L')

def pil2mask(x):
    if x.mode == 'RGB':
        r, g, b = x.split()
        x = Image.fromarray(np.uint8(0.2989 * np.array(r) + 0.5870 * np.array(g) + 0.1140 * np.array(b)), 'L')
    elif x.mode != 'L':
        raise ValueError("Unsupported image mode, expected 'RGB' or 'L', got {}".format(x.mode))
    mask = torch.from_numpy(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)
    return mask

# Resolve FFMPEG, idea borrowed from VideoHelperSuite
# https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True, capture_output=True).stdout.decode("utf-8")
    except Exception as e:
        logger.error(f"Error checking ffmpeg version at {path}: {e}")
        return 0
    score = 0
    # Rough layout of the importance of various features
    simple_criterion = [("libvpx", 20), ("264", 10), ("265", 3), ("svtav1", 5), ("libopus", 1)]
    for criterion in simple_criterion:
        if criterion[0] in version:
            score += criterion[1]
    # Obtain rough compile year from copyright information
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score

def find_ffmpeg():
    ffmpeg_paths = []
    # Attempt to use imageio_ffmpeg if available
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_paths.append(get_ffmpeg_exe())
    except ImportError:
        logger.warning("imageio_ffmpeg is not available, trying system ffmpeg")

    # Check for system ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        ffmpeg_paths.append(system_ffmpeg)

    if not ffmpeg_paths:
        logger.error("No valid ffmpeg found.")
        return None

    # Select the ffmpeg path with the highest suitability score
    ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)
    if ffmpeg_path:
        logger.info(f"Using ffmpeg at {ffmpeg_path}")
    return ffmpeg_path


def path_sjoin(inp, rel):
    """
    Joins the input_dir and relative_path securely ensuring that the
    resulting path is within the input_dir.
    """
    # Normalize the base directory and the intended path
    base = os.path.normpath(inp)
    target = os.path.normpath(os.path.join(base, rel))
    # Check if the target path is within the base directory
    if not target.startswith(base):
        raise ValueError("Attempt to access a file outside the defined input directory!")
    return target


# Exports

ffmpeg_path = find_ffmpeg()