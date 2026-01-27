"""
Image preprocessing utilities for VLA policies.

This module provides common image processing utilities that all VLA policy
implementations can use for consistent image handling across the codebase.

Consolidated from:
- OpenVLA robot_rlds_dataset._process_image: PIL-based processing with augmentation
- Pi0 data_conversion._process_image: NumPy-based processing
"""

from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image


ArrayLike = Union[np.ndarray, 'torch.Tensor']


def process_image(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    augment: bool = False,
    crop_ratio: float = 0.9,
    normalize: bool = True,
    to_chw: bool = True,
) -> np.ndarray:
    """
    Process image with resize, crop, and normalization.
    
    Standard image processing pipeline for VLA models:
    1. Convert to PIL Image
    2. Apply random or center crop
    3. Resize to target size
    4. Normalize to [0, 1]
    5. Convert to CHW format
    
    Args:
        image: Input image as numpy array (H, W, C) in [0, 255] or PIL Image
        target_size: Target size as (height, width)
        augment: If True, apply random crop; otherwise center crop
        crop_ratio: Ratio of image to keep when cropping (0.0 to 1.0)
        normalize: If True, normalize pixel values to [0, 1]
        to_chw: If True, convert from HWC to CHW format
        
    Returns:
        Processed image as numpy array:
            - Shape (C, H, W) if to_chw else (H, W, C)
            - Values in [0, 1] if normalize else [0, 255]
            
    Example:
        >>> img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> processed = process_image(img, target_size=(224, 224), augment=False)
        >>> processed.shape
        (3, 224, 224)
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype(np.uint8))
    else:
        image_pil = image
    
    # Apply crop
    w, h = image_pil.size
    if crop_ratio < 1.0:
        if augment:
            image_pil = random_crop(image_pil, crop_ratio)
        else:
            image_pil = center_crop(image_pil, crop_ratio)
    
    # Resize to target size (height, width)
    image_pil = image_pil.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Convert to numpy array
    image_np = np.array(image_pil, dtype=np.float32)
    
    # Normalize to [0, 1]
    if normalize:
        image_np = image_np / 255.0
    
    # Convert to CHW format
    if to_chw:
        image_np = hwc_to_chw(image_np)
    
    return image_np


def random_crop(
    image: Union[np.ndarray, Image.Image],
    crop_ratio: float = 0.9,
    seed: Optional[int] = None,
) -> Union[np.ndarray, Image.Image]:
    """
    Apply random crop to image.
    
    Args:
        image: Input image as numpy array (H, W, C) or PIL Image
        crop_ratio: Ratio of image to keep (0.0 to 1.0)
        seed: Optional random seed for reproducibility
        
    Returns:
        Cropped image (same type as input)
    """
    if seed is not None:
        np.random.seed(seed)
    
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        w, h = image.size
    else:
        h, w = image.shape[:2]
    
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    
    left = np.random.randint(0, w - crop_w + 1)
    top = np.random.randint(0, h - crop_h + 1)
    
    if is_pil:
        return image.crop((left, top, left + crop_w, top + crop_h))
    else:
        return image[top:top + crop_h, left:left + crop_w]


def center_crop(
    image: Union[np.ndarray, Image.Image],
    crop_ratio: float = 0.9,
) -> Union[np.ndarray, Image.Image]:
    """
    Apply center crop to image.
    
    Args:
        image: Input image as numpy array (H, W, C) or PIL Image
        crop_ratio: Ratio of image to keep (0.0 to 1.0)
        
    Returns:
        Cropped image (same type as input)
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        w, h = image.size
    else:
        h, w = image.shape[:2]
    
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    
    if is_pil:
        return image.crop((left, top, left + crop_w, top + crop_h))
    else:
        return image[top:top + crop_h, left:left + crop_w]


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """
    Convert image from HWC (Height, Width, Channels) to CHW format.
    
    Args:
        image: Input image array (H, W, C)
        
    Returns:
        Image array in CHW format (C, H, W)
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {image.shape}")
    return np.transpose(image, (2, 0, 1))


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    """
    Convert image from CHW (Channels, Height, Width) to HWC format.
    
    Args:
        image: Input image array (C, H, W)
        
    Returns:
        Image array in HWC format (H, W, C)
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {image.shape}")
    return np.transpose(image, (1, 2, 0))


def normalize_image(
    image: np.ndarray,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Normalize image pixel values.
    
    If mean/std not provided, simply scales [0, 255] to [0, 1].
    If mean/std provided, applies ImageNet-style normalization.
    
    Args:
        image: Input image array
        mean: Per-channel mean values (default: scale to [0, 1])
        std: Per-channel standard deviation values
        
    Returns:
        Normalized image array
    """
    image = image.astype(np.float32)
    
    if image.max() > 1.0:
        image = image / 255.0
    
    if mean is not None and std is not None:
        mean = np.array(mean).reshape(-1, 1, 1) if image.ndim == 3 and image.shape[0] == 3 else np.array(mean)
        std = np.array(std).reshape(-1, 1, 1) if image.ndim == 3 and image.shape[0] == 3 else np.array(std)
        image = (image - mean) / std
    
    return image


def denormalize_image(
    image: np.ndarray,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Denormalize image pixel values.
    
    Reverses the normalization applied by normalize_image.
    
    Args:
        image: Normalized image array
        mean: Per-channel mean values used in normalization
        std: Per-channel standard deviation values used in normalization
        
    Returns:
        Denormalized image array with values in [0, 1] or [0, 255]
    """
    image = image.astype(np.float32)
    
    if mean is not None and std is not None:
        mean = np.array(mean).reshape(-1, 1, 1) if image.ndim == 3 and image.shape[0] == 3 else np.array(mean)
        std = np.array(std).reshape(-1, 1, 1) if image.ndim == 3 and image.shape[0] == 3 else np.array(std)
        image = image * std + mean
    
    return image


def resize_image(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int],
    interpolation: str = 'bilinear',
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image (H, W, C) or PIL Image
        target_size: Target size as (height, width)
        interpolation: Interpolation method ('bilinear', 'nearest', 'bicubic')
        
    Returns:
        Resized image as numpy array (H, W, C)
    """
    interp_map = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
    }
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    resample = interp_map.get(interpolation, Image.BILINEAR)
    resized = image.resize((target_size[1], target_size[0]), resample)
    
    return np.array(resized)


def stack_images(
    images: dict,
    view_order: Optional[list] = None,
) -> np.ndarray:
    """
    Stack multiple camera view images into a single array.
    
    Args:
        images: Dictionary mapping view names to image arrays
        view_order: Optional list specifying order of views
        
    Returns:
        Stacked images with shape (num_views, C, H, W) or (num_views, H, W, C)
    """
    if view_order is None:
        view_order = sorted(images.keys())
    
    return np.stack([images[view] for view in view_order], axis=0)


def unstack_images(
    stacked: np.ndarray,
    view_names: list,
) -> dict:
    """
    Unstack images into a dictionary of views.
    
    Args:
        stacked: Stacked images with shape (num_views, ...)
        view_names: List of view names
        
    Returns:
        Dictionary mapping view names to image arrays
    """
    return {name: stacked[i] for i, name in enumerate(view_names)}

