"""
Image processing utilities for dataset conversion.

Common image processing functions used across data converters.
"""

import numpy as np
from typing import Tuple, Optional


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Process image to ensure correct format (HWC, uint8).
    
    Handles conversion from CHW to HWC and ensures uint8 dtype.
    
    Args:
        image: Input image array (can be CHW or HWC, float or uint8)
        
    Returns:
        Processed image in HWC uint8 format
    """
    # Ensure image is in HWC format (H, W, C)
    if len(image.shape) == 3 and image.shape[0] == 3:
        # Convert from CHW to HWC
        image = np.transpose(image, (1, 2, 0))
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    return image


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """
    Convert image from HWC to CHW format.
    
    Args:
        image: Image in HWC format (H, W, C)
        
    Returns:
        Image in CHW format (C, H, W)
    """
    if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
        return np.transpose(image, (2, 0, 1))
    return image


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    """
    Convert image from CHW to HWC format.
    
    Args:
        image: Image in CHW format (C, H, W)
        
    Returns:
        Image in HWC format (H, W, C)
    """
    if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
        return np.transpose(image, (1, 2, 0))
    return image


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image (HWC format)
        target_size: Target size (height, width)
        interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic')
        
    Returns:
        Resized image
    """
    try:
        import cv2
        
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
        }
        interp = interp_map.get(interpolation, cv2.INTER_LINEAR)
        
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=interp)
    except ImportError:
        from PIL import Image
        
        interp_map = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
        }
        interp = interp_map.get(interpolation, Image.BILINEAR)
        
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((target_size[1], target_size[0]), interp)
        return np.array(pil_image)


def center_crop(
    image: np.ndarray,
    crop_size: Tuple[int, int]
) -> np.ndarray:
    """
    Center crop image to specified size.
    
    Args:
        image: Input image (HWC format)
        crop_size: Target crop size (height, width)
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return image[start_h:start_h + crop_h, start_w:start_w + crop_w]


def normalize_image(
    image: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Normalize image to [0, 1] or using mean/std.
    
    Args:
        image: Input image (uint8 or float)
        mean: Optional mean for normalization
        std: Optional std for normalization
        
    Returns:
        Normalized image as float32
    """
    # Convert to float
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Apply mean/std normalization if provided
    if mean is not None and std is not None:
        image = (image - mean) / std
    
    return image.astype(np.float32)


def denormalize_image(
    image: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Denormalize image back to [0, 255] uint8.
    
    Args:
        image: Normalized image (float)
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized image as uint8
    """
    # Reverse mean/std normalization if provided
    if mean is not None and std is not None:
        image = image * std + mean
    
    # Convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image

