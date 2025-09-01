import cv2
import numpy as np
from PIL import Image

# To set zoom intensity 
zoom_factor = 2

# 90st Waiting Area
WA_x = 250
WA_y = 50

# Management Area
MA_x = 150 
MA_y = 100

def zoom_crop(img_np):
    """
    Zoom into a manually defined region in the image.
    Input: img_np = OpenCV image (NumPy array, BGR)
    Returns: zoomed image as NumPy array (still in BGR for OpenCV)
    """
    # Convert BGR (OpenCV) to RGB (PIL expects RGB)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    img = Image.fromarray(img_rgb)

    w, h = img.size
    # ==================================

    crop_w = w // zoom_factor
    crop_h = h // zoom_factor

    end_x = min(WA_x + crop_w, w)
    end_y = min(WA_y + crop_h, h)

    box = (WA_x, WA_y, end_x, end_y)
    cropped = img.crop(box)

    # Resize back to original size
    zoomed = cropped.resize((w, h), Image.LANCZOS)

    # Convert back to NumPy (RGB)
    zoomed_np = np.array(zoomed)

    # Convert RGB back to BGR for OpenCV
    zoomed_bgr = cv2.cvtColor(zoomed_np, cv2.COLOR_RGB2BGR)
    return zoomed_bgr
