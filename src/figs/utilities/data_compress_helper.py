"""
Helper functions for data synthesis.
"""

import numpy as np

from PIL import Image
from io import BytesIO

def decompress_data(image_dict:dict[str,str|np.ndarray]) -> dict[str,str|np.ndarray]:
    """
    We apply a compression if the images are too large to be saved in the .pt file. This function
    decompresses the images back to their original form.

    Args:
        image_dict:    Dictionary containing image data.

    Returns:
        image_dict:    Dictionary with decompressed images.
    """

    # Check if the image_dict has the key 'images' and if it is not empty
    assert 'rgb' in image_dict, "No images found in data dictionary"
    
    # Check if the image are processed or not. We do this by checking the array
    # order. If the order is (N, C, H, W) then the images are processed. If the
    # order is (N, H, W, C) then the images are unprocessed. We only compress if
    # the images are processed.
    if type(image_dict['rgb'][0]) == bytes:
        prc_imgs = image_dict['rgb']
        raw_imgs = []
        for frame_idx in range(len(prc_imgs)):
            # Process each image here
            imgpil = Image.open(BytesIO(prc_imgs[frame_idx]))
            raw_imgs.append(np.array(imgpil))

        image_dict['rgb'] = np.stack(raw_imgs,axis=0)
    else:
        pass

    return image_dict

def compress_data(Images):
    """
    We apply a compression if the images are too large to be saved in the .pt file. This function
    compresses the images to a smaller size.

    Args:
        Images:    List of dictionaries containing image data.

    Returns:
        Images:    List of dictionaries with compressed images.
    """

    # Check if the Images list is not empty and contains dictionaries with 'images' key
    assert 'rgb' in Images[0], "No images found in data dictionary"

    for image_dict in Images:
        # Check if the image are processed or not. We do this by checking the array
        # order. If the order is (N, C, H, W) then the images are processed. If the
        # order is (N, H, W, C) then the images are unprocessed. We only compress if
        # the images are processed.

        if image_dict['rgb'].shape[-1] == 3:
            raw_imgs = image_dict['rgb']
            prc_imgs = []
            for frame_idx in range(raw_imgs.shape[0]):
                img_arr = raw_imgs[frame_idx]
                imgpil = Image.fromarray(img_arr)

                buffer = BytesIO()
                imgpil.save(buffer, format='PNG')   
                buffer.seek(0)
                prc_imgs.append(buffer.getvalue())

            image_dict['rgb'] = prc_imgs
        else:
            pass
    
    return Images