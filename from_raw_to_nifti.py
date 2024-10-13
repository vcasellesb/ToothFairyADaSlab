import os
import numpy as np
import SimpleITK as sitk

def process_one(input_path: str) -> sitk.Image:
    input_array = np.load(input_path)
    image = sitk.GetImageFromArray(input_array)        
    return image

def process_all(input_images: list[str], save_path: str) -> None:

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image in input_images:
        sitk_image = process_one(image)
        sitk_image.SetOrigin((0, 0, 0))
        sitk_image.SetSpacing((1, 1, 1))
        sitk.WriteImage(
            image=sitk_image, 
            fileName=os.path.join(save_path, os.path.basename(os.path.dirname(image)) + '.mha')
        )


if __name__ == "__main__":
    raw_path = 'raw/'
    to_pprocess = [os.path.join(raw_path, f, 'data.npy') for f in os.listdir(raw_path) if f.startswith('P')]
    test_path = 'test/images/cbct'
    process_all(input_images=to_pprocess, save_path=test_path)