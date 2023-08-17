import nibabel as nib
import sys
import os
import numpy as np
import shutil
import subprocess
from typing import Union

def turn3dto2d(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    image: str or array. Preferred (because of the bloody affine) is the str image path (although I'm aware of the lower 
    performance)
    """
    if isinstance(image, str):
        nibimage = nib.load(image)
        data_array = nibimage.get_fdata()
        affine = nibimage.affine 
        # aquí deixo el affine, de moment no el necessito (ara si que el necessito)
    elif isinstance(image, np.ndarray):
        data_array = image
        affine = np.eye(4)
    else:
        raise Exception("Non valid image type")
    assert data_array.ndim == 3, f'Image {image} is ndim {data_array.ndim}, not converting it'

    oldshape = np.array(data_array.shape)
    newshape = np.delete(oldshape, np.where(oldshape == 1))
    new_array= data_array.reshape(newshape[0], newshape[1])


    return new_array, affine

def turn2dto3dcoronal(image: Union[str, np.ndarray]) -> np.ndarray:

    if isinstance(image, str):
        nibimage = nib.load(image)
        data_array = nibimage.get_fdata()
    elif isinstance(image, np.ndarray):
        data_array = image
    else: raise Exception("Non valid image type")
    assert data_array.ndim == 2, f'Image {image} is ndim {data_array.ndim}'
    new_array= data_array.reshape(data_array.shape[0], 1, data_array.shape[1])

    return new_array

def turn2dto3dsaggital(image: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image, str):
        nibimage = nib.load(image)
        data_array = nibimage.get_fdata()
    elif isinstance(image, np.ndarray):
        data_array = image
    else: raise Exception("Non valid image type")
    
    assert data_array.ndim == 2, f'Image {image} is ndim {data_array.ndim}'
    new_array= data_array.reshape(1, data_array.shape[0], data_array.shape[1])

    return new_array


def reshape(folder, func) -> None:

    if 'reshapen' not in os.listdir(folder):
        os.mkdir(os.path.join(folder, 'reshapen'))

    files = [i for i in os.listdir(folder) if i.endswith('.nii.gz')]
    for file in files:
        try:
            new_image_array, affine = func(os.path.join(folder, file))
            new_image = nib.Nifti1Image(new_image_array, affine) # canvio la affine per a provar si afecta el performance
            
            nib.save(new_image, os.path.join(folder, 'reshapen', file))
        except AssertionError as e: 
            print(e)
            continue

        subprocess.run('mv' + f' {os.path.join(folder, "reshapen", file)}' + f' {os.path.join(folder, file)}', shell = True)
    
    subprocess.run('rmdir' + f' {os.path.join(folder, "reshapen")}', shell = True)


if __name__ == "__main__":
    folder = sys.argv[1]
    if sys.argv[2] == "3dto2d":
        reshape(folder, func = turn3dto2d)
    elif sys.argv[2] == "2dto3dsaggital":
        reshape(folder, func = turn2dto3dsaggital)
    elif sys.argv[2] == "2dto3dcoronal":
        reshape(folder, func = turn2dto3dcoronal)
    else:
        print("Invalid configuration for reshaping!")