import os
import subprocess
import nibabel as nib
import numpy as np
from typing import Union

keys_split = {'axial': '-z', 'coronal': '-y', 'saggital': '-x'}

def rotTFimage(filename: Union[nib.Nifti1Image, str], save_path: str) -> nib.Nifti1Image:

    """
    Function to rotate images to be able to segment them using our algorithm
    Returns the image path
    """
    
    if isinstance(filename, nib.Nifti1Image):
        image = filename
    elif (isinstance(filename, str)) and filename.endswith('.nii.gz'):
        image = nib.load(filename)


    data=image.get_fdata()
    affine = image.affine

    # we rotate the data array twice to match our algorithm requirements
    rotated1=np.rot90(data, 3, axes=(0,2))
    rotated2=np.rot90(rotated1, 2)
    
    rotated_img = nib.Nifti1Image(rotated2, affine=affine)

    nib.save(rotated_img, save_path)

    return rotated_img, save_path


def rot_back_to_original_orientation(filename: Union[nib.Nifti1Image, str]) -> np.ndarray:
    if isinstance(filename, nib.Nifti1Image):
        image = filename
    elif (isinstance(filename, str)) and filename.endswith('.nii.gz'):
        image = nib.load(filename)

    data=image.get_fdata()
    affine = image.affine

    rotated1 = np.rot90(data, 2)
    rotated2 = np.rot90(rotated1, 1, axes = (0, 2))
    return rotated2
 

def split(input_nii: str, target_axis) -> str:

    """
    Uses a nii image as a input and creates a folder with the same name where it stores
    the result of splitting it along the 3 axes
    """
    
    output_folder = input_nii.split('.')[0]
    output_filename = output_folder.split('/')[-1]
    
    os.makedirs(f'{output_folder}/{target_axis}', exist_ok=True)

    command = f'fslsplit {input_nii} {output_folder}/{target_axis}/{output_filename}- {keys_split[target_axis]}'
    subprocess.run(command, shell=True)

    return output_folder









    







