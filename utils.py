import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import join, load_json
import subprocess
import sys
import re
from typing import Tuple
from preprocess import rot_back_to_original_orientation, rotTFimage
from shutil import rmtree, move

def sitk2nii(sitk_image: sitk.Image, wheretosave: str) -> Tuple[nib.Nifti1Image, str]:
    """
    Reads sitk image and returns and saves a nii image
    """

    if not os.path.exists(wheretosave):
        os.makedirs(wheretosave, exist_ok=True)

    image_array=sitk.GetArrayFromImage(sitk_image)
    new_file_name=join(wheretosave, f'image.nii.gz')
    nib_image = nib.Nifti1Image(image_array, np.eye(4))
    nib_image = rotTFimage(nib_image)
    nib.save(nib_image, new_file_name)
    return nib_image, new_file_name

def nii2sitk(input_image: str):
    """
    Reads a Nifti file and saves it back to .mha extension
    """
    image_array = rot_back_to_original_orientation(input_image)
    image_array = image_array.astype(np.uint8)
    sitk_img = sitk.GetImageFromArray(image_array)
    return sitk_img

def find_targets(input_path):
    """
    This function should use the segmentation algorithm class to find the input images
    """
    mhafiles = [join(input_path, f) for f in os.listdir(input_path) if 
                (f.endswith('.tif') or f.endswith('.mha')) and "normalized" not in f]

    return mhafiles

def add_case_id(file: str) -> None:
    """
    Adds the _0000 identifier to the images (Required by the nnUNet predictor, don't look at me)
    """
    command = f'mv {file} {file.replace(".nii.gz", "_0000.nii.gz")}'
    subprocess.run(command, shell = True)
    return None

def check_path(path):
    files = []
    for root, _, files in os.walk(path):
        for file in files:
            files.append(join(root, file))

    return files

def check_image_format(image):
    array = image.get_fdata()
    return array.dtype

def load_trpatients(jsonfile='/Users/vicentcaselles/work/research/ChallengesMICAI/Dataset_final/splits.json'):
    patientsjson = load_json(jsonfile)
    trpatients = patientsjson['train']
    return trpatients

def getpatientid(patient):
    """Returns the patient ID (according to the nnUNet naming convention. Won't work if the image is not
    in that format)"""

    return int(re.findall(r'\d+', patient)[0])

def cleanup(final_image_name: str) -> None:
    targets_to_remove = [final_image_name.replace('.mha', '.nii.gz'), 
                         final_image_name.replace('_finalissim.mha', '_final.nii.gz')]
    directory_to_remove = final_image_name.split('_finalissim.mha')[0]
    os.remove(targets_to_remove[0])
    os.remove(targets_to_remove[1])
    rmtree(directory_to_remove)

    move(final_image_name, final_image_name.replace('_normalized_finalissim.mha', '.mha'))

    sitk_final_image = sitk.ReadImage(final_image_name.replace('_normalized_finalissim.mha', '.mha'))

    sitk_array = sitk.GetArrayFromImage(sitk_final_image)

    sitk_back_to_image = sitk.GetImageFromArray(sitk_array)
    return sitk_back_to_image

def get_volume_image(image: sitk.Image)->int:
    return np.count_nonzero(image)

def checkup(mask_sitk: sitk.Image, input_sitk) -> None:
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    input_array = sitk.GetArrayFromImage(input_sitk)
    assert get_volume_image(mask_array) > 10, "Empty mask!"
    assert mask_array.shape == input_array.shape, "Different shape between input image and label"

if __name__ == "__main__":
    # nii_filename = sys.argv[1]
    # image_nib = nib.load(nii_filename)

    # sitk_image = nii2sitk(image_nib)
    # sitk.WriteImage(sitk_image, sys.argv[2])

    sitk_filename = sys.argv[1]
    image_sitk = sitk.ReadImage(sitk_filename)
    sitk2nii(image_sitk, wheretosave='.')