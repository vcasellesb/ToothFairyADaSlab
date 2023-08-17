import numpy as np
import nibabel as nib
import sys
from batchgenerators.utilities.file_and_folder_operations import load_json

def get_min_target(image: np.array):
    return np.min(image)

def get_max_target(image: np.array):
    return np.max(image)

def obtain_minmax_input_image(image_array: np.array):
    return np.min(image_array), np.max(image_array)

def obtain_minmax_dataset(list_of_images: list):
    mins = []
    maxs = []
    for image in list_of_images:
        loaded_image = nib.load(image)
        array_image = loaded_image.get_fdata()
        mins.append(get_min_target(array_image))
        maxs.append(get_max_target(array_image))
    return np.median(mins), np.median(maxs)

def normalize(image_array_to_normalize: np.array) -> np.ndarray:
    # mean_intensity = dataset_properties['mean']
    # std_intensity  = dataset_properties['std']
    lower_bound = 0
    upper_bound = 2048
    # max_dataset = dataset_properties['max']
    # min_dataset = dataset_properties['min']
    input_image_min, input_image_max = obtain_minmax_input_image(image_array_to_normalize)
    # min_image_to_normalize, max_image_to_normalize = obtain_minmax_input_image(image_array_to_normalize)
    normalized_image_array = (((image_array_to_normalize - input_image_min)/(input_image_max - input_image_min)) 
                              * (upper_bound - lower_bound)) + lower_bound
    
    normalized_image_array = np.clip(normalized_image_array, lower_bound, upper_bound)
    return normalized_image_array

def convert_image(image, image_name):
    
    if isinstance(image, nib.Nifti1Image):
        image_array = image.get_fdata()
        affine = image.affine

    elif isinstance(image, np.ndarray):
        image_array = image

    normalized_image = normalize(image_array)
    normalized_nifti = nib.Nifti1Image(normalized_image, np.eye(4))
    
    normalized_filename =  image_name.replace('.nii.gz', '_normalized.nii.gz')
    nib.save(normalized_nifti, normalized_filename)
    return normalized_image, normalized_filename


if __name__ == "__main__":
    fingerprint = load_json('models/dataset_fingerprint.json')
    dataset_properties = fingerprint['general_intensity_properties_per_channel']["0"]
    convert_image(sys.argv[1], dataset_properties=dataset_properties)
