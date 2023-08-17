import subprocess
import os
import numpy as np
import nibabel as nib
from reshape_arrays import turn3dto2d, turn2dto3dcoronal, turn2dto3dsaggital
from batchgenerators.utilities.file_and_folder_operations import join
import sys


## This is just a few parametres that are used during the post processing
keys_merge = {'axial': '-z ', 'saggital': '-x', 'coronal': '-y'}
keys_reshaping = {'saggital': turn2dto3dsaggital, 'coronal': turn2dto3dcoronal}
# keys_seg_maths = {'coronal': 0.4, 'axial': 0.97, 'saggital': 0.7}
# keys_seg_maths = {'coronal': 0.4, 'axial': 0.4, 'saggital': 0.4}

def probabilities_array_3d_to_2d(np_array_probabilities: np.ndarray) -> np.ndarray:
    """
    Turns a probability array in 3d to 2d
    """
    array_probabilities_nerve_label = np_array_probabilities['probabilities'][1]

    probability_array_in_2d, _ = turn3dto2d(array_probabilities_nerve_label)
    transposed_array_in_2d = np.transpose(probability_array_in_2d)

    return transposed_array_in_2d

def turn_probabilities_niigz_back_to_3d(image_name: str, axis: str) -> str:

    if axis == "axial":
        # axial axis requires an special treatment, since fslsplit returns 2d images. Easier processing
        array = np.load(image_name)

        return np.transpose(array['probabilities'][1])
    
    probabilties_np = np.load(image_name)
    
    probability_niigz_in_2d = probabilities_array_3d_to_2d(probabilties_np)

    functiontouse = keys_reshaping[axis]

    probability_niigz_to_3d = functiontouse(probability_niigz_in_2d)

    return probability_niigz_to_3d

def process_all_probabilty_arrays(image_directory: str, axis: str) -> list:

    probability_npzs = [join(image_directory,f) for f in os.listdir(image_directory) if f.endswith('.npz')]

    probability_npzs_in_3d = []

    for prob_npz in probability_npzs:

        probabilty_niigz_in_3d = turn_probabilities_niigz_back_to_3d(prob_npz, axis)
        probability_nifti_image_in_3d = nib.Nifti1Image(probabilty_niigz_in_3d, np.eye(4))
        nib.save(probability_nifti_image_in_3d, prob_npz.replace('.npz', '_probability3d.nii.gz'))

        probability_npzs_in_3d.append(prob_npz.replace('.npz', '_probability3d.nii.gz'))

    return probability_npzs_in_3d

def align_input_slice_with_predicted_array(prob_niftis_in_3d: list, image_name, axis, input_images_dir) -> str:

    for nifti in prob_niftis_in_3d:
        nifti_without_path = os.path.basename(nifti)
        input_image = join(input_images_dir, image_name, 'image_normalized', axis,
                           nifti_without_path.replace('_probability3d.nii.gz', '_0000.nii.gz'))
        
        command = (f'fslcpgeom {input_image} {nifti} -d')

        subprocess.run(command, shell = True)
    
    return prob_niftis_in_3d
        

def merge_all_3d_probability_images(prob_niftis_in_3d: list, conf, image_name) -> str:

    dir_to_save_merged_image = os.path.dirname(prob_niftis_in_3d[0])

    prob_niftis_in_3d = sorted(prob_niftis_in_3d)

    command = f'fslmerge {keys_merge[conf]} {join(dir_to_save_merged_image,image_name)}_prob_merged.nii.gz {" ".join(prob_niftis_in_3d)}'

    subprocess.run(command, shell = True)

    return f'{join(dir_to_save_merged_image,image_name)}_prob_merged.nii.gz'


def seg_maths_thr_per_axis(probability_3d_merged_image: str, axis: str) -> str:

    command = f'seg_maths {probability_3d_merged_image} {probability_3d_merged_image.replace("_prob_merged.nii.gz", "_prob_thr.nii.gz")}'

    subprocess.run(command, shell=True)

    return f'{probability_3d_merged_image.replace("_prob_merged.nii.gz", "_prob_thr.nii.gz")}'


def align_input_image_affine_mask_affine_per_axis(input_image: str, mask_axis: str) -> str:

    command = f'fslcpgeom {input_image} {mask_axis} -d'

    subprocess.run(command, shell = True)
        
    return mask_axis


def seg_maths_add_all_thr_masks(list_of_thr_masks: list, image_name: str) -> str:

    assert len(list_of_thr_masks) == 3, f"There were more masks than there should be. I found {len(list_of_thr_masks)}"

    command = f'seg_maths {list_of_thr_masks[0]} -add {list_of_thr_masks[1]} -add {list_of_thr_masks[2]} -thr 1.1 -bin {image_name}_final.nii.gz'

    subprocess.run(command, shell=True)

    return f'{image_name}_final.nii.gz'


def seg_maths_dil_ero(final_mask: str) -> str:

    command = f'seg_maths {final_mask} -dil 2 -ero 2 {final_mask.replace("_final.nii.gz", "_finalissim.nii.gz")}'

    subprocess.run(command, shell=True)

    return f'{final_mask.replace("_final.nii.gz", "_finalissim.nii.gz")}'


def merge_all_3d_images(dir_with_images: str, conf, image_name) -> str:

    subprocess.run('fslmerge' + f' {keys_merge[conf]}' + f' {dir_with_images+image_name}_merged.nii.gz' + f' {dir_with_images + "/*.nii.gz"}', shell=True)

    return f'{dir_with_images+image_name}_merged.nii.gz'

def check_affine_is_npeye(affine: np.ndarray) -> bool:
    return np.all(affine == np.eye(4))

def check_affines_between_mask_and_input_are_aligned(affine_image: np.ndarray, affine_mask: np.ndarray) -> bool:
    return np.all(affine_image == affine_mask)

def align_affines_between_mask_and_input(image_path: str, mask_path: str) -> np.ndarray:
    image = nib.load(image_path)
    affine_image = image.affine
    mask = nib.load(mask_path)
    affine_mask = mask.affine

    image_affine_is_eye = check_affine_is_npeye(affine_image)
    mask_affine_is_eye = check_affine_is_npeye(affine_mask)
    are_they_aligned = check_affines_between_mask_and_input_are_aligned(affine_image=affine_image, affine_mask=affine_mask)


    assert image_affine_is_eye, "Image affine should be an identity matrix!!!"
    
    if not mask_affine_is_eye:
        subprocess.run('fslcpgeom' + f' {image_path}' + f' {mask_path}', shell=True)
        return mask_path

    else:
        return None

if __name__ == "__main__":
    image1 = nib.load(sys.argv[1])
    image2 = nib.load(sys.argv[2])
    affine1 = np.eye(4)
    affine2 = image2.affine




