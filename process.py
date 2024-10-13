from pathlib import Path
import SimpleITK as sitk
import torch
import torch.nn as nn
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from preprocess import rotTFimage, split
from utils import find_targets, sitk2nii, add_case_id, nii2sitk, checkup
import os, time
from reshape_arrays import reshape, turn3dto2d
from postprocess import process_all_probabilty_arrays, merge_all_3d_probability_images, seg_maths_add_all_thr_masks, seg_maths_dil_ero, seg_maths_thr_per_axis, align_affines_between_mask_and_input, align_input_slice_with_predicted_array, align_input_image_affine_mask_affine_per_axis
import uuid
from test_performance import get_dict_results, mean_dice_score

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator, ## This is used to check that the image filename number is unique
    UniqueImagesValidator, ## This is used to check that the images are themselves unique
)

I_AM_IN_DOCKER = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)

def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class SimpleNet(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        avg = x.double().mean()
        return torch.where(x > avg, 1, 0)

class Toothfairy_algorithm(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            output_file=Path('/output/results.json') if I_AM_IN_DOCKER else Path('output/results.json'),
            input_path=Path('/input/images/cbct/') if I_AM_IN_DOCKER else Path('test/images/cbct'),
            output_path=Path('/output/images/inferior-alveolar-canal/') if I_AM_IN_DOCKER else Path('output/images/inferior-alveolar-canal/'),
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        self.models_dir = Path('/models/') if I_AM_IN_DOCKER else Path('models')
        self.targets = find_targets(input_path = self._input_path)
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)

    def preprocess(self, input_image: str, unique_folder_name):

        input_nii, nii_filename = sitk2nii(input_image, wheretosave=join(self.working_dir, 'input', unique_folder_name))
        _, rotated_image = rotTFimage(input_nii, nii_filename)
        axis = ["saggital", "coronal", "axial"]
        for ax in axis:

            splitted=split(rotated_image, target_axis=ax)

            path = join(splitted, ax)
            files = os.listdir(path)
            for file in files:
                add_case_id(join(path, file))
            if ax == "axial":
                continue
            else:
                reshape(path, func = turn3dto2d)

        return rotated_image.split('.nii.gz')[0]
    
    def postprocess(self, input_image: str):
        target_axes = ["saggital", "coronal", "axial"]

        list_of_thresholded_images = []

        for axis in target_axes:

            convert_npz_to_nifti_correct_shape = process_all_probabilty_arrays(
                join(self.working_dir, 'output', input_image, axis), axis)
            
                            
            nifti_probabilities_with_affine_aligned = align_input_slice_with_predicted_array(convert_npz_to_nifti_correct_shape, 
                                                                                                image_name=input_image,
                                                                                                axis=axis, 
                                                                                                input_images_dir=join(self.working_dir, 'input'))
            
            merged_probability_arrays = merge_all_3d_probability_images(nifti_probabilities_with_affine_aligned, axis, input_image)

            thresholded_image = seg_maths_thr_per_axis(merged_probability_arrays, axis=axis)

            aligned_input_affine_mask_affine_axis = align_input_image_affine_mask_affine_per_axis(input_image=join(self.working_dir, 'input', input_image, 'image.nii.gz'), mask_axis=thresholded_image)

            list_of_thresholded_images.append(aligned_input_affine_mask_affine_axis)
        
        final_image = seg_maths_add_all_thr_masks(list_of_thresholded_images, image_name=join(self.working_dir, 'output', input_image))
        
        sitk_image_final = nii2sitk(final_image)
        return sitk_image_final
        
    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image):

        predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=get_default_device(),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
        )
  
        datasets = ["Dataset002_saggital", "Dataset001_coronal", "Dataset003_axial"]

        unique_folder_name = uuid.uuid4().hex
        self.working_dir = Path(f'/working/{unique_folder_name}') if I_AM_IN_DOCKER else Path(f'working/{unique_folder_name}')
        if not self.working_dir.exists():
            os.makedirs(self.working_dir, exist_ok=True)
        
        starting_point = self.preprocess(input_image=input_image, unique_folder_name=unique_folder_name)

        for dataset in datasets:

            axis = dataset.split('_')[-1]

            predictor.initialize_from_trained_model_folder(
            join(self.models_dir, dataset, 'nnUNetTrainer__nnUNetPlans__2d'),
            use_folds=('all'),
            checkpoint_name='checkpoint_final.pth',
            )

            predictor.predict_from_files(join(starting_point, axis),
                                join(self.working_dir, 'output', unique_folder_name, axis),
                                save_probabilities=True, overwrite=True,
                                num_processes_preprocessing=3, num_processes_segmentation_export=3,
                                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
            
        result = self.postprocess(input_image=unique_folder_name)

        try:
            checkup(mask_sitk=result, input_sitk = input_image)
        except AssertionError as e:
            print(e)

        return result

    def test_performance(self, dir_labels_ground_truth):
        labels_ai = [join(self._output_path, f) for f in os.listdir(self._output_path) if f.endswith('.mha')]
        labels_ground_truth = [join(dir_labels_ground_truth, f) for f in os.listdir(dir_labels_ground_truth) if f.endswith('.mha')]
        targets = self.targets

        results_dict_test = get_dict_results(labels_ground_truth, labs_GT=labels_ai, targets=targets)
        dice_score_final = mean_dice_score(results_dict=results_dict_test)

        print(dice_score_final)
        return dice_score_final

if __name__ == "__main__":

    start_time = time.time()

    Toothfairy_algorithm().process()

    end_time = time.time()

    print("Elapsed time:", end_time - start_time)

    if not I_AM_IN_DOCKER:
        print('Evaluating...')
        Toothfairy_algorithm().test_performance(dir_labels_ground_truth=os.path.abspath('ground_truths'))