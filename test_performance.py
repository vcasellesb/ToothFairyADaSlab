import subprocess
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json
from utils import getpatientid

def dice_score(ground_truth_lab: str, ai_lab: str) -> str:

    dice_score=subprocess.run('seg_stats ' + f' {ground_truth_lab}' + ' -d' + f' {ai_lab}',
                               shell=True, capture_output=True, text=True)
    return dice_score.stdout

def compute_dice(gt: sitk.Image, pred: sitk.Image) -> float:
    overlap_measure = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measure.SetNumberOfThreads(1)
    overlap_measure.Execute(gt, pred)
    return overlap_measure.GetDiceCoefficient()

def compute_hd95(gt: sitk.Image, pred: sitk.Image) -> float:
    # gt.SetSpacing(np.array([1, 1, 1]).astype(np.float64))
    # pred.SetSpacing(np.array([1, 1, 1]).astype(np.float64))

    signed_distance_map = sitk.SignedMaurerDistanceMap(
        gt, squaredDistance=False, useImageSpacing=True
    )

    ref_distance_map = sitk.Abs(signed_distance_map)
    ref_surface = sitk.LabelContour(gt, fullyConnected=True)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(ref_surface)

    num_ref_surface_pixels = int(statistics_image_filter.GetSum())


    signed_distance_map_pred = sitk.SignedMaurerDistanceMap( pred, squaredDistance=False, useImageSpacing=True)
    seg_distance_map = sitk.Abs(signed_distance_map_pred)

    seg_surface = sitk.LabelContour(pred > 0.5, fullyConnected=True)

    seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)

    ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    statistics_image_filter.Execute(seg_surface > 0.5)

    num_seg_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_seg_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances)))  #

    all_surface_distances = seg2ref_distances + ref2seg_distances
    return np.percentile(all_surface_distances, 95)

def get_dict_results(labs_AI: list[str], 
                     labs_GT:list[str]) -> tuple[dict, dict]:
    
    labs_AI = sorted(labs_AI)
    labs_GT = sorted(labs_GT)
    labs_AI.sort(key=len)
    labs_GT.sort(key=len)

    assert len(labs_AI) == len(labs_GT), 'Not the same number of AI labels and ground truth labels'

    results = dict()
    results_hd95 = {}
    for AI, GT in zip(labs_AI, labs_GT):
        assert getpatientid(AI) == getpatientid(GT), f"Labels not comparable.\nGot {AI = } \n{GT = }"

        gt: sitk.Image = sitk.ReadImage(GT)
        pred: sitk.Image = sitk.ReadImage(AI)
        pred = sitk.Cast(pred, sitk.sitkUInt8)


        results[getpatientid(AI)] = compute_dice(gt, pred)
        results_hd95[getpatientid(AI)] = str(np.round(compute_hd95(gt=gt, pred=pred), 4))
        
    return results, results_hd95

def compute_mean_and_std(results_dict: dict) -> dict:
    results = np.array([float(r) for r in results_dict.values()])
    mean = np.mean(results)
    std = np.std(results)

    results_dict['mean'] = mean
    results_dict['std'] = std
    return results_dict


if __name__ == "__main__":
    from batchgenerators.utilities.file_and_folder_operations import load_json
    for file in ['output/results.json', 'output/results_hd95.json']:
        results_dict = load_json(file)
        results_dict = compute_mean_and_std(results_dict)
        save_json(results_dict, file.replace('.json', '_.json'))