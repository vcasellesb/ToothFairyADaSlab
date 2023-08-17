import subprocess
from batchgenerators.utilities.file_and_folder_operations import save_json
from utils import getpatientid

def dice_score(ground_truth_lab, ai_lab):

    dice_score=subprocess.run('seg_stats ' + f' {ground_truth_lab}' + ' -d' + f' {ai_lab}' + ' | grep "Label\[1\]"',
                               shell=True, capture_output=True, text=True)
    return dice_score.stdout

def get_dict_results(labs_AI, labs_GT, targets) -> dict:
    labs_AI = sorted(labs_AI)
    labs_GT = sorted(labs_GT)
    targets = sorted(targets)

    assert len(labs_AI) == len(labs_GT), 'Not the same number of AI labels and ground truth labels'

    results = dict()
    for AI, GT, target in zip(labs_AI, labs_GT, targets):
        assert getpatientid(AI) == getpatientid(GT), "Labels not comparable"

        results[getpatientid(AI)] = dice_score(AI, GT)
    
    save_json(results, 'output/DICE_results.json')
    
    return results

def mean_dice_score(results_dict: dict) -> float:

    sirs=results_dict.keys()
    final_score = 0
    for sir in sirs:
        final_score += float(results_dict[sir].split(" ")[-1].split('\n')[0])
    return final_score/len(sirs)