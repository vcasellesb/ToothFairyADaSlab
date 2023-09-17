import subprocess

keys_voting_strategy = {'union_voting': 0.99, 'majority_voting': 1.99, 'unanimous': 2.99, 'softmax_average': 1.49}

def vote_axis(label: str) -> str:

    """
    Thresholds a softmax label (probabilities) by 0.5 and binarizes it.
    """

    command = f'seg_maths {label} -thr 0.49 -bin {label.replace(".nii.gz", "_binned.nii.gz")}'
    subprocess.run(command, shell = True)

    return label.replace(".nii.gz", "_binned.nii.gz")


def voting(labels: list, voting_strategy: str, image_name: str) -> str:

    if voting_strategy != "softmax_average":
        labels = [vote_axis(lab) for lab in labels]
    assert len(labels) == 3, "More labels than there should be"

    print(labels)

    thr = keys_voting_strategy[voting_strategy]
    command = f'seg_maths {labels[0]} -add {labels[1]} -add {labels[2]} -thr {thr} -bin {image_name}_{voting_strategy}.nii.gz'

    subprocess.run(command, shell = True)

    return f'{image_name}_{voting_strategy}.nii.gz'