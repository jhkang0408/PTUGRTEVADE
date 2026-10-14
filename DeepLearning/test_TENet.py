"""Target estimation accuracy of TENet and the alternative methods (Table III).

For every test clip, the candidate set consists of the real targets of the clip, dummy targets (random positions) up to
`targetNumber` targets, and the null target. The estimated target of each body part is compared with the ground truth
at every frame; the accuracy is averaged over the head, right hand and left hand. The experiment is repeated `trial`
times with newly sampled dummy targets.

    python test_TENet.py --model Ours      --targetNumber 3
    python test_TENet.py --model Baseline  --targetNumber 5
    python test_TENet.py --model Diffusion --targetNumber 7
    python test_TENet.py --model Indep     --targetNumber 3
"""
import os
import gc
import argparse
import numpy as np
import torch

from evaluation_utils import (SUBJECTS_ALL, repertories_for, ensure_split, user_clip_path, load_clip, fmt,
                              build_candidates_TENet, prepare_TENet_sequence, estimate_target_indices, gather_targets,
                              load_TENet, TENET_BATCH_SIZES)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Ours', choices=['Ours', 'Baseline', 'Diffusion', 'Indep'], help='TENet variant')
parser.add_argument('--targetNumber', type=int, default=3, help='Number of targets (candidates = targets + null)')
parser.add_argument('--trial', type=int, default=10, help='Number of trials (dummy targets are re-sampled)')
parser.add_argument('--batchSize', type=int, default=None, help='Window batch size (default depends on the model)')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--rawDataSplit', default=False, type=lambda x: x.lower() in ['true', '1'], help='Re-split raw/Data.txt into clips')
args = parser.parse_args()

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model, WINDOW_SIZE = load_TENet(args.model, device)
    K = args.targetNumber + 1
    BATCH_SIZE = args.batchSize if args.batchSize is not None else TENET_BATCH_SIZES[args.model]

    ensure_split('TENet', SUBJECTS_ALL, WINDOW_SIZE, force=args.rawDataSplit)

    accuracy_all, accuracy_head, accuracy_right, accuracy_left, accuracy_joint = [], [], [], [], []
    for trial in range(args.trial):
        count = count_head = count_right = count_left = count_joint = 0
        frames = 0
        with torch.no_grad():
            for subject in SUBJECTS_ALL:
                print('[trial %d] %s' % (trial + 1, subject), flush=True)
                for repertorie in repertories_for(subject):
                    for i in range(repertorie['num']):
                        data = load_clip(user_clip_path('TENet', subject, repertorie['repertorie'], i + 1)).to(device)
                        target_GT = data[:, 12:21]
                        targets, _ = build_candidates_TENet(data, repertorie['repertorie'], K, device)
                        data, all_targets = prepare_TENet_sequence(data, targets, WINDOW_SIZE)
                        head_idx, right_idx, left_idx = estimate_target_indices(args.model, model, data, all_targets, WINDOW_SIZE, K, BATCH_SIZE, device)
                        target_pred = gather_targets(all_targets, head_idx, right_idx, left_idx)

                        head_correct = torch.norm(target_GT[:, 0:3] - target_pred[:, 0:3], dim=1) <= 1e-2
                        right_correct = torch.norm(target_GT[:, 3:6] - target_pred[:, 3:6], dim=1) <= 1e-2
                        left_correct = torch.norm(target_GT[:, 6:9] - target_pred[:, 6:9], dim=1) <= 1e-2
                        frames += target_GT.shape[0]
                        count_head += head_correct.sum().item()
                        count_right += right_correct.sum().item()
                        count_left += left_correct.sum().item()
                        count_joint += (head_correct & right_correct & left_correct).sum().item()
                        count += head_correct.sum().item() + right_correct.sum().item() + left_correct.sum().item()

        accuracy_all.append(count / (frames * 3))
        accuracy_head.append(count_head / frames)
        accuracy_right.append(count_right / frames)
        accuracy_left.append(count_left / frames)
        accuracy_joint.append(count_joint / frames)
        print("[Trial %d] Overall: %.3f | Head: %.3f | Right: %.3f | Left: %.3f | All Correct: %.3f" % (
            trial + 1, accuracy_all[-1], accuracy_head[-1], accuracy_right[-1], accuracy_left[-1], accuracy_joint[-1]), flush=True)

    print('')
    print('=== TENet (%s), %d targets, %d trials ===' % (args.model, args.targetNumber, args.trial))
    print('Accuracy [%]       : ' + fmt([100 * a for a in accuracy_all]))
    print('  Head            : ' + fmt([100 * a for a in accuracy_head]))
    print('  Right hand      : ' + fmt([100 * a for a in accuracy_right]))
    print('  Left hand       : ' + fmt([100 * a for a in accuracy_left]))
    print('  All parts correct: ' + fmt([100 * a for a in accuracy_joint]))
