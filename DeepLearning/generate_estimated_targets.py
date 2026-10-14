"""Generates the estimated targets used by the BP-GTNet / UGNet evaluations ('Estimated Targets(N)/').

For every test clip and trial, TENet estimates the targets of the user from the candidate set (real targets + dummy
targets + null target, see test_TENet.py). The avatar's targets are then sampled at new random positions (a dissimilar
environment) with a one-to-one correspondence to the user's candidates, so that both the ground-truth and the estimated
user targets can be mapped to avatar targets. One file per clip is written:

    Estimated Targets(N)/<subject>/<repertoire>/<trial>/<clip>.txt   (T x 39, tab separated)
        [ 0: 9] ground-truth avatar targets      (spherical, avatar reference frame)
        [ 9:18] estimated avatar targets
        [18:21] ray features of the candidate set (head, right hand, left hand)
        [21:30] ground-truth user targets        (spherical, user reference frame)
        [30:39] estimated user targets

    python generate_estimated_targets.py --targetNumber 3
    python generate_estimated_targets.py --targetNumber 5 --model Baseline     # -> 'Estimated Targets(5)_Baseline'
"""
import os
import os.path as osp
import gc
import argparse
import numpy as np
import torch

from evaluation_utils import (SUBJECTS_ALL, repertories_for, ensure_split, user_clip_path, load_clip,
                              build_candidates_with_avatar, prepare_TENet_sequence, estimate_target_indices, gather_targets,
                              sample_fake_target, load_TENet, TENET_BATCH_SIZES)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Ours', choices=['Ours', 'Baseline', 'Diffusion', 'Indep'], help='TENet variant')
parser.add_argument('--targetNumber', type=int, default=3, help='Number of targets (candidates = targets + null)')
parser.add_argument('--trial', type=int, default=10, help='Number of trials (dummy / avatar targets are re-sampled)')
parser.add_argument('--batchSize', type=int, default=None, help='Window batch size (default depends on the model)')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--rawDataSplit', default=False, type=lambda x: x.lower() in ['true', '1'], help='Re-split raw/Data.txt into clips')
parser.add_argument('--out', default='Estimated Targets', help='Output folder prefix')
args = parser.parse_args()


def sample_avatar_target(repertorie, i, T, device):
    """Avatar target i of a clip (spherical, constant over the clip). Touching targets lie within reach (0.4 ~ 0.8 m),
    except for the explanation repertoires whose explained object is placed at distance 1."""
    horizontal_angle = torch.rand(1, device=device) * torch.pi - (torch.pi / 2)
    vertical_angle = torch.rand(1, device=device) * (2 * torch.pi / 3) - (torch.pi / 3)
    if repertorie.split()[0] == 'Pointing':
        distance = torch.ones(1, device=device)
    elif repertorie == 'Touching single target with explanation' and i == 0:
        distance = torch.ones(1, device=device)
    elif repertorie == 'Touching single target with explanation shift shift' and i == 2:
        distance = torch.ones(1, device=device)
    else:
        distance = torch.rand(1, device=device) * 0.4 + 0.4
    return torch.stack([horizontal_angle.expand(T), vertical_angle.expand(T), distance.expand(T)], dim=1)


def find_target_index(gt_values, all_targets, num_real_targets, K):
    """Index of the candidate that equals the ground-truth target at every frame (K-1 for the null target)."""
    indices = torch.zeros(gt_values.shape[0], dtype=torch.long, device=gt_values.device)
    for i in range(num_real_targets):
        match = torch.all(torch.abs(gt_values - all_targets[i]) < 1e-6, dim=1)
        indices[match] = i
    indices[torch.all(torch.abs(gt_values) < 1e-6, dim=1)] = K - 1
    return indices


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model, WINDOW_SIZE = load_TENet(args.model, device)
    K = args.targetNumber + 1
    BATCH_SIZE = args.batchSize if args.batchSize is not None else TENET_BATCH_SIZES[args.model]
    out_dir = args.out + '(' + str(args.targetNumber) + ')' + ('' if args.model == 'Ours' else '_' + args.model)

    ensure_split('TENet', SUBJECTS_ALL, WINDOW_SIZE, force=args.rawDataSplit)

    for trial in range(args.trial):
        with torch.no_grad():
            for subject in SUBJECTS_ALL:
                print('[trial %d] %s' % (trial + 1, subject), flush=True)
                for repertorie in repertories_for(subject):
                    save_path = osp.join(out_dir, subject, repertorie['repertorie'], str(trial + 1))
                    os.makedirs(save_path, exist_ok=True)
                    for i in range(repertorie['num']):
                        data = load_clip(user_clip_path('TENet', subject, repertorie['repertorie'], i + 1)).to(device)
                        T = data.shape[0]
                        target_GT_user = data[:, 30:39]      # ground-truth user targets in the BP-GTNet reference frame

                        # 1) target estimation on the user side
                        targets_TENet, targets_user, ray, fake = build_candidates_with_avatar(data, repertorie['repertorie'], K, device)
                        data_p, all_targets_TENet = prepare_TENet_sequence(data, targets_TENet, WINDOW_SIZE)
                        all_targets_user = torch.stack(targets_user, dim=0)
                        head_idx, right_idx, left_idx = estimate_target_indices(args.model, model, data_p, all_targets_TENet, WINDOW_SIZE, K, BATCH_SIZE, device)
                        target_pred_user = gather_targets(all_targets_user, head_idx, right_idx, left_idx)

                        # 2) avatar targets at new positions (one-to-one correspondence with the user's candidates)
                        targets_avatar = [sample_avatar_target(repertorie['repertorie'], i_target, T, device) for i_target in range(K - 1)]
                        targets_avatar.append(torch.zeros(T, 3, device=device))
                        all_targets_avatar = torch.stack(targets_avatar, dim=0)
                        target_pred_avatar = gather_targets(all_targets_avatar, head_idx, right_idx, left_idx)

                        num_real_targets = K - fake - 1
                        gt_head = find_target_index(target_GT_user[:, 0:3], all_targets_user, num_real_targets, K)
                        gt_right = find_target_index(target_GT_user[:, 3:6], all_targets_user, num_real_targets, K)
                        gt_left = find_target_index(target_GT_user[:, 6:9], all_targets_user, num_real_targets, K)
                        target_GT_avatar = gather_targets(all_targets_avatar, gt_head, gt_right, gt_left)

                        combined = np.concatenate([target_GT_avatar.cpu().numpy(),
                                                   target_pred_avatar.cpu().numpy(),
                                                   ray.cpu().numpy(),
                                                   target_GT_user.cpu().numpy(),
                                                   target_pred_user.cpu().numpy()], axis=1)
                        np.savetxt(osp.join(save_path, str(i + 1) + '.txt'), combined, delimiter='\t')
    print('saved to ' + out_dir)
