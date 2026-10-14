"""Shared helpers for the evaluation scripts.

    test_TENet.py                  Table III : target estimation accuracy of TENet and the alternative methods
    generate_estimated_targets.py  builds 'Estimated Targets(N)/' (TENet estimates + sampled avatar targets) used below
    test_BP_GTNet.py               Table IV  : FGD / TMP / DIP of BP-GTNet and its ablations, DIP-TE after IK post-processing
    test_UGNet.py                  Table IV  : the UGNet baseline (our previous work)

All metrics are evaluated on the test subjects listed below; see the README for the data layout.
"""
import os
import os.path as osp
import numpy as np
import scipy.linalg as linalg
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from data_parser import data_parser
from model import TENet, TENet_Baseline, TENet_Diffusion, TENet_Indep, BP_GTNet, MPNet_MoE, UGNet_MoE, ConvolutionalAutoEncoder

# ---------------------------------------------------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------------------------------------------------
SUBJECTS_MAIN = ['161', '172', '173', '174', '180']
SUBJECTS_EXTRA = ['158(HJ)', '165(YN)', '166(HS)', '174(KB)', '175(RS)', '180(DH)', '187(ES)']
SUBJECTS_ALL = SUBJECTS_MAIN + SUBJECTS_EXTRA

REPERTORIES_MAIN = [{'repertorie': 'Pointing at single target', 'num': 27},
                    {'repertorie': 'Pointing at two targets', 'num': 8},
                    {'repertorie': 'Pointing at single target with gaze shift', 'num': 8},
                    {'repertorie': 'Pointing at single target with explanation', 'num': 8},
                    {'repertorie': 'Touching single target', 'num': 30},
                    {'repertorie': 'Touching single target with gaze shift', 'num': 8},
                    {'repertorie': 'Touching single target with explanation', 'num': 8},
                    {'repertorie': 'Touching two targets', 'num': 8}]
REPERTORIES_EXTRA = [{'repertorie': 'Pointing at single target with explanation shift shift', 'num': 8},
                     {'repertorie': 'Touching single target with explanation shift shift', 'num': 8}]
REPERTORIES_DIP = [{'repertorie': 'Pointing at single target', 'num': 27},
                   {'repertorie': 'Touching single target', 'num': 30}]


def repertories_for(subject):
    return REPERTORIES_MAIN if subject in SUBJECTS_MAIN else REPERTORIES_EXTRA


def is_right_handed(subject):
    """Subjects 173 and 174 performed the deictic gestures with the left hand."""
    return subject not in ('173', '174')


def test_data_dir(network, subject):
    return 'data(raw)/' + network + '/test/' + subject + '/'


def ensure_split(network, subjects, window_size=30, force=False):
    """Splits raw/Data.txt of the test set into per-clip split/Data<i>.txt files (once, unless force=True)."""
    for subject in subjects:
        parser = data_parser(repertories_for(subject), window_size, 0, network, 0)
        parser.raw_data_dir = test_data_dir(network, subject)
        need_split = force
        for repertorie in parser.repertories:
            folder_split = osp.join(parser.raw_data_dir, repertorie['repertorie'], 'split')
            os.makedirs(folder_split, exist_ok=True)
            if len(os.listdir(folder_split)) < repertorie['num']:
                need_split = True
        if need_split:
            print('splitting raw data: ' + parser.raw_data_dir)
            parser.split()


def user_clip_path(network, subject, repertorie, index):
    return osp.join(test_data_dir(network, subject), repertorie, 'split', 'Data' + str(index) + '.txt')


def estimated_target_path(target_dir, subject, repertorie, trial, index):
    return osp.join(target_dir, subject, repertorie, str(trial), str(index) + '.txt')


def load_clip(path):
    return torch.tensor(np.loadtxt(path).astype(np.float32), dtype=torch.float32)


def mean_std(values):
    values = np.array(values, dtype=np.float64)
    return values.mean(), values.std()


def fmt(values, decimals=2):
    m, s = mean_std(values)
    return ('%.' + str(decimals) + 'f +- %.' + str(decimals) + 'f') % (m, s)


# ---------------------------------------------------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------------------------------------------------
def pad_front(x, n):
    """Prepends n copies of the first frame."""
    return torch.cat([x[0:1].repeat(n, *([1] * (x.dim() - 1))), x], dim=0)


def moving_average_bool(X, start, count, window=15):
    """Trailing moving average (window frames, front-padded with the first frame) of the boolean columns
    X[:, start:start+count]; this turns the binary ray/default features into continuous ones."""
    Xp = pad_front(X, window)
    out = X.clone()
    for j in range(count):
        cum_sum = torch.cumsum(Xp[:, start + j], dim=0)
        out[:, start + j] = (cum_sum[window:] - cum_sum[:-window]) / window
    return out


def consistency_confidence(targets, ray_head):
    """Per-part confidence of a target sequence (T, 9): it grows by 1/15 per frame while the estimated target of the
    part stays the same and the ray feature of the head is on, decays otherwise, and resets when the target changes."""
    targets = targets.detach().cpu().numpy()
    ray_head = ray_head.detach().cpu().numpy()
    T = targets.shape[0]
    conf = np.zeros((T, 3), dtype=np.float32)
    for part, cols in enumerate([slice(0, 3), slice(3, 6), slice(6, 9)]):
        c = 0.0
        for i in range(T - 1):
            if np.array_equal(targets[i + 1, cols], targets[i, cols]):
                part_nonzero = not np.all(targets[i + 1, cols] == 0)
                head_nonzero = not np.all(targets[i + 1, 0:3] == 0)
                if part == 0:
                    if part_nonzero:
                        c = min(1.0, c + 1.0 / 15.0) if ray_head[i + 1] == 1 else max(0.0, c - 1.0 / 15.0)
                else:
                    if part_nonzero and head_nonzero:
                        c = min(1.0, c + 1.0 / 15.0) if ray_head[i + 1] == 1 else max(0.0, c - 1.0 / 15.0)
                    else:
                        c = max(0.0, c - 1.0 / 15.0)
            else:
                c = 0.0
            conf[i + 1, part] = c
    return torch.tensor(conf)


# ---------------------------------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------------------------------
def sph2cart(s):
    """(horizontal angle, vertical angle, distance) -> (x, y, z) in the reference frame."""
    h, v, d = s[..., 0], s[..., 1], s[..., 2]
    return torch.stack([d * torch.sin(h) * torch.cos(v), d * torch.sin(v), d * torch.cos(h) * torch.cos(v)], dim=-1)


def look_at_root_rotation(target_position, eps=1e-8):
    """Rotation (B, 3, 3) of the target frame: forward points from the target to the root, up is the world up."""
    forward = -target_position / (torch.norm(target_position, dim=1, keepdim=True) + eps)
    world_up = torch.tensor([0, 1, 0], dtype=forward.dtype, device=forward.device).expand_as(forward)
    right = torch.cross(world_up, forward, dim=1)
    right = right / torch.norm(right, dim=1, keepdim=True)
    up = torch.cross(forward, right, dim=1)
    return torch.stack([right, up, forward], dim=2)


def relative_direction(direction, rotation):
    return torch.bmm(rotation.transpose(1, 2), direction.unsqueeze(2)).squeeze(2)


def relative_position(position, origin, rotation):
    return torch.bmm(rotation.transpose(1, 2), (position - origin).unsqueeze(2)).squeeze(2)


def sixd2matrot(pose_6d):
    """(..., 6) = (forward, up) -> orthonormal rotation matrices (..., 3, 3) with columns (right, up, forward)."""
    fwd = F.normalize(pose_6d[..., 0:3], dim=-1)
    up_raw = pose_6d[..., 3:6]
    up = up_raw - (fwd * up_raw).sum(dim=-1, keepdim=True) * fwd
    up = F.normalize(up, dim=-1)
    right = torch.cross(up, fwd, dim=-1)
    right = F.normalize(right, dim=-1)
    up = torch.cross(fwd, right, dim=-1)
    return torch.stack([right, up, fwd], dim=-1)


def rot6_geodesic_deg(A_pose_6d, B_pose_6d):
    """Geodesic angle [deg] between the rotations of 3 parts, (N, T, 18) x 2 -> (N, T, 3)."""
    N, T, _ = A_pose_6d.shape
    A = A_pose_6d.reshape(N, T, 3, 6)
    B = B_pose_6d.reshape(N, T, 3, 6)
    Rrel = sixd2matrot(A).transpose(-1, -2) @ sixd2matrot(B)
    trace = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.rad2deg(torch.acos(cos))


def fingertip_world(forward, up, position, local):
    """Index fingertip position from the hand frame (forward, up, position) and its hand-local offset."""
    right = F.normalize(torch.cross(up, forward, dim=1), dim=1)
    rotation = torch.stack([right, up, forward], dim=2)
    return torch.bmm(rotation, local.unsqueeze(2)).squeeze(2) + position


def recompute_head_features_wrt_target(User, head_target_sph, eps=1e-8):
    """Re-expresses the head features of the encoder input (User[:, 15:24]) with respect to the *estimated* head target
    (the exported data stores them with respect to the ground-truth target)."""
    target_position = sph2cart(head_target_sph)
    rotation = look_at_root_rotation(target_position, eps)
    head_forward = relative_direction(User[:, 42:45], rotation)
    head_up = relative_direction(User[:, 45:48], rotation)
    head_position_magnitude = torch.norm(User[:, 48:51], dim=1, keepdim=True)
    head_position_in_target = relative_position(User[:, 48:51], target_position, rotation)
    head_position = head_position_magnitude * head_position_in_target / (torch.norm(head_position_in_target, dim=1, keepdim=True) + eps)

    is_at_origin = torch.norm(target_position, dim=1, keepdim=True) < 1e-6
    zeros = torch.zeros_like(head_forward)
    User[:, 15:18] = torch.where(is_at_origin.expand_as(head_forward), zeros, head_forward)
    User[:, 18:21] = torch.where(is_at_origin.expand_as(head_up), zeros, head_up)
    User[:, 21:24] = torch.where(is_at_origin.expand_as(head_position), zeros, head_position)
    return User


def target_relative_end_effectors(Target, EE_World, eps=1e-8):
    """Target-view motion (TMP): expresses the end-effectors (T, 27) = 3 x (forward, up, position) in the frame of the
    target of each part (Target: (T, 9) spherical). Parts without a target (0, 0, 0) are zeroed."""
    out = []
    for part in range(3):
        target_position = sph2cart(Target[:, 3 * part:3 * part + 3])
        rotation = look_at_root_rotation(target_position, eps)
        ee = EE_World[:, 9 * part:9 * part + 9]
        forward = relative_direction(ee[:, 0:3], rotation)
        up = relative_direction(ee[:, 3:6], rotation)
        position = relative_position(ee[:, 6:9], target_position, rotation)
        is_at_origin = torch.norm(target_position, dim=1, keepdim=True) < 1e-6
        zeros = torch.zeros_like(forward)
        out += [torch.where(is_at_origin.expand_as(forward), zeros, forward),
                torch.where(is_at_origin.expand_as(up), zeros, up),
                torch.where(is_at_origin.expand_as(position), zeros, position)]
    return torch.cat(out, -1)


# ---------------------------------------------------------------------------------------------------------------------
# TENet: candidate targets and target estimation
# ---------------------------------------------------------------------------------------------------------------------
def n_real_targets(repertorie):
    if repertorie in ('Pointing at single target', 'Touching single target'):
        return 1
    if repertorie.endswith('shift shift'):
        return 3
    return 2


def sample_fake_target(repertorie, T, device):
    """A dummy target candidate (T, 3) in spherical coordinates, constant over the clip."""
    horizontal_angle = torch.rand(1, device=device) * torch.pi - (torch.pi / 2)        # -pi/2 ~ pi/2
    vertical_angle = torch.rand(1, device=device) * (2 * torch.pi / 3) - (torch.pi / 3)  # -pi/3 ~ pi/3
    if repertorie.split()[0] == 'Pointing':
        distance = torch.ones(1, device=device)
    else:
        distance = torch.rand(1, device=device) * 0.4 + 0.4
    return torch.stack([horizontal_angle.expand(T), vertical_angle.expand(T), distance.expand(T)], dim=1)


def build_candidates_TENet(data, repertorie, K, device):
    """Candidate targets of a TENet test clip: the real targets of the clip (data[:, 21:30]), K-1-n dummy targets and
    the null target. Returns the list of (T, 3) spherical targets and the number of dummy targets."""
    T = data.shape[0]
    n = n_real_targets(repertorie)
    targets = [data[:, 21 + 3 * j:24 + 3 * j] for j in range(n)]
    fake = K - 1 - n
    for _ in range(fake):
        targets.append(sample_fake_target(repertorie, T, device))
    targets.append(torch.zeros(T, 3, device=device))
    return targets, fake


def build_candidates_with_avatar(data, repertorie, K, device):
    """Same as build_candidates_TENet, but the dummy targets are sampled in the reference frame used by BP-GTNet
    (data[:, 39:48]) and converted to the TENet frame; also returns the ray features of the candidates."""
    T = data.shape[0]
    n = n_real_targets(repertorie)
    targets_TENet = [data[:, 21 + 3 * j:24 + 3 * j] for j in range(n)]
    targets_avatar = [data[:, 39 + 3 * j:42 + 3 * j] for j in range(n)]
    fake = K - 1 - n
    head_position = data[:, 48:51]
    for _ in range(fake):
        target = sample_fake_target(repertorie, T, device)
        targets_avatar.append(target)
        direction = sph2cart(target) - head_position
        direction = direction / torch.norm(direction, dim=1, keepdim=True)
        horizontal_angle = torch.atan2(direction[:, 0], direction[:, 2])
        vertical_angle = torch.atan2(direction[:, 1], torch.sqrt(direction[:, 0] ** 2 + direction[:, 2] ** 2))
        if repertorie.split()[0] == 'Pointing':
            distance = torch.ones(T, device=device)
        else:
            distance = torch.clamp(torch.norm(sph2cart(target) - head_position, dim=1), 0.0, 1.0)
        targets_TENet.append(torch.stack([horizontal_angle, vertical_angle, distance], dim=1))

    # ray features of the candidate set (gaze / eye-finger rays within 10 degrees of a candidate)
    Gaze, EF_RightHand, EF_LeftHand = data[:, 51:54], data[:, 54:57], data[:, 57:60]
    ray = torch.zeros(T, 3, device=device)
    for target in targets_avatar:
        et = sph2cart(target) - head_position
        et = et / torch.norm(et, dim=1, keepdim=True)
        for j, ef in enumerate((Gaze, EF_RightHand, EF_LeftHand)):
            angle = torch.acos(torch.sum(et * ef, dim=1).clamp(-1.0, 1.0)) * 180 / torch.pi
            ray[:, j] = torch.where(angle < 10.0, 1.0, ray[:, j])

    targets_TENet.append(torch.zeros(T, 3, device=device))
    targets_avatar.append(torch.zeros(T, 3, device=device))
    return targets_TENet, targets_avatar, ray, fake


def prepare_TENet_sequence(data, targets, window):
    """Smooths the default features (cols 9:12), front-pads the sequence by window-1 frames and stacks the candidates.
    Returns (padded data, candidates (K, T, 3))."""
    data = moving_average_bool(data, 9, 3)
    data = pad_front(data, window - 1)
    return data, torch.stack(targets, dim=0)


def estimate_target_indices(kind, model, data, all_targets, window, K, batch_size, device):
    """Estimates, for every frame, the candidate index of the head / right-hand / left-hand target.

    kind 'Ours' / 'Baseline': posterior over all K^3 target combinations (joint estimation)
    kind 'Indep'            : independent posterior per part
    kind 'Diffusion'        : regressed target, nearest candidate per part
    data: padded sequence (T + window - 1, >=12), all_targets: (K, T, 3)"""
    T = all_targets.shape[1]
    head_idx = torch.zeros(T, dtype=torch.long, device=device)
    right_idx = torch.zeros(T, dtype=torch.long, device=device)
    left_idx = torch.zeros(T, dtype=torch.long, device=device)

    if kind in ('Ours', 'Baseline'):
        num_combinations = K ** 3
        combination = torch.arange(num_combinations, device=device)
        head_c = (combination // (K ** 2)) % K
        right_c = (combination // K) % K
        left_c = combination % K

    for t_start in range(0, T, batch_size):
        t_end = min(t_start + batch_size, T)
        batch_len = t_end - t_start
        batch_targets = all_targets[:, t_start:t_end]                                                   # (K, B, 3)
        X_windows = torch.stack([data[t:t + window, :12] for t in range(t_start, t_end)], dim=0)      # (B, W, 12)

        if kind in ('Ours', 'Baseline'):
            batch_grid, comb_grid = torch.meshgrid(torch.arange(batch_len, device=device), combination, indexing='ij')
            flat_batch, flat_comb = batch_grid.reshape(-1), comb_grid.reshape(-1)
            X_batch = X_windows.repeat_interleave(num_combinations, dim=0)
            Y_batch = torch.zeros((batch_len * num_combinations, 9), device=device)
            Y_batch[:, 0:3] = batch_targets[head_c[flat_comb], flat_batch]
            Y_batch[:, 3:6] = batch_targets[right_c[flat_comb], flat_batch]
            Y_batch[:, 6:9] = batch_targets[left_c[flat_comb], flat_batch]
            log_probs = model.posterior_probability(X_batch, Y_batch).reshape(batch_len, num_combinations)
            _, max_indices = torch.max(F.softmax(log_probs, dim=1), dim=1)
            head_idx[t_start:t_end] = (max_indices // (K ** 2)) % K
            right_idx[t_start:t_end] = (max_indices // K) % K
            left_idx[t_start:t_end] = max_indices % K

        elif kind == 'Indep':
            batch_grid, target_grid = torch.meshgrid(torch.arange(batch_len, device=device), torch.arange(K, device=device), indexing='ij')
            single_target = batch_targets[target_grid.reshape(-1), batch_grid.reshape(-1)]            # (B*K, 3)
            X_batch = X_windows.repeat_interleave(K, dim=0)
            Y_batch = torch.cat([single_target, single_target, single_target], dim=-1)
            log_prob_head, log_prob_right, log_prob_left = model.posterior_probability(X_batch, Y_batch)
            head_idx[t_start:t_end] = torch.argmax(log_prob_head.reshape(batch_len, K), dim=1)
            right_idx[t_start:t_end] = torch.argmax(log_prob_right.reshape(batch_len, K), dim=1)
            left_idx[t_start:t_end] = torch.argmax(log_prob_left.reshape(batch_len, K), dim=1)

        elif kind == 'Diffusion':
            pred = model.sample_ddpm(X_windows)                                                      # (B, 9)
            candidates = batch_targets.permute(1, 0, 2)                                              # (B, K, 3)
            head_idx[t_start:t_end] = torch.argmin(torch.norm(pred[:, 0:3].unsqueeze(1) - candidates, dim=2), dim=1)
            right_idx[t_start:t_end] = torch.argmin(torch.norm(pred[:, 3:6].unsqueeze(1) - candidates, dim=2), dim=1)
            left_idx[t_start:t_end] = torch.argmin(torch.norm(pred[:, 6:9].unsqueeze(1) - candidates, dim=2), dim=1)
        else:
            raise ValueError('unknown TENet kind: ' + kind)

    return head_idx, right_idx, left_idx


def gather_targets(all_targets, head_idx, right_idx, left_idx):
    """(K, T, 3) candidates + per-frame indices -> (T, 9) estimated targets."""
    time_indices = torch.arange(all_targets.shape[1], device=all_targets.device)
    return torch.cat((all_targets[head_idx, time_indices], all_targets[right_idx, time_indices], all_targets[left_idx, time_indices]), -1)


# ---------------------------------------------------------------------------------------------------------------------
# IK post-processing for touching gestures (DIP-TE after IK)
# ---------------------------------------------------------------------------------------------------------------------
def ik_touching_error(rec, threshold=0.10, rate=0.03, conf_threshold=0.75):
    """Touching error [cm] of a clip before and after the IK post-processing of the real-time system.

    rec holds per-frame arrays: tip_raw (avatar fingertip, network output), target_gt, target_est (avatar targets),
    flag (touching frames), conf (confidence of the hand target), d_user_est (distance between the user's fingertip and
    the user's estimated target [m]) and valid (frames with a ground-truth target).

    While conf > conf_threshold and the user's fingertip is within `threshold` of the estimated target, the IK weight w
    of the avatar's fingertip end-effector grows by `rate` per frame (otherwise it decays by `rate`), and the fingertip
    is moved toward the estimated target: tip = lerp(tip_raw, target_est, w). Both errors take the minimum over the
    touching frames, as DIP-TE does. Returns (before, after) or None if the clip has no touching frame."""
    T = len(rec['flag'])
    eval_mask = (rec['flag'] == 1) & (rec['valid'] == 1)
    if eval_mask.sum() == 0:
        return None
    d_raw = np.linalg.norm(rec['target_gt'] - rec['tip_raw'], axis=1) * 100
    est = rec['target_est'].copy()
    hold = None
    for t in range(T):     # the system keeps the last (non-null) estimated target as the IK target
        if np.any(est[t] != 0):
            hold = est[t]
        elif hold is not None:
            est[t] = hold
    w = 0.0
    d_ik = np.zeros(T, dtype=np.float32)
    for t in range(T):
        active = (rec['conf'][t] > conf_threshold) and (rec['d_user_est'][t] < threshold) and np.any(est[t] != 0)
        w = min(1.0, max(0.0, w + rate if active else w - rate))
        tip = rec['tip_raw'][t] + w * (est[t] - rec['tip_raw'][t]) if np.any(est[t] != 0) else rec['tip_raw'][t]
        d_ik[t] = np.linalg.norm(rec['target_gt'][t] - tip) * 100
    return float(d_raw[eval_mask].min()), float(d_ik[eval_mask].min())


def new_record():
    return {k: [] for k in ['tip_raw', 'target_gt', 'target_est', 'flag', 'conf', 'd_user_est', 'valid']}


def finalize_record(rec):
    return {k: np.array(v, dtype=np.float32) for k, v in rec.items()}


# ---------------------------------------------------------------------------------------------------------------------
# FGD (Frechet Gesture Distance)
# ---------------------------------------------------------------------------------------------------------------------
class NpyFolderDataset(Dataset):
    def __init__(self, folder_name):
        self.paths = [osp.join(folder_name, str(i) + '.npy') for i in range(len(os.listdir(folder_name)))]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return {'data': torch.from_numpy(np.load(self.paths[idx]))}


class MotionWindowWriter:
    """Stores the real and generated end-effector motions as 30-frame windows (stride 15) for the FGD metric
    (temporary files under data/, one folder pair per process)."""
    def __init__(self, tag=None):
        tag = str(os.getpid()) if tag is None else str(tag)
        self.real_dir = 'data/test_FGD_real_' + tag + '/'
        self.generated_dir = 'data/test_FGD_generated_' + tag + '/'
        for d in (self.real_dir, self.generated_dir):
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.remove(osp.join(d, f))
        self.count = 0

    def cleanup(self):
        for d in (self.real_dir, self.generated_dir):
            for f in os.listdir(d):
                os.remove(osp.join(d, f))
            os.rmdir(d)

    def write(self, real, generated):
        np.save(osp.join(self.real_dir, str(self.count)), real[-30:, :].cpu().numpy())
        np.save(osp.join(self.generated_dir, str(self.count)), generated[-30:, :].cpu().numpy())
        self.count += 1
        real = torch.transpose(real.unfold(0, 30, 15), 1, 2)
        generated = torch.transpose(generated.unfold(0, 30, 15), 1, 2)
        for i in range(real.shape[0]):
            np.save(osp.join(self.real_dir, str(self.count)), real[i].cpu().numpy())
            np.save(osp.join(self.generated_dir, str(self.count)), generated[i].cpu().numpy())
            self.count += 1


def compute_embeddings(autoencoder, folder, device):
    embeddings = []
    for data in DataLoader(NpyFolderDataset(folder), batch_size=1, shuffle=False, drop_last=False):
        embeddings.append(autoencoder(data['data'].to(device), True).squeeze())
    return torch.stack(embeddings).detach().cpu().numpy()


def frechet_distance(real_embeddings, generated_embeddings):
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def fgd_of_writer(writer, device):
    autoencoder = load_AutoEncoder(device)
    real = compute_embeddings(autoencoder, writer.real_dir, device)
    generated = compute_embeddings(autoencoder, writer.generated_dir, device)
    writer.cleanup()
    return frechet_distance(real, generated)


# ---------------------------------------------------------------------------------------------------------------------
# Trained models
# ---------------------------------------------------------------------------------------------------------------------
MODEL_DIR = 'model(trained)'
TENET_CHECKPOINTS = {'Ours': 'TENet_Ours_10.pt',
                     'Baseline': 'TENet_Baseline_20.pt',
                     'Diffusion': 'TENet_Diffusion_50.pt',
                     'Indep': 'TENet_Indep_10.pt'}
TENET_BATCH_SIZES = {'Ours': 32, 'Baseline': 32, 'Diffusion': 1024, 'Indep': 128}
BP_GTNET_CHECKPOINTS = {'Ours': ('BP_GTNet_Ours_90.pt', True, True),
                        'without_MD': ('BP_GTNet_without_MD_90.pt', False, True),
                        'without_PAD': ('BP_GTNet_without_PAD_90.pt', True, False),
                        'without_EE_Dual': ('BP_GTNet_without_EE_Dual_90.pt', True, True)}


def load_TENet(kind, device):
    """Returns (model, window size) of a TENet variant: 'Ours', 'Baseline', 'Diffusion' or 'Indep'."""
    checkpoint = torch.load(osp.join(MODEL_DIR, TENET_CHECKPOINTS[kind]), map_location=device)
    c = checkpoint['config']
    if kind == 'Diffusion':
        model = TENet_Diffusion(T=c['T'], n_diffusion_layers=c['n_diffusion_layers'], input_dim=c['input_dim'],
                                hidden_dim=c['hidden_dim'], window_dim=c['window_dim'], target_dim=c['target_dim'])
    else:
        kwargs = dict(head_depth=c['head_depth'], head_sz=c['head_sz'], feat_sz=c['feat_sz'], depth=c['depth'],
                      condition_dim=c['condition_dim'], input_dim=c['input_dim'], window_dim=c['window_dim'],
                      target_dim=c['target_dim'], n_flow_layers=c['n_flow_layers'],
                      n_encoder_layers=c['n_encoder_layers'], n_encoder_head=c['n_encoder_head'])
        if kind == 'Ours':
            model = TENet(use_null_token=c['use_null_token'], **kwargs)
        elif kind == 'Baseline':
            model = TENet_Baseline(**kwargs)
        elif kind == 'Indep':
            model = TENet_Indep(use_null_token=c['use_null_token'], **kwargs)
        else:
            raise ValueError('unknown TENet kind: ' + kind)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, c['window_dim']


def load_BP_GTNet(kind, device):
    """Returns (model, window size, user pose window size) of BP-GTNet or one of its ablations:
    'Ours', 'without_MD', 'without_PAD', 'without_EE_Dual'."""
    file_name, use_MD, use_PAD = BP_GTNET_CHECKPOINTS[kind]
    checkpoint = torch.load(osp.join(MODEL_DIR, file_name), map_location=device)
    c = checkpoint['config']
    model = BP_GTNet(rng=np.random.RandomState(23456),
                     num_experts=c['num_experts'],
                     num_experts_part=c['num_experts_part'],
                     window_size=c['window_size'],
                     user_pose_window_size=c['user_pose_window_size'],
                     layer_size_encoder=c['layer_size_encoder'],
                     input_size_encoder=c['input_size_encoder'],
                     head_codebook_channels=c['head_codebook_channels'],
                     head_codebook_dimension=c['head_codebook_dimension'],
                     head_temperature=c['head_temperature'],
                     hand_hidden_size=c['hand_hidden_size'],
                     hand_codebook_size=c['hand_codebook_size'],
                     hand_codebook_layer_size=c['hand_codebook_layer_size'],
                     model_gating=c['model_gating'],
                     layer_size_gating=c['layer_size_gating'],
                     input_size_gating=c['input_size_gating'],
                     hidden_size_gating=c['hidden_size_gating'],
                     input_size_moe=c['input_size_moe'],
                     use_cuda=c['use_cuda'],
                     use_MD=use_MD,
                     use_PAD=use_PAD)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, c['window_size'], c['user_pose_window_size']


def load_UGNet(device, window_size=30):
    """Returns the UGNet baseline: {'right': (MPNet, UGNet), 'left': (MPNet, UGNet)}."""
    models = {}
    for handedness in ('right', 'left'):
        mpnet = MPNet_MoE(rng=np.random.RandomState(23456), num_experts=9, input_size=6 + 5, hidden_size=24, output_size=3, use_cuda=True)
        mpnet.load_state_dict(torch.load(osp.join(MODEL_DIR, 'UGNet_MPNet_' + handedness + '_80.pt'), map_location=device))
        ugnet = UGNet_MoE(rng=np.random.RandomState(23456), num_experts=9, input_size=24 + (5 + 3), output_size=24, window_size=window_size, use_cuda=True)
        ugnet.load_state_dict(torch.load(osp.join(MODEL_DIR, 'UGNet_' + handedness + '_120.pt'), map_location=device))
        models[handedness] = (mpnet.to(device).eval(), ugnet.to(device).eval())
    return models


def load_AutoEncoder(device):
    checkpoint = torch.load(osp.join(MODEL_DIR, 'AutoEncoder_100.pt'), map_location=device)
    c = checkpoint['config']
    model = ConvolutionalAutoEncoder(input_size=c['input_size'], hidden_size=c['hidden_size'], kernel_size=c['kernel_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model
