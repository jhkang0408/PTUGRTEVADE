"""Quantitative evaluation of BP-GTNet and its ablations (Table IV), plus DIP-TE after the IK post-processing.

Metrics (see Section V.B of the paper)
    FGD    : Frechet gesture distance between the generated and the real end-effector motions (movement naturalness)
    TMP    : target-view motion preservation, position [cm] and rotation [deg] errors of the end-effectors expressed
             in the frame of their targets
    DIP    : deictic intention preservation, pointing error DIP-PE [deg] and touching error DIP-TE [cm] of the avatar's
             index fingertip with respect to the ground-truth target. DIP-TE is also reported after the IK
             post-processing of the real-time system (the fingertip is pulled to the estimated target while the user
             keeps touching it; see evaluation_utils.ik_touching_error).

The avatar is driven by the estimated targets in 'Estimated Targets(N)/' (generate_estimated_targets.py), one
environment per trial; mean +- std over the trials is reported.

    python test_BP_GTNet.py --model Ours            --metric ALL --targetNumber 3
    python test_BP_GTNet.py --model without_MD      --metric DIP --targetNumber 5
    python test_BP_GTNet.py --model without_PAD     --metric TMP --targetNumber 7
    python test_BP_GTNet.py --model without_EE_Dual --metric FGD --targetNumber 3   # needs data(raw)/BP_GTNet_without_EE_Dual
"""
import os
import gc
import argparse
import numpy as np
import torch
from torch.nn import functional as F

from evaluation_utils import (SUBJECTS_ALL, SUBJECTS_MAIN, REPERTORIES_DIP, repertories_for, is_right_handed, ensure_split,
                              user_clip_path, estimated_target_path, load_clip, fmt, pad_front, moving_average_bool,
                              consistency_confidence, sph2cart, fingertip_world, recompute_head_features_wrt_target,
                              target_relative_end_effectors, rot6_geodesic_deg, MotionWindowWriter, fgd_of_writer,
                              ik_touching_error, new_record, finalize_record, load_BP_GTNet)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Ours', choices=['Ours', 'without_MD', 'without_PAD', 'without_EE_Dual'], help='BP-GTNet variant')
parser.add_argument('--metric', default='ALL', choices=['FGD', 'TMP', 'DIP', 'ALL'], help='Metric')
parser.add_argument('--targetNumber', type=int, default=3, help='Number of targets (3, 5 or 7)')
parser.add_argument('--trial', type=int, default=10, help='Number of trials (avatar environments)')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--targets', default='Estimated Targets', help='Prefix of the estimated-target folder')
parser.add_argument('--rawDataSplit', default=False, type=lambda x: x.lower() in ['true', '1'], help='Re-split raw/Data.txt into clips')
parser.add_argument('--ik_threshold', type=float, default=0.10, help='IK post-processing: distance threshold [m] between the user fingertip and the estimated target')
parser.add_argument('--ik_rate', type=float, default=0.03, help='IK post-processing: IK weight increment per frame (0.01 per rendered frame at 90 fps = 0.03 per 30-fps frame)')
parser.add_argument('--ik_confidence', type=float, default=0.75, help='IK post-processing: confidence threshold of the hand target')
args = parser.parse_args()


class Runner(object):
    def __init__(self, model, window_size, user_pose_window_size, device, head_target_transform, writer=None):
        self.model = model
        self.W = window_size
        self.P = user_pose_window_size
        self.device = device
        self.head_target_transform = head_target_transform
        self.writer = writer
        # TMP
        self.Position_Error_TMP = 0
        self.Rotation_Error_TMP = 0
        self.TMP_Count = 0
        # DIP
        self.Pointing_Error_DIP = 0
        self.Touching_Error_DIP = 0
        self.Touching_Error_DIP_IK = 0
        self.Pointing_Count = 0
        self.Touching_Count = 0

    # ------------------------------------------------------------------------------------------------------------------
    def prepare(self, User, Avatar):
        """Builds the network inputs of a clip from the user's data (T x 78) and the estimated targets (T x 39)."""
        User = User.to(self.device)
        Avatar = Avatar.to(self.device)
        S = {'T': User.shape[0],
             'User_Target_GT': User[:, 0:9].clone(),
             'Avatar_Target_GT': Avatar[:, 0:9],
             'Avatar_Target': Avatar[:, 9:18],
             'User_Target': Avatar[:, 30:39]}

        User[:, 9:12] = Avatar[:, 18:21]                                   # ray features of the candidate set
        if self.head_target_transform:                                      # dual end-effector representation of the head
            User = recompute_head_features_wrt_target(User, Avatar[:, 30:33])
        S['Confidence_GT'] = consistency_confidence(Avatar[:, 0:9], User[:, 9]).to(self.device)
        Confidence = consistency_confidence(Avatar[:, 9:18], User[:, 9]).to(self.device)
        User = moving_average_bool(User, 9, 6)

        S['User'] = pad_front(User, self.W).unsqueeze(0)
        S['Avatar_Target_p'] = pad_front(S['Avatar_Target'], self.W).unsqueeze(0)
        S['Avatar_Target_GT_p'] = pad_front(S['Avatar_Target_GT'], self.W).unsqueeze(0)
        S['User_Target_p'] = pad_front(S['User_Target'], self.W).unsqueeze(0)
        S['Confidence_p'] = pad_front(Confidence, self.W).unsqueeze(0)
        return S

    def rollout(self, S):
        """Autoregressive generation of the avatar's end-effectors; yields the network output (1 x 36) per frame."""
        User, Avatar_Target, Confidence = S['User'], S['Avatar_Target_p'], S['Confidence_p']
        W, P = self.W, self.P
        X_Pose_Avatar = None
        for t in range(S['T']):
            X_Target_User = Avatar_Target[:, t:t + W, 0:3]
            X_Target_Avatar = torch.cat((Avatar_Target[:, t:t + W], User[:, t:t + W, 9:15]), -1)
            X_Confidence = Confidence[:, t:t + W, :]
            X_Pose_User = User[:, t + W - P:t + W, 15:42]
            X_Pose_Avatar_input = User[:, t + W - 1, 42:78] if t == 0 else X_Pose_Avatar

            X_Pose_Avatar = self.model(X_Target_User, X_Target_Avatar, X_Pose_User, X_Pose_Avatar_input, X_Confidence, False)
            if t != 0:
                for a, b in ((0, 3), (3, 6), (9, 12), (12, 15), (15, 18), (24, 27), (27, 30)):   # unit vectors
                    X_Pose_Avatar[:, a:b] = F.normalize(X_Pose_Avatar[:, a:b], dim=-1)
            yield X_Pose_Avatar

    @staticmethod
    def end_effectors(X_Pose_Avatar):
        """(forward, up, position) of the head, right hand and left hand -> (27,)"""
        return torch.cat((X_Pose_Avatar[0, 0:9], X_Pose_Avatar[0, 12:21], X_Pose_Avatar[0, 24:33]), -1)

    # ------------------------------------------------------------------------------------------------------------------
    def run_FGD(self, User, Avatar):
        S = self.prepare(User, Avatar)
        generated = torch.stack([self.end_effectors(out) for out in self.rollout(S)])
        U = S['User'][0, self.W:]
        real = torch.cat((U[:, 42:51], U[:, 54:63], U[:, 66:75]), -1)
        self.writer.write(real, generated)

    # ------------------------------------------------------------------------------------------------------------------
    def run_TMP(self, User, Avatar):
        S = self.prepare(User, Avatar)
        Avatar_EE_World = torch.stack([self.end_effectors(out) for out in self.rollout(S)])
        U = S['User'][0, self.W:]
        User_EE_World = torch.cat((U[:, 42:51], U[:, 54:63], U[:, 66:75]), -1)

        User_Target_GT, Avatar_Target_GT = S['User_Target_GT'], S['Avatar_Target_GT']
        User_EE_Target = target_relative_end_effectors(User_Target_GT, User_EE_World)
        Avatar_EE_Target = target_relative_end_effectors(Avatar_Target_GT, Avatar_EE_World)

        # valid frames per part: a ground-truth target exists and its confidence has saturated
        valid = torch.stack([~torch.all(User_Target_GT[:, 3 * p:3 * p + 3] == 0, dim=1) & (S['Confidence_GT'][:, p] == 1.0)
                             for p in range(3)], dim=1).float()                                  # (T, 3)

        rot_cols = [slice(0, 6), slice(9, 15), slice(18, 24)]
        pos_cols = [slice(6, 9), slice(15, 18), slice(24, 27)]
        U_rot6 = torch.cat([User_EE_Target[:, c] for c in rot_cols], dim=-1).unsqueeze(0)
        A_rot6 = torch.cat([Avatar_EE_Target[:, c] for c in rot_cols], dim=-1).unsqueeze(0)
        rot_diff = rot6_geodesic_deg(U_rot6, A_rot6)[0] * valid                                 # (T, 3) [deg]
        U_pos = torch.stack([User_EE_Target[:, c] for c in pos_cols], dim=1)
        A_pos = torch.stack([Avatar_EE_Target[:, c] for c in pos_cols], dim=1)
        pos_diff = torch.norm(U_pos - A_pos, dim=-1) * 100.0 * valid                             # (T, 3) [cm]

        self.Position_Error_TMP += pos_diff.sum().item()
        self.Rotation_Error_TMP += rot_diff.sum().item()
        self.TMP_Count += valid.sum().item()

    # ------------------------------------------------------------------------------------------------------------------
    def run_DIP(self, User, Avatar, handedness, ispointing):
        S = self.prepare(User, Avatar)
        W = self.W
        if handedness == 'right':
            tcol, flagcol, confcol = slice(3, 6), 10, 1
            dec = (54, 57, 60, 63)      # user (decoder features): hand forward, up, position, fingertip (hand-local)
            out = (12, 15, 18, 21)      # avatar output          : hand forward, up, position, fingertip (hand-local)
        else:
            tcol, flagcol, confcol = slice(6, 9), 11, 2
            dec = (66, 69, 72, 75)
            out = (24, 27, 30, 33)

        Pointing_Error, Touching_Error, n_eval = 100, 100, 0
        rec = new_record()
        zero = torch.zeros(3, device=self.device)
        for t, X_Pose_Avatar in enumerate(self.rollout(S)):
            k = t + W - 1
            target_sph = S['Avatar_Target_GT_p'][:, k, tcol]
            valid = not torch.allclose(target_sph, zero.unsqueeze(0), atol=1e-6)
            Target_Position = sph2cart(target_sph)
            Head_Position = X_Pose_Avatar[:, 6:9]
            f, u, p, l = out
            Fingertip = fingertip_world(X_Pose_Avatar[:, f:f + 3], X_Pose_Avatar[:, u:u + 3], X_Pose_Avatar[:, p:p + 3], X_Pose_Avatar[:, l:l + 3])
            flagged = S['User'][:, k, flagcol].item() == 1.0

            if valid and flagged:
                if ispointing == 'pointing':
                    dir_to_target = F.normalize(Target_Position - Head_Position, dim=1)
                    dir_to_fingertip = F.normalize(Fingertip - Head_Position, dim=1)
                    cos_angle = torch.clamp((dir_to_target * dir_to_fingertip).sum(dim=1), -1.0 + 1e-6, 1.0 - 1e-6)
                    Pointing_Error = min(Pointing_Error, torch.rad2deg(torch.acos(cos_angle)).item())
                else:
                    Touching_Error = min(Touching_Error, (torch.norm(Target_Position - Fingertip, dim=1) * 100).item())
                n_eval += 1

            if ispointing != 'pointing':
                # per-frame record for the IK post-processing (user side: fingertip vs. estimated target)
                f, u, p, l = dec
                Uk = S['User'][:, k]
                user_tip = fingertip_world(Uk[:, f:f + 3], Uk[:, u:u + 3], Uk[:, p:p + 3], Uk[:, l:l + 3])
                user_target = sph2cart(S['User_Target_p'][:, k, tcol])
                rec['tip_raw'].append(Fingertip[0].cpu().numpy())
                rec['target_gt'].append(Target_Position[0].cpu().numpy())
                rec['target_est'].append(sph2cart(S['Avatar_Target_p'][:, k, tcol])[0].cpu().numpy())
                rec['flag'].append(float(flagged))
                rec['conf'].append(S['Confidence_p'][0, k, confcol].item())
                rec['d_user_est'].append(torch.norm(user_target - user_tip, dim=1).item())
                rec['valid'].append(float(valid))

        if n_eval == 0:
            print("skip!")
            return
        if ispointing == 'pointing':
            self.Pointing_Error_DIP += Pointing_Error
            self.Pointing_Count += 1
        else:
            self.Touching_Error_DIP += Touching_Error
            self.Touching_Error_DIP_IK += ik_touching_error(finalize_record(rec), args.ik_threshold, args.ik_rate, args.ik_confidence)[1]
            self.Touching_Count += 1


def clips(subjects, repertories_of):
    for subject in subjects:
        for repertorie in repertories_of(subject):
            for i in range(repertorie['num']):
                yield subject, repertorie['repertorie'], i + 1


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model, WINDOW_SIZE, USER_POSE_WINDOW_SIZE = load_BP_GTNet(args.model, device)
    NETWORK = 'BP_GTNet' if args.model != 'without_EE_Dual' else 'BP_GTNet_without_EE_Dual'
    HEAD_TARGET_TRANSFORM = args.model != 'without_EE_Dual'
    TARGET_DIR = args.targets + '(' + str(args.targetNumber) + ')'
    ensure_split(NETWORK, SUBJECTS_ALL, WINDOW_SIZE, force=args.rawDataSplit)

    metrics = ['FGD', 'TMP', 'DIP'] if args.metric == 'ALL' else [args.metric]
    results = {}
    for metric in metrics:
        print('\n' + '=' * 50 + '\nRunning metric: ' + metric + '\n' + '=' * 50 + '\n')

        if metric == 'FGD':
            FGD = []
            for trial in range(args.trial):
                writer = MotionWindowWriter()
                runner = Runner(model, WINDOW_SIZE, USER_POSE_WINDOW_SIZE, device, HEAD_TARGET_TRANSFORM, writer)
                with torch.no_grad():
                    for subject, repertorie, i in clips(SUBJECTS_ALL, repertories_for):
                        if i == 1:
                            print('FGD,' + subject + ',' + repertorie, flush=True)
                        runner.run_FGD(load_clip(user_clip_path(NETWORK, subject, repertorie, i)),
                                       load_clip(estimated_target_path(TARGET_DIR, subject, repertorie, trial + 1, i)))
                    FGD.append(fgd_of_writer(writer, device))
                print("[Trial %d] [FGD: %.3f]" % (trial + 1, FGD[-1]), flush=True)
            results['FGD'] = FGD

        elif metric == 'TMP':
            TMP_PE, TMP_RE = [], []
            for trial in range(args.trial):
                runner = Runner(model, WINDOW_SIZE, USER_POSE_WINDOW_SIZE, device, HEAD_TARGET_TRANSFORM)
                with torch.no_grad():
                    for subject, repertorie, i in clips(SUBJECTS_ALL, repertories_for):
                        if i == 1:
                            print('TMP,' + subject + ',' + repertorie, flush=True)
                        runner.run_TMP(load_clip(user_clip_path(NETWORK, subject, repertorie, i)),
                                       load_clip(estimated_target_path(TARGET_DIR, subject, repertorie, trial + 1, i)))
                TMP_PE.append(runner.Position_Error_TMP / runner.TMP_Count)
                TMP_RE.append(runner.Rotation_Error_TMP / runner.TMP_Count)
                print("[Trial %d] [Position_Error_TMP: %.3f] [Rotation_Error_TMP: %.3f]" % (trial + 1, TMP_PE[-1], TMP_RE[-1]), flush=True)
            results['TMP-PE'], results['TMP-RE'] = TMP_PE, TMP_RE

        elif metric == 'DIP':
            DIP_PE, DIP_TE, DIP_TE_IK = [], [], []
            for trial in range(args.trial):
                runner = Runner(model, WINDOW_SIZE, USER_POSE_WINDOW_SIZE, device, HEAD_TARGET_TRANSFORM)
                with torch.no_grad():
                    for subject, repertorie, i in clips(SUBJECTS_MAIN, lambda s: REPERTORIES_DIP):
                        if i == 1:
                            print('DIP,' + subject + ',' + repertorie, flush=True)
                        handedness = 'right' if is_right_handed(subject) else 'left'
                        ispointing = 'pointing' if repertorie == 'Pointing at single target' else 'touching'
                        runner.run_DIP(load_clip(user_clip_path(NETWORK, subject, repertorie, i)),
                                       load_clip(estimated_target_path(TARGET_DIR, subject, repertorie, trial + 1, i)),
                                       handedness, ispointing)
                DIP_PE.append(runner.Pointing_Error_DIP / runner.Pointing_Count)
                DIP_TE.append(runner.Touching_Error_DIP / runner.Touching_Count)
                DIP_TE_IK.append(runner.Touching_Error_DIP_IK / runner.Touching_Count)
                print("[Trial %d] [Pointing_Error_DIP: %.3f] [Touching_Error_DIP: %.3f] [Touching_Error_DIP (IK): %.3f]" % (
                    trial + 1, DIP_PE[-1], DIP_TE[-1], DIP_TE_IK[-1]), flush=True)
            results['DIP-PE'], results['DIP-TE'], results['DIP-TE (IK)'] = DIP_PE, DIP_TE, DIP_TE_IK

    print('')
    print('=== BP-GTNet (%s), %d targets, %d trials ===' % (args.model, args.targetNumber, args.trial))
    for name, unit in (('FGD', ''), ('TMP-PE', ' [cm]'), ('TMP-RE', ' [deg]'), ('DIP-PE', ' [deg]'), ('DIP-TE', ' [cm]'), ('DIP-TE (IK)', ' [cm]')):
        if name in results:
            print('%-12s: %s%s' % (name, fmt(results[name], 3 if name == 'FGD' else 2), unit))
