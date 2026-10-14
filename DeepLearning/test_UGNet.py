"""Quantitative evaluation of the UGNet baseline (Table IV), our previous gesture translation network
(Kang et al., "Real-time Translation of Upper-body Gestures to Virtual Avatars in Dissimilar Telepresence
Environments", IEEE TVCG 2025). The metrics are the same as in test_BP_GTNet.py; the avatar is driven by the same
estimated targets ('Estimated Targets(N)/'), so both methods are compared under identical conditions.

UGNet consists of a motion progression network (MPNet) and a gesture network (UGNet), one pair per handedness; its
own data format is used (data(raw)/UGNet/test, 47 columns per frame).

    python test_UGNet.py --metric ALL --targetNumber 3
"""
import os
import gc
import argparse
import numpy as np
import torch
from torch.nn import functional as F

from evaluation_utils import (SUBJECTS_ALL, SUBJECTS_MAIN, REPERTORIES_DIP, repertories_for, is_right_handed, ensure_split,
                              user_clip_path, estimated_target_path, load_clip, fmt, pad_front, moving_average_bool,
                              consistency_confidence, sph2cart, sixd2matrot, target_relative_end_effectors, rot6_geodesic_deg,
                              MotionWindowWriter, fgd_of_writer, ik_touching_error, new_record, finalize_record, load_UGNet)

parser = argparse.ArgumentParser()
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

WINDOW_SIZE = 30
NETWORK = 'UGNet'


def hand_columns(IsRightHand):
    """Column offsets of the hand (forward, up, position, fingertip) in the user data and the target columns."""
    return (23 if IsRightHand else 35), (slice(3, 6) if IsRightHand else slice(6, 9))


def world_end_effectors(head_forward, head_up, head_position, hands):
    """Head frame + hands given in the head frame -> (B, 27) world end-effectors (forward, up, position) x 3."""
    head_rot = sixd2matrot(torch.cat([head_forward, head_up], dim=-1))
    out = [head_forward, head_up, head_position]
    for forward, up, position in hands:
        out += [(head_rot @ forward.unsqueeze(-1)).squeeze(-1),
                (head_rot @ up.unsqueeze(-1)).squeeze(-1),
                (head_rot @ position.unsqueeze(-1)).squeeze(-1) + head_position]
    return torch.cat(out, dim=-1)


def user_fingertip_world(User, IsRightHand):
    """World position of the user's index fingertip (T, 3) from the head frame and the hand (head-local) features."""
    hb, _ = hand_columns(IsRightHand)
    Head_Rot = sixd2matrot(User[:, 14:20])
    Hand_Rot = Head_Rot @ sixd2matrot(User[:, hb:hb + 6])
    Hand_Pos = (Head_Rot @ User[:, hb + 6:hb + 9].unsqueeze(-1)).squeeze(-1) + User[:, 20:23]
    return (Hand_Rot @ User[:, hb + 9:hb + 12].unsqueeze(-1)).squeeze(-1) + Hand_Pos


def recompute_angle_features(User, User_Target, IsRightHand, eps=1e-8):
    """Re-computes the five angle features User[:, 0:5] of MPNet with respect to the *estimated* targets."""
    _, tcol = hand_columns(IsRightHand)
    Head_Target = sph2cart(User_Target[:, 0:3])
    Hand_Target = sph2cart(User_Target[:, tcol])
    Head_Pos = User[:, 20:23]
    Head_Rot = sixd2matrot(User[:, 14:20])
    Gaze_World = (Head_Rot @ User[:, 11:14].unsqueeze(-1)).squeeze(-1)
    Fingertip_World = user_fingertip_world(User, IsRightHand)

    ET_Head = F.normalize(Head_Target - Head_Pos, dim=-1, eps=eps)
    ET_Hand = F.normalize(Hand_Target - Head_Pos, dim=-1, eps=eps)
    EF_Hand = F.normalize(Fingertip_World - Head_Pos, dim=-1, eps=eps)
    Gaze = F.normalize(Gaze_World, dim=-1, eps=eps)
    HF = F.normalize(User[:, 14:17], dim=-1, eps=eps)
    neg_up = torch.tensor([0.0, -1.0, 0.0], device=User.device).expand(User.shape[0], 3)

    def angle(a, b):
        return torch.acos(torch.clamp((a * b).sum(dim=-1), -1.0 + eps, 1.0 - eps))

    User[:, 0] = angle(ET_Head, Gaze)
    User[:, 1] = angle(ET_Head, HF)
    User[:, 2] = angle(ET_Hand, EF_Hand)
    User[:, 3] = angle(HF, EF_Hand)
    User[:, 4] = angle(EF_Hand, neg_up)
    return User, Fingertip_World, Hand_Target


class Runner(object):
    def __init__(self, models, device, writer=None):
        self.models = models
        self.device = device
        self.writer = writer
        self.W = WINDOW_SIZE
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
    def prepare(self, User, Avatar, IsRightHand):
        """Builds the network inputs of a clip from the user's data (T x 47) and the estimated targets (T x 39)."""
        User = User.to(self.device)
        Avatar = Avatar.to(self.device)
        S = {'T': User.shape[0],
             'Avatar_Target_GT': Avatar[:, 0:9],
             'Avatar_Target': Avatar[:, 9:18],
             'User_Target_GT': Avatar[:, 21:30],
             'User_Target': Avatar[:, 30:39],
             'Confidence_GT': consistency_confidence(Avatar[:, 0:9], Avatar[:, 18]).to(self.device)}
        Confidence = consistency_confidence(Avatar[:, 9:18], Avatar[:, 18]).to(self.device)
        Ray = moving_average_bool(Avatar[:, 18:21].clone(), 0, 3)
        User, user_tip, user_target = recompute_angle_features(User, S['User_Target'], IsRightHand)

        W = self.W
        S['User'] = pad_front(User, W).unsqueeze(0)
        S['Avatar_Target_p'] = pad_front(S['Avatar_Target'], W).unsqueeze(0)
        S['Avatar_Target_GT_p'] = pad_front(S['Avatar_Target_GT'], W).unsqueeze(0)
        S['Ray_p'] = pad_front(Ray, W).unsqueeze(0)
        S['Confidence_p'] = pad_front(Confidence, W)
        S['d_user_est_p'] = pad_front(torch.norm(user_target - user_tip, dim=1), W)
        return S

    def rollout(self, S, IsRightHand):
        """Autoregressive generation; yields (world end-effectors (27,) of the generated frame, blended output (1 x 24))."""
        MPNet, UGNet = self.models['right' if IsRightHand else 'left']
        User, Avatar_Target = S['User'], S['Avatar_Target_p']
        W = self.W
        hb, tcol = hand_columns(IsRightHand)
        ob = 35 if IsRightHand else 23      # the other hand, copied from the user

        Idle_Threshold, Gaze_Threshold, DeicticGesture_Threshold = 0.15, 0.25, 0.15
        Head_Target_Changing, Head_Target_Changing_Count = False, 0
        Hand_Target_Changing, Hand_Target_Changing_Count = False, 0
        Head_Blending_Count, Hand_Blending_Count = 0, 0
        Head_Blending_Window_Length, Hand_Blending_Window_Length = 20, 20

        Progression = torch.zeros(1, W, 3).to(self.device)
        Progression[:, :, -1] = 1
        X_Pose_Avatar = None
        for t in range(S['T']):
            progression = MPNet(User[:, t:W + t, :11])
            Progression = torch.cat((Progression[:, 1:, :], progression.unsqueeze(1)), 1)
            Gaze_Progression, DeicticGesture_Progression, Idle_Progression = Progression[0, -1, 0], Progression[0, -1, 1], Progression[0, -1, 2]

            if t == 0:
                X_Pose_Avatar = torch.cat((User[:, t + W - 1, 11:23], User[:, t + W - 1, hb:hb + 12]), -1)
            X_Pose_Avatar = torch.cat((torch.ones(1, W - 1, 24).to(self.device), X_Pose_Avatar.unsqueeze(1)), 1)
            X_Pose_Avatar = UGNet(torch.cat((X_Pose_Avatar,
                                             torch.cat((Avatar_Target[:, t:W + t, :2], Avatar_Target[:, t:W + t, tcol]), -1),
                                             Progression), -1))

            # generated frame (before blending): head + generated hand; the other hand is copied from the user
            cur = X_Pose_Avatar
            Uo = User[:, t + W - 1]
            gen_hand = (cur[:, 12:15], cur[:, 15:18], cur[:, 18:21])
            other_hand = (Uo[:, ob:ob + 3], Uo[:, ob + 3:ob + 6], Uo[:, ob + 6:ob + 9])
            hands = (gen_hand, other_hand) if IsRightHand else (other_hand, gen_hand)
            frame = world_end_effectors(cur[:, 3:6], cur[:, 6:9], cur[:, 9:12], hands).squeeze(0)

            # target changes
            if torch.allclose(Avatar_Target[:, W + t - 1, 0:2], Avatar_Target[:, W + t - 2, 0:2]) == False:
                Head_Target_Changing, Head_Target_Changing_Count = True, 0
            if Head_Target_Changing:
                if Head_Target_Changing_Count < 45:
                    Head_Target_Changing_Count += 1
                elif Head_Target_Changing_Count >= 45:
                    Head_Target_Changing, Head_Target_Changing_Count = False, 0
            if torch.allclose(Avatar_Target[:, W + t - 1, tcol], Avatar_Target[:, W + t - 2, tcol]) == False:
                Hand_Target_Changing, Hand_Target_Changing_Count = True, 0
            if Hand_Target_Changing:
                if Hand_Target_Changing_Count < 45:
                    Hand_Target_Changing_Count += 1
                elif Head_Target_Changing_Count >= 45:
                    Hand_Target_Changing, Hand_Target_Changing_Count = False, 0

            # blending toward the user's motion while idle
            if Idle_Progression < Idle_Threshold and Gaze_Progression < Gaze_Threshold:
                if not Head_Target_Changing and Head_Blending_Count < Head_Blending_Window_Length:
                    Head_Blending_Count += 1
            elif Head_Blending_Count > 0:
                Head_Blending_Count -= 1
            Head_Blending = Head_Blending_Count / Head_Blending_Window_Length
            if Idle_Progression < Idle_Threshold and DeicticGesture_Progression < DeicticGesture_Threshold:
                if not Hand_Target_Changing and Hand_Blending_Count < Hand_Blending_Window_Length:
                    Hand_Blending_Count += 1
            elif Hand_Blending_Count > 0:
                Hand_Blending_Count -= 1
            Hand_Blending = Hand_Blending_Count / Hand_Blending_Window_Length

            Un = User[:, W + t]
            X_Pose_Avatar[:, 0:3] = F.normalize(torch.lerp(X_Pose_Avatar[:, 0:3], Un[:, 11:14], Head_Blending), p=2, dim=1)
            X_Pose_Avatar[:, 3:6] = F.normalize(torch.lerp(X_Pose_Avatar[:, 3:6], Un[:, 14:17], Head_Blending), p=2, dim=1)
            X_Pose_Avatar[:, 6:9] = F.normalize(torch.lerp(X_Pose_Avatar[:, 6:9], Un[:, 17:20], Head_Blending), p=2, dim=1)
            X_Pose_Avatar[:, 9:12] = torch.lerp(X_Pose_Avatar[:, 9:12], Un[:, 20:23], Head_Blending)
            X_Pose_Avatar[:, 12:15] = F.normalize(torch.lerp(X_Pose_Avatar[:, 12:15], Un[:, hb:hb + 3], Hand_Blending), p=2, dim=1)
            X_Pose_Avatar[:, 15:18] = F.normalize(torch.lerp(X_Pose_Avatar[:, 15:18], Un[:, hb + 3:hb + 6], Hand_Blending), p=2, dim=1)
            X_Pose_Avatar[:, 18:21] = torch.lerp(X_Pose_Avatar[:, 18:21], Un[:, hb + 6:hb + 9], Hand_Blending)
            X_Pose_Avatar[:, 21:24] = torch.lerp(X_Pose_Avatar[:, 21:24], Un[:, hb + 9:hb + 12], Hand_Blending)
            yield frame, X_Pose_Avatar

    def real_end_effectors(self, S):
        U = S['User'][0, self.W:]
        return world_end_effectors(U[:, 14:17], U[:, 17:20], U[:, 20:23],
                                   ((U[:, 23:26], U[:, 26:29], U[:, 29:32]), (U[:, 35:38], U[:, 38:41], U[:, 41:44])))

    # ------------------------------------------------------------------------------------------------------------------
    def run_FGD(self, User, Avatar, IsRightHand):
        S = self.prepare(User, Avatar, IsRightHand)
        generated = torch.stack([frame for frame, _ in self.rollout(S, IsRightHand)])
        self.writer.write(self.real_end_effectors(S), generated)

    # ------------------------------------------------------------------------------------------------------------------
    def run_TMP(self, User, Avatar, IsRightHand):
        S = self.prepare(User, Avatar, IsRightHand)
        Avatar_EE_World = torch.stack([frame for frame, _ in self.rollout(S, IsRightHand)])
        User_EE_World = self.real_end_effectors(S)
        User_Target_GT, Avatar_Target_GT = S['User_Target_GT'], S['Avatar_Target_GT']
        User_EE_Target = target_relative_end_effectors(User_Target_GT, User_EE_World)
        Avatar_EE_Target = target_relative_end_effectors(Avatar_Target_GT, Avatar_EE_World)

        valid = torch.stack([~torch.all(User_Target_GT[:, 3 * p:3 * p + 3] == 0, dim=1) & (S['Confidence_GT'][:, p] == 1.0)
                             for p in range(3)], dim=1).float()
        rot_cols = [slice(0, 6), slice(9, 15), slice(18, 24)]
        pos_cols = [slice(6, 9), slice(15, 18), slice(24, 27)]
        U_rot6 = torch.cat([User_EE_Target[:, c] for c in rot_cols], dim=-1).unsqueeze(0)
        A_rot6 = torch.cat([Avatar_EE_Target[:, c] for c in rot_cols], dim=-1).unsqueeze(0)
        rot_diff = rot6_geodesic_deg(U_rot6, A_rot6)[0] * valid
        U_pos = torch.stack([User_EE_Target[:, c] for c in pos_cols], dim=1)
        A_pos = torch.stack([Avatar_EE_Target[:, c] for c in pos_cols], dim=1)
        pos_diff = torch.norm(U_pos - A_pos, dim=-1) * 100.0 * valid

        self.Position_Error_TMP += pos_diff.sum().item()
        self.Rotation_Error_TMP += rot_diff.sum().item()
        self.TMP_Count += valid.sum().clamp(min=1e-6).item()

    # ------------------------------------------------------------------------------------------------------------------
    def run_DIP(self, User, Avatar, IsRightHand, ispointing):
        S = self.prepare(User, Avatar, IsRightHand)
        W = self.W
        _, tcol = hand_columns(IsRightHand)
        flagcol, confcol = (1, 1) if IsRightHand else (2, 2)

        Pointing_Error, Touching_Error, n_eval = 100, 100, 0
        rec = new_record()
        zero = torch.zeros(3, device=self.device)
        for t, (_, X_Pose_Avatar) in enumerate(self.rollout(S, IsRightHand)):
            k = t + W - 1
            target_sph = S['Avatar_Target_GT_p'][:, k, tcol]
            valid = not torch.allclose(target_sph, zero.unsqueeze(0), atol=1e-6)
            Target_Position = sph2cart(target_sph)
            Head_Position = X_Pose_Avatar[:, 9:12]
            # fingertip: hand-local -> head-local -> world
            Head_Right = F.normalize(torch.cross(X_Pose_Avatar[:, 6:9], X_Pose_Avatar[:, 3:6], dim=1), dim=1)
            Head_Rot = torch.stack([Head_Right, X_Pose_Avatar[:, 6:9], X_Pose_Avatar[:, 3:6]], dim=2)
            Hand_Right = F.normalize(torch.cross(X_Pose_Avatar[:, 15:18], X_Pose_Avatar[:, 12:15], dim=1), dim=1)
            Hand_Rot = torch.stack([Hand_Right, X_Pose_Avatar[:, 15:18], X_Pose_Avatar[:, 12:15]], dim=2)
            Fingertip_HeadLocal = torch.bmm(Hand_Rot, X_Pose_Avatar[:, 21:24].unsqueeze(2)).squeeze(2) + X_Pose_Avatar[:, 18:21]
            Fingertip = torch.bmm(Head_Rot, Fingertip_HeadLocal.unsqueeze(2)).squeeze(2) + Head_Position
            flagged = S['Ray_p'][:, k, flagcol].item() == 1.0

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
                rec['tip_raw'].append(Fingertip[0].cpu().numpy())
                rec['target_gt'].append(Target_Position[0].cpu().numpy())
                rec['target_est'].append(sph2cart(S['Avatar_Target_p'][:, k, tcol])[0].cpu().numpy())
                rec['flag'].append(float(flagged))
                rec['conf'].append(S['Confidence_p'][k, confcol].item())
                rec['d_user_est'].append(S['d_user_est_p'][k].item())
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

    models = load_UGNet(device, WINDOW_SIZE)
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
                runner = Runner(models, device, writer)
                with torch.no_grad():
                    for subject, repertorie, i in clips(SUBJECTS_ALL, repertories_for):
                        if i == 1:
                            print('FGD,' + subject + ',' + repertorie, flush=True)
                        runner.run_FGD(load_clip(user_clip_path(NETWORK, subject, repertorie, i)),
                                       load_clip(estimated_target_path(TARGET_DIR, subject, repertorie, trial + 1, i)),
                                       is_right_handed(subject))
                    FGD.append(fgd_of_writer(writer, device))
                print("[Trial %d] [FGD: %.3f]" % (trial + 1, FGD[-1]), flush=True)
            results['FGD'] = FGD

        elif metric == 'TMP':
            TMP_PE, TMP_RE = [], []
            for trial in range(args.trial):
                runner = Runner(models, device)
                with torch.no_grad():
                    for subject, repertorie, i in clips(SUBJECTS_ALL, repertories_for):
                        if i == 1:
                            print('TMP,' + subject + ',' + repertorie, flush=True)
                        runner.run_TMP(load_clip(user_clip_path(NETWORK, subject, repertorie, i)),
                                       load_clip(estimated_target_path(TARGET_DIR, subject, repertorie, trial + 1, i)),
                                       is_right_handed(subject))
                TMP_PE.append(runner.Position_Error_TMP / runner.TMP_Count)
                TMP_RE.append(runner.Rotation_Error_TMP / runner.TMP_Count)
                print("[Trial %d] [Position_Error_TMP: %.3f] [Rotation_Error_TMP: %.3f]" % (trial + 1, TMP_PE[-1], TMP_RE[-1]), flush=True)
            results['TMP-PE'], results['TMP-RE'] = TMP_PE, TMP_RE

        elif metric == 'DIP':
            DIP_PE, DIP_TE, DIP_TE_IK = [], [], []
            for trial in range(args.trial):
                runner = Runner(models, device)
                with torch.no_grad():
                    for subject, repertorie, i in clips(SUBJECTS_MAIN, lambda s: REPERTORIES_DIP):
                        if i == 1:
                            print('DIP,' + subject + ',' + repertorie, flush=True)
                        ispointing = 'pointing' if repertorie == 'Pointing at single target' else 'touching'
                        runner.run_DIP(load_clip(user_clip_path(NETWORK, subject, repertorie, i)),
                                       load_clip(estimated_target_path(TARGET_DIR, subject, repertorie, trial + 1, i)),
                                       is_right_handed(subject), ispointing)
                DIP_PE.append(runner.Pointing_Error_DIP / runner.Pointing_Count)
                DIP_TE.append(runner.Touching_Error_DIP / runner.Touching_Count)
                DIP_TE_IK.append(runner.Touching_Error_DIP_IK / runner.Touching_Count)
                print("[Trial %d] [Pointing_Error_DIP: %.3f] [Touching_Error_DIP: %.3f] [Touching_Error_DIP (IK): %.3f]" % (
                    trial + 1, DIP_PE[-1], DIP_TE[-1], DIP_TE_IK[-1]), flush=True)
            results['DIP-PE'], results['DIP-TE'], results['DIP-TE (IK)'] = DIP_PE, DIP_TE, DIP_TE_IK

    print('')
    print('=== UGNet (Baseline), %d targets, %d trials ===' % (args.targetNumber, args.trial))
    for name, unit in (('FGD', ''), ('TMP-PE', ' [cm]'), ('TMP-RE', ' [deg]'), ('DIP-PE', ' [deg]'), ('DIP-TE', ' [cm]'), ('DIP-TE (IK)', ' [cm]')):
        if name in results:
            print('%-12s: %s%s' % (name, fmt(results[name], 3 if name == 'FGD' else 2), unit))
