import os
import gc
import time
import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import TENet
from model_comparison_TENet import TENet_Baseline, TENet_Indep
from data_parser import data_parser
from data_loader import Dataset

class Runner(object):                            
    def __init__(self, device, model, lr):               
        self.device = device
        self.model = model.to(self.device)
        if OPTIMIZER=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        elif OPTIMIZER=='AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        if SCHEDULER=='ExpLR':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=GAMMA)
        elif SCHEDULER=='CosineWR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=0)
        self.learning_rate = lr
        self.iter = 0
                
    def run(self, dataloader, epoch, mode='TRAIN'):
        if mode == 'TRAIN':
            self.model.train()

        epoch_loss = 0
                        
        pbar = tqdm(dataloader)
        for iter, data in enumerate(pbar):
            DATA = data['data'].to(self.device)                              
            
            X = DATA[:, :, :INPUT_SIZE]
            Y = DATA[:, -1, -OUTPUT_SIZE:]                                
            _, _, log_prob = self.model(X, Y)
            loss = -log_prob.mean()
                                
            if mode=='TRAIN':
                loss.backward()
                if GRADIENT_CLIPPING != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIPPING)                
                self.optimizer.step()
                self.optimizer.zero_grad()

            if iter % 50 == 0:
                if mode == 'TRAIN':
                    writer.add_scalar("Train", loss.item(), self.iter)
                    self.iter += 1
            
            pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, iter+1, len(dataloader)))
            pbar.set_postfix({"loss":((loss.item()))})                       
            
            epoch_loss += loss.item()
                
        epoch_loss /= len(dataloader)
        
        self.scheduler.step()

        return epoch_loss

import os.path as osp
from torch.nn import functional as F

class Evaluator(object):
    """Measure the target-estimation accuracy on the test-split data every epoch.
    Reuses the Runner.run logic from test_TENet.py as-is, but uses an eval-only batch size."""
    def __init__(self, device, model, test_batch_size):
        self.device = device
        self.model = model
        self.test_batch_size = test_batch_size
        self.reset()

    def reset(self):
        self.count = 0
        self.count_head = 0
        self.count_righthand = 0
        self.count_lefthand = 0
        self.iter = 0
        # For accuracy restricted to frames where the GT target is not null(0) (i.e., a target exists)
        self.count_head_nn = 0
        self.count_righthand_nn = 0
        self.count_lefthand_nn = 0
        self.iter_head_nn = 0
        self.iter_righthand_nn = 0
        self.iter_lefthand_nn = 0

    def run(self, data_TENet, repertorie, target_number):
        data_TENet = data_TENet.to(self.device)
        target_TENet_GT = data_TENet[:, 12:21]
        targets_TENet = []
        FAKE_TARGET_NUMBER = 0
        if repertorie == 'Pointing and gazing at a single target' or\
           repertorie == 'Touching and gazing at a single target':
            FAKE_TARGET_NUMBER = target_number - 2
            targets_TENet.append(data_TENet[:, 21:24])
        elif repertorie == 'Explaning with pointing' or\
             repertorie == 'Pointing at a single target with gaze shift' or\
             repertorie == 'Pointing and gazing at two targets in sequence' or\
             repertorie == 'Explaning with touching' or\
             repertorie == 'Touching a single target with gaze shift' or\
             repertorie == 'Touching and gazing at two targets in sequence':
            FAKE_TARGET_NUMBER = target_number - 3
            targets_TENet.append(data_TENet[:, 21:24])
            targets_TENet.append(data_TENet[:, 24:27])
        elif repertorie == 'Referencing multiple targets with diverse gestures (pointing)' or\
             repertorie == 'Referencing multiple targets with diverse gestures (touching)':
            FAKE_TARGET_NUMBER = target_number - 4
            targets_TENet.append(data_TENet[:, 21:24])
            targets_TENet.append(data_TENet[:, 24:27])
            targets_TENet.append(data_TENet[:, 27:30])

        for i in range(FAKE_TARGET_NUMBER):
            horizontal_angle = torch.rand(1, device=self.device) * torch.pi - (torch.pi/2)  # -π/2 ~ π/2
            vertical_angle = torch.rand(1, device=self.device) * (2*torch.pi/3) - (torch.pi/3)  # -π/3 ~ π/3
            if 'pointing' in repertorie.lower():
                distance = torch.ones(1, device=self.device)
            else:
                distance = torch.rand(1, device=self.device) * 0.4 + 0.4
            horizontal_angle = horizontal_angle.expand(data_TENet.shape[0])
            vertical_angle = vertical_angle.expand(data_TENet.shape[0])
            distance = distance.expand(data_TENet.shape[0])
            target_TENet = torch.stack([horizontal_angle, vertical_angle, distance], dim=1)
            targets_TENet.append(target_TENet)
        targets_TENet.append(torch.zeros(data_TENet.shape[0], 3, device=self.device))

        # default feature
        first_data_TENet = data_TENet[0:1].repeat(15, 1)
        data_TENet = torch.cat([first_data_TENet, data_TENet], dim=0)
        original_bool_data = data_TENet[:, 9:12].clone()
        continuous_bool_data = torch.zeros_like(original_bool_data)
        for feature_idx in range(3):
            cum_sum = torch.cumsum(data_TENet[:, 9+feature_idx], dim=0)
            continuous_bool_data[15:, feature_idx] = (cum_sum[15:] - cum_sum[:-15]) / 15
        data_TENet[:, 9:12] = continuous_bool_data
        data_TENet = data_TENet[15:]

        # Padding
        first_data_TENet = data_TENet[0:1].repeat(self.model.window_dim-1, 1)
        data_TENet = torch.cat([first_data_TENet, data_TENet], dim=0)
        padded_targets_TENet = []
        for i in range(target_number):
            first_target_TENet = targets_TENet[i][0:1].repeat(self.model.window_dim-1, 1)
            padded_target_TENet = torch.cat([first_target_TENet, targets_TENet[i]], dim=0)
            padded_targets_TENet.append(padded_target_TENet)
        targets_TENet = padded_targets_TENet

        # Compute the number of combinations and prepare indices
        num_combinations = target_number ** 3
        combination_indices = torch.arange(num_combinations, device=self.device)
        head_divisor = target_number ** 2
        right_divisor = target_number
        head_indices = (combination_indices // head_divisor) % target_number
        right_indices = (combination_indices // right_divisor) % target_number
        left_indices = combination_indices % target_number

        # Prepare the input and output tensors
        total_length = data_TENet.shape[0]
        effective_length = total_length - self.model.window_dim + 1
        batch_size = self.test_batch_size

        max_indices = torch.zeros(effective_length, dtype=torch.long, device=self.device)
        confidences = torch.zeros(effective_length, device=self.device)

        all_targets_TENet = torch.stack([targets_TENet[i][self.model.window_dim-1:total_length] for i in range(target_number)], dim=0)

        for t_start in range(0, effective_length, batch_size):
            t_end = min(t_start + batch_size, effective_length)
            batch_len = t_end - t_start

            batch_targets = all_targets_TENet[:, t_start:t_end]  # [num_targets, batch_len, 3]
            X_windows = torch.stack([data_TENet[t:t+self.model.window_dim, :12] for t in range(t_start, t_end)], dim=0)

            batch_indices = torch.arange(batch_len, device=self.device)
            comb_indices = torch.arange(num_combinations, device=self.device)
            batch_grid, comb_grid = torch.meshgrid(batch_indices, comb_indices, indexing='ij')
            flat_batch_indices = batch_grid.reshape(-1)
            flat_comb_indices = comb_grid.reshape(-1)

            head_idx = head_indices[flat_comb_indices].clamp(0, target_number-1)
            right_idx = right_indices[flat_comb_indices].clamp(0, target_number-1)
            left_idx = left_indices[flat_comb_indices].clamp(0, target_number-1)

            X_batch = X_windows.repeat_interleave(num_combinations, dim=0)

            Y_batch = torch.zeros((batch_len * num_combinations, 9), device=self.device)
            Y_batch[:, 0:3] = batch_targets[head_idx, flat_batch_indices]
            Y_batch[:, 3:6] = batch_targets[right_idx, flat_batch_indices]
            Y_batch[:, 6:9] = batch_targets[left_idx, flat_batch_indices]

            log_probs = self.model.posterior_probability(X_batch, Y_batch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log_probs = log_probs.reshape(batch_len, num_combinations)

            probs = F.softmax(log_probs, dim=1)
            batch_max_probs, batch_max_indices = torch.max(probs, dim=1)

            max_indices[t_start:t_end] = batch_max_indices
            confidences[t_start:t_end] = batch_max_probs

            del batch_targets, X_windows, batch_grid, comb_grid
            del flat_batch_indices, flat_comb_indices
            del head_idx, right_idx, left_idx
            del X_batch, Y_batch, log_probs, probs
            del batch_max_probs, batch_max_indices

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        head_indices_full = (max_indices // head_divisor) % target_number
        right_indices_full = (max_indices // right_divisor) % target_number
        left_indices_full = max_indices % target_number

        time_indices = torch.arange(effective_length, device=self.device)
        head_target_pred_tensor = all_targets_TENet[head_indices_full, time_indices]
        righthand_target_pred_tensor = all_targets_TENet[right_indices_full, time_indices]
        lefthand_target_pred_tensor = all_targets_TENet[left_indices_full, time_indices]
        target_TENet_pred = torch.cat((head_target_pred_tensor, righthand_target_pred_tensor, lefthand_target_pred_tensor), -1)

        head_diff = torch.norm(target_TENet_GT[:, 0:3] - target_TENet_pred[:, 0:3], dim=1)
        right_diff = torch.norm(target_TENet_GT[:, 3:6] - target_TENet_pred[:, 3:6], dim=1)
        left_diff = torch.norm(target_TENet_GT[:, 6:9] - target_TENet_pred[:, 6:9], dim=1)

        head_is_correct = (head_diff <= 1e-2)
        right_is_correct = (right_diff <= 1e-2)
        left_is_correct = (left_diff <= 1e-2)

        head_correct = head_is_correct.sum().item()
        right_correct = right_is_correct.sum().item()
        left_correct = left_is_correct.sum().item()

        self.iter += effective_length * 3
        self.count += head_correct + right_correct + left_correct
        self.count_head += head_correct
        self.count_righthand += right_correct
        self.count_lefthand += left_correct

        # Aggregate separately only the frames where the GT target is not null(0,0,0)
        head_nonnull = torch.any(target_TENet_GT[:, 0:3] != 0, dim=1)
        right_nonnull = torch.any(target_TENet_GT[:, 3:6] != 0, dim=1)
        left_nonnull = torch.any(target_TENet_GT[:, 6:9] != 0, dim=1)
        self.count_head_nn += (head_is_correct & head_nonnull).sum().item()
        self.count_righthand_nn += (right_is_correct & right_nonnull).sum().item()
        self.count_lefthand_nn += (left_is_correct & left_nonnull).sum().item()
        self.iter_head_nn += head_nonnull.sum().item()
        self.iter_righthand_nn += right_nonnull.sum().item()
        self.iter_lefthand_nn += left_nonnull.sum().item()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Ours', help='Model')
parser.add_argument('--inputRepresentation', default='', help='Input Representation')
parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--directory_id', type=int, default=0, help='Directory ID')
parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--epochs', type=int, default=100, help='Epochs')
parser.add_argument('--batchSize', type=int, default=16, help='Batch Size')
parser.add_argument('--windowSizeTENet', type=int, default=45, help='Window Size (TENet)')
parser.add_argument('--headDepth', type=int, default=2, help='Head Depth')
parser.add_argument('--headSz', type=int, default=128, help='Head Sz')
parser.add_argument('--featSz', type=int, default=128, help='Feature Sz')
parser.add_argument('--depth', type=int, default=3, help='Depth')
parser.add_argument('--conditionSize', type=int, default=256, help='Condition Size')
parser.add_argument('--flowLayerSize', type=int, default=5, help='Flow Layer Size')
parser.add_argument('--encoderLayerSize', type=int, default=6, help='Encoder Layer Size')
parser.add_argument('--encoderHeadSize', type=int, default=8, help='Encoder Head Size')
parser.add_argument('--encoderDropout', type=float, default=0, help='Encoder Dropout')
parser.add_argument('--useNullToken', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--optimizer', default='Adam', help='Optimizer')
parser.add_argument('--scheduler', default='ExpLR', help='Scheduler')
parser.add_argument('--gamma', type=float, default=0.9, help='Gamma')
parser.add_argument('--weightDecay', type=float, default=0, help='Weight Decay')
parser.add_argument('--gradientClipping', type=float, default=0, help='Gradient Clipping')
parser.add_argument('--rawDataSplit', default=False, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--testBatchSize', type=int, default=128, help='Test Batch Size (eval window batching, independent of train batch size)')
parser.add_argument('--testTargetNumber', type=int, default=3, help='Max number of real targets used at eval time (candidates = testTargetNumber + 1 null)')
args = parser.parse_args()

if __name__ == '__main__':
    script_start = time.time()

    gc.collect()
    torch.cuda.empty_cache()
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%m%d_%H%M%S')

    MODEL = args.model
    INPUT_REPRESENTATION = args.inputRepresentation    
    GPU_ID = args.gpu_id
    DIRECTORY_ID = args.directory_id
    SHUFFLE = args.shuffle        
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchSize    
    WINDOW_SIZE_TENet = args.windowSizeTENet
    CLIP_SIZE = 0
    HEAD_DEPTH = args.headDepth
    HEAD_SZ = args.headSz
    FEAT_SZ = args.featSz
    DEPTH = args.depth
    CONDITION_SIZE = args.conditionSize    
    FLOW_LAYER_SIZE = args.flowLayerSize
    ENCODER_LAYER_SIZE = args.encoderLayerSize
    ENCODER_HEAD_SIZE = args.encoderHeadSize
    ENCODER_DROPOUT = args.encoderDropout
    USE_NULL_TOKEN = args.useNullToken
    LR = args.lr    
    OPTIMIZER = args.optimizer
    SCHEDULER = args.scheduler
    GAMMA = args.gamma    
    WEIGHT_DECAY = args.weightDecay
    GRADIENT_CLIPPING = args.gradientClipping
    TEST_BATCH_SIZE = args.testBatchSize
    TEST_TARGET_NUMBER = args.testTargetNumber

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  
        
    INPUT_SIZE = 12
    OUTPUT_SIZE = 9
    
    if MODEL == 'Ours':
        model = TENet(head_depth=HEAD_DEPTH,
                      head_sz=HEAD_SZ,
                      feat_sz=FEAT_SZ,
                      depth=DEPTH,
                      condition_dim=CONDITION_SIZE,        
                      input_dim=INPUT_SIZE,
                      window_dim=WINDOW_SIZE_TENet,
                      target_dim=OUTPUT_SIZE,
                      n_flow_layers=FLOW_LAYER_SIZE,
                      n_encoder_layers=ENCODER_LAYER_SIZE,
                      n_encoder_head=ENCODER_HEAD_SIZE,
                      encoder_dropout=ENCODER_DROPOUT,
                      use_null_token=USE_NULL_TOKEN)
    elif MODEL == 'Baseline':
        model = TENet_Baseline(head_depth=HEAD_DEPTH,
                               head_sz=HEAD_SZ,
                               feat_sz=FEAT_SZ,
                               depth=DEPTH,
                               condition_dim=CONDITION_SIZE,        
                               input_dim=INPUT_SIZE,
                               window_dim=WINDOW_SIZE_TENet,
                               target_dim=OUTPUT_SIZE,
                               n_flow_layers=FLOW_LAYER_SIZE,
                               n_encoder_layers=ENCODER_LAYER_SIZE,
                               n_encoder_head=ENCODER_HEAD_SIZE,
                               encoder_dropout=ENCODER_DROPOUT) 
    elif MODEL == 'Ours_Indep':
        model = TENet_Indep(head_depth=HEAD_DEPTH,
                            head_sz=HEAD_SZ,
                            feat_sz=FEAT_SZ,
                            depth=DEPTH,
                            condition_dim=CONDITION_SIZE,        
                            input_dim=INPUT_SIZE,
                            window_dim=WINDOW_SIZE_TENet,
                            target_dim=OUTPUT_SIZE,
                            n_flow_layers=FLOW_LAYER_SIZE,
                            n_encoder_layers=ENCODER_LAYER_SIZE,
                            n_encoder_head=ENCODER_HEAD_SIZE,
                            encoder_dropout=ENCODER_DROPOUT,
                            use_null_token=USE_NULL_TOKEN) 
        
    runner = Runner(device=device, model=model, lr=LR)

    save_dir = "TENet("+str(MODEL)+"),"+\
                        str(BATCH_SIZE)+","+str(WINDOW_SIZE_TENet)+","+\
                        str(HEAD_DEPTH)+","+str(HEAD_SZ)+","+str(FEAT_SZ)+","+str(DEPTH)+","+str(CONDITION_SIZE)+","+str(FLOW_LAYER_SIZE)+","+str(ENCODER_LAYER_SIZE)+","+str(ENCODER_HEAD_SIZE)+","+str(ENCODER_DROPOUT)+","+str(USE_NULL_TOKEN)+","+\
                        str(LR)+","+str(OPTIMIZER)+","+str(SCHEDULER)+","+str(GAMMA)+","+str(WEIGHT_DECAY)+","+str(GRADIENT_CLIPPING)+","+nowDatetime
                                     
    writer = SummaryWriter('logs/'+save_dir+'/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_config = {"head_depth": HEAD_DEPTH,
                    "head_sz": HEAD_SZ,
                    "feat_sz": FEAT_SZ,
                    "depth": DEPTH,
                    "condition_dim": CONDITION_SIZE,        
                    "input_dim": INPUT_SIZE,
                    "window_dim": WINDOW_SIZE_TENet,
                    "target_dim": OUTPUT_SIZE,
                    "n_flow_layers": FLOW_LAYER_SIZE,
                    "n_encoder_head": ENCODER_HEAD_SIZE,
                    "n_encoder_layers": ENCODER_LAYER_SIZE,
                    "use_null_token": USE_NULL_TOKEN}
    
    REPERTORIES = [{'repertorie': 'Pointing and gazing at a single target', 'num': 27},                   
                   {'repertorie': 'Pointing and gazing at two targets in sequence', 'num': 210},
                   {'repertorie': 'Pointing at a single target with gaze shift', 'num': 180},
                   {'repertorie': 'Explaning with pointing', 'num': 126},     
                   {'repertorie': 'Referencing multiple targets with diverse gestures (pointing session 1)', 'num': 126},                   
                   {'repertorie': 'Referencing multiple targets with diverse gestures (pointing session 2)', 'num': 126},                   
                   {'repertorie': 'Transition (Pointing)', 'num': 15},
                   {'repertorie': 'Pointing and gazing at a single target (mirrored)', 'num': 27},
                   {'repertorie': 'Pointing and gazing at two targets in sequence (mirrored)', 'num': 210},
                   {'repertorie': 'Pointing at a single target with gaze shift (mirrored)', 'num': 180},
                   {'repertorie': 'Explaning with pointing (mirrored)', 'num': 126},
                   {'repertorie': 'Referencing multiple targets with diverse gestures (pointing session 1) (mirrored)', 'num': 126},
                   {'repertorie': 'Referencing multiple targets with diverse gestures (pointing session 2) (mirrored)', 'num': 126},                   
                   {'repertorie': 'Transition (Pointing) (mirrored)', 'num': 15},
                   {'repertorie': 'Touching and gazing at a single target', 'num': 30},                   
                   {'repertorie': 'Touching and gazing at two targets in sequence', 'num': 210},
                   {'repertorie': 'Touching a single target with gaze shift', 'num': 210},
                   {'repertorie': 'Explaning with touching', 'num': 126},     
                   {'repertorie': 'Referencing multiple targets with diverse gestures (touching)', 'num': 126},                   
                   {'repertorie': 'Transition (Touching)', 'num': 15},
                   {'repertorie': 'Touching and gazing at a single target (mirrored)', 'num': 30},
                   {'repertorie': 'Touching and gazing at two targets in sequence (mirrored)', 'num': 210},
                   {'repertorie': 'Touching a single target with gaze shift (mirrored)', 'num': 210},
                   {'repertorie': 'Explaning with touching (mirrored)', 'num': 126},
                   {'repertorie': 'Referencing multiple targets with diverse gestures (touching) (mirrored)', 'num': 126},                   
                   {'repertorie': 'Transition (Touching) (mirrored)', 'num': 15}]
    
    # REPERTORIES = [{'repertorie': 'Pointing and gazing at a single target', 'num': 27},
    #                {'repertorie': 'Pointing and gazing at two targets in sequence', 'num': 210},
    #                {'repertorie': 'Pointing at a single target with gaze shift', 'num': 180},
    #                {'repertorie': 'Explaning with pointing', 'num': 126},
    #                {'repertorie': 'Referencing multiple targets with diverse gestures (pointing session 1)', 'num': 126},
    #                {'repertorie': 'Referencing multiple targets with diverse gestures (pointing session 2)', 'num': 126},
    #                {'repertorie': 'Transition (Pointing)', 'num': 15},
    #                {'repertorie': 'Pointing and gazing at a single target (mirrored)', 'num': 27},
    #                {'repertorie': 'Pointing and gazing at two targets in sequence (mirrored)', 'num': 210},
    #                {'repertorie': 'Pointing at a single target with gaze shift (mirrored)', 'num': 180},
    #                {'repertorie': 'Explaning with pointing (mirrored)', 'num': 126},
    #                {'repertorie': 'Referencing multiple targets with diverse gestures (pointing session 1) (mirrored)', 'num': 126},
    #                {'repertorie': 'Referencing multiple targets with diverse gestures (pointing session 2) (mirrored)', 'num': 126},
    #                {'repertorie': 'Transition (Pointing) (mirrored)', 'num': 15}]                
    
    DATA_PARSER = data_parser(REPERTORIES, WINDOW_SIZE_TENet, CLIP_SIZE, "TENet", DIRECTORY_ID)    
    if args.rawDataSplit == True:
        DATA_PARSER.delete_raw_data_dir()
        DATA_PARSER.split()
    DATA_PARSER.delete_data_dir()
    DATA_PARSER.generate(0, False, False)
    DATA_PARSER.__len__()        
    train_data_set = Dataset(directory_id=DIRECTORY_ID, network="TENet")
    train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=4, pin_memory=True, drop_last=True)
    NUM_BATCHES = len(train_data_set) // BATCH_SIZE
    print('# of train example: {}, # of batches {}'.format(len(train_data_set), NUM_BATCHES))

    # ---------------------------------------------------------------------------------------------
    # Preparation for measuring accuracy on the test-split data every epoch.
    # Evaluate only the intersection of the trained categories (left/right suffix removed) and the test categories (per-subject REPERTORIES1/2).
    # ---------------------------------------------------------------------------------------------
    TEST_RAW_DIR = 'data(raw)/TENet/test/'
    TEST_SUBJECTS = ['161(HS)/', '172(CH)/', '173(JH)/', '174(JN)/', '180(SC)/',
                     '158(HJ)/', '165(YN)/', '166(HS)/', '174(KB)/', '175(RS)/', '180(DH)/', '187(ES)/']
    REPERTORIES1_SUBJECTS = ['161(HS)/', '172(CH)/', '173(JH)/', '174(JN)/', '180(SC)/']
    TEST_REPERTORIES1 = [{'repertorie': 'Pointing and gazing at a single target', 'num': 27},
                         {'repertorie': 'Pointing and gazing at two targets in sequence', 'num': 8},
                         {'repertorie': 'Pointing at a single target with gaze shift', 'num': 8},
                         {'repertorie': 'Explaning with pointing', 'num': 8},
                         {'repertorie': 'Touching and gazing at a single target', 'num': 30},
                         {'repertorie': 'Touching a single target with gaze shift', 'num': 8},
                         {'repertorie': 'Explaning with touching', 'num': 8},
                         {'repertorie': 'Touching and gazing at two targets in sequence', 'num': 8}]
    TEST_REPERTORIES2 = [{'repertorie': 'Referencing multiple targets with diverse gestures (pointing)', 'num': 8},
                         {'repertorie': 'Referencing multiple targets with diverse gestures (touching)', 'num': 8}]

    def _strip_side(name):
        return name.replace(' (mirrored)', '')
    TRAINED_CATEGORIES = set(_strip_side(r['repertorie']) for r in REPERTORIES)

    # Actual list of evaluation targets: (subject, repertorie_name, num)
    EVAL_PLAN = []
    for subject in TEST_SUBJECTS:
        test_reps = TEST_REPERTORIES1 if subject in REPERTORIES1_SUBJECTS else TEST_REPERTORIES2
        for rep in test_reps:
            if rep['repertorie'] in TRAINED_CATEGORIES:
                EVAL_PLAN.append((subject, rep['repertorie'], rep['num']))
    print('# of eval (subject, repertorie) pairs: {}'.format(len(EVAL_PLAN)))

    evaluator = Evaluator(device=device, model=model, test_batch_size=TEST_BATCH_SIZE)
    # candidates = (up to 3 real targets) + (1 null) = TEST_TARGET_NUMBER + 1
    EVAL_TARGET_NUMBER = TEST_TARGET_NUMBER + 1
    accuracy_log = []  # (epoch, overall, head, right, left, all_correct)

    train_start = time.time()
    for epoch in range(1, EPOCHS+1):
        epoch_start = time.time()
        total_train_loss = runner.run(train_data_loader, epoch, 'TRAIN')
        torch.save({'model_state_dict': model.state_dict(), 'config': model_config}, os.path.join(save_dir, f"TENet_{MODEL}_{epoch}.pt"))
        training_time = time.time() - epoch_start
        log = "Train Time = %.1f, [Epoch %d/%d] [Loss: %.6f]" % (training_time, epoch, EPOCHS, total_train_loss)
        print(log)

        # -------------------- Per-epoch accuracy evaluation (1 trial) --------------------
        model.eval()
        evaluator.reset()
        with torch.no_grad():
            for subject, repertorie_name, num in EVAL_PLAN:
                folder_split = TEST_RAW_DIR + subject + repertorie_name + '/' + 'split'
                for i in range(num):
                    data_TENet = np.loadtxt(osp.join(folder_split, 'Data'+str(i+1)+'.txt')).astype(np.float32)
                    data_TENet = torch.tensor(data_TENet, dtype=torch.float32)
                    evaluator.run(data_TENet, repertorie_name, EVAL_TARGET_NUMBER)

        if evaluator.iter > 0:
            accuracy = evaluator.count / evaluator.iter
            accuracy_head = evaluator.count_head / evaluator.iter * 3
            accuracy_righthand = evaluator.count_righthand / evaluator.iter * 3
            accuracy_lefthand = evaluator.count_lefthand / evaluator.iter * 3
            accuracy_head_nn = evaluator.count_head_nn / evaluator.iter_head_nn if evaluator.iter_head_nn > 0 else 0.0
            accuracy_righthand_nn = evaluator.count_righthand_nn / evaluator.iter_righthand_nn if evaluator.iter_righthand_nn > 0 else 0.0
            accuracy_lefthand_nn = evaluator.count_lefthand_nn / evaluator.iter_lefthand_nn if evaluator.iter_lefthand_nn > 0 else 0.0
        else:
            accuracy = accuracy_head = accuracy_righthand = accuracy_lefthand = 0.0
            accuracy_head_nn = accuracy_righthand_nn = accuracy_lefthand_nn = 0.0
        accuracy_log.append((epoch, accuracy, accuracy_head, accuracy_righthand, accuracy_lefthand,
                             accuracy_head_nn, accuracy_righthand_nn, accuracy_lefthand_nn))
        eval_log = "[Epoch %d/%d] Overall: %.3f | Head: %.3f | Right: %.3f | Left: %.3f | Head(NN): %.3f | Right(NN): %.3f | Left(NN): %.3f" % (
            epoch, EPOCHS, accuracy, accuracy_head, accuracy_righthand, accuracy_lefthand,
            accuracy_head_nn, accuracy_righthand_nn, accuracy_lefthand_nn)
        print(eval_log)

    # After all training is finished, save the accuracy results as txt
    with open(save_dir+"/accuracy.txt", "w") as file:
        file.write("Epoch,Overall,Head,RightHand,LeftHand,Head_NonNull,RightHand_NonNull,LeftHand_NonNull\n")
        for (ep, ov, hd, rt, lf, hd_nn, rt_nn, lf_nn) in accuracy_log:
            file.write(f"{ep},{ov},{hd},{rt},{lf},{hd_nn},{rt_nn},{lf_nn}\n")

    print("Total Time= %.2f" %(time.time()-train_start))
    print("\n")       
    writer.flush()
    writer.close()
    
    text_to_save = "Total Time= %.2f" %(time.time()-script_start)
    with open(save_dir+"/training time.txt", "w") as file:
        file.write(text_to_save)             