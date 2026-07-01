import os
import gc
import time
import datetime 
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import BP_GTNet
from data_parser import data_parser
from data_loader import Dataset

class Runner(object):                            
    def __init__(self, device, model, lr):               
        self.device = device
        self.model = model.to(self.device)
        if OPTIMIZER=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif OPTIMIZER=='AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)        
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.learning_rate = lr
        self.iter = 0       
        self.dt = 1.0 / 30.0 
        
    def calculate_velocity_loss(self, Y_curr, Y_prev, Y_hat_curr, Y_hat_prev):
        Y_vel = (Y_curr - Y_prev) / self.dt
        Y_hat_vel = (Y_hat_curr - Y_hat_prev) / self.dt
        
        return torch.mean(torch.abs(Y_vel - Y_hat_vel), dim=-1)    
    
    def calculate_acceleration_loss(self, Y_curr, Y_prev, Y_prev_prev, Y_hat_curr, Y_hat_prev, Y_hat_prev_prev):
        Y_acc = (Y_curr - 2 * Y_prev + Y_prev_prev) / (self.dt ** 2)
        Y_hat_acc = (Y_hat_curr - 2 * Y_hat_prev + Y_hat_prev_prev) / (self.dt ** 2)
        
        return torch.mean(torch.abs(Y_acc - Y_hat_acc), dim=-1)
                 
    def run(self, dataloader, epoch, Bernoulli, mode='TRAIN'):
        if mode == 'TRAIN':
            self.model.train()

        epoch_loss = 0    
        epoch_loss_rec = 0
        epoch_loss_rec_head = 0        
        epoch_loss_rec_righthand = 0
        epoch_loss_rec_lefthand = 0
        epoch_loss_vel = 0  
        epoch_loss_acc = 0          
        epoch_loss_vq = 0
                       
        pbar = tqdm(dataloader)
        for iter, data in enumerate(pbar):
            DATA = data['data'].to(self.device)                              
            
            Y_list, Y_hat_list = [], []
            for t in range(T):                        
                X_Target_User   = DATA[:, t:t+WINDOW_SIZE_BP_GTNet, 0:3]
                X_Target_Avatar = DATA[:, t:t+WINDOW_SIZE_BP_GTNet, :INPUT_SIZE_GATING]
                Confidence_full = DATA[:, t:t+WINDOW_SIZE_BP_GTNet, INPUT_SIZE_GATING:INPUT_SIZE_GATING+3]
                Confidence_Head_t    = Confidence_full[:, -1, 0]
                Confidence_RightHand_t    = Confidence_full[:, -1, 1]
                Confidence_LeftHand_t    = Confidence_full[:, -1, 2]
        
                X_Pose_User   = DATA[:, t+WINDOW_SIZE_BP_GTNet-USER_POSE_WINDOW_SIZE:t+WINDOW_SIZE_BP_GTNet, INPUT_SIZE_GATING+3:INPUT_SIZE_GATING+3+INPUT_SIZE_ENCODER]
                X_Pose_Avatar = DATA[:, t+WINDOW_SIZE_BP_GTNet-2, -INPUT_SIZE_MoE:]
                Y             = DATA[:, t+WINDOW_SIZE_BP_GTNet-1, -INPUT_SIZE_MoE:]
                Y_prev        = DATA[:, t+WINDOW_SIZE_BP_GTNet-2, -INPUT_SIZE_MoE:]
                Y_prev_prev   = DATA[:, t+WINDOW_SIZE_BP_GTNet-3, -INPUT_SIZE_MoE:]
                
                use_autoreg = (t != 0) and (Bernoulli.sample().int() == 1)
                if use_autoreg and len(Y_hat_list) >= 1:
                    X_Pose_Avatar = Y_hat_list[-1]
                    
                Y_hat, Y_head_hat, Y_righthand_hat, Y_lefthand_hat, losses_vq = self.model(X_Target_User, 
                                                                                           X_Target_Avatar, 
                                                                                           X_Pose_User, 
                                                                                           X_Pose_Avatar,
                                                                                           Confidence_full,
                                                                                           True)
                # Reconstruction
                if WEIGHT_HEAD_POS != 1:
                    w = torch.ones(Y.shape[-1], device=self.device)
                    w[6:9] = WEIGHT_HEAD_POS
                    loss_rec = torch.mean(torch.abs(Y - Y_hat) * w)
                else:
                    loss_rec = F.l1_loss(Y, Y_hat, reduction='mean')
                loss_rec            = F.l1_loss(Y,            Y_hat,           reduction='mean')
                loss_rec_head       = F.l1_loss(Y[:, 0:9],    Y_head_hat,      reduction='mean')
                loss_rec_righthand  = F.l1_loss(Y[:, 12:21],  Y_righthand_hat, reduction='mean')
                loss_rec_lefthand   = F.l1_loss(Y[:, 24:33],  Y_lefthand_hat,  reduction='mean')
                
                # Velocity/acceleration (per-batch vectors) - each guarded independently
                loss_vel = torch.tensor(0.0, device=self.device)
                if WEIGHT_VEL != 0:
                    if use_autoreg and len(Y_hat_list) >= 1:
                        main_head_vel = self.calculate_velocity_loss(Y[:, 0:12], Y_list[-1][:, 0:12], Y_hat[:, 0:12], Y_hat_list[-1][:, 0:12])                    
                        main_righthand_vel = self.calculate_velocity_loss(Y[:, 12:24], Y_list[-1][:, 12:24], Y_hat[:, 12:24], Y_hat_list[-1][:, 12:24])                        
                        main_lefthand_vel = self.calculate_velocity_loss(Y[:, 24:36], Y_list[-1][:, 24:36], Y_hat[:, 24:36], Y_hat_list[-1][:, 24:36])                        
                    else:
                        main_head_vel = self.calculate_velocity_loss(Y[:, 0:12], Y_prev[:, 0:12], Y_hat[:, 0:12], Y_prev[:, 0:12])                    
                        main_righthand_vel = self.calculate_velocity_loss(Y[:, 12:24], Y_prev[:, 12:24], Y_hat[:, 12:24], Y_prev[:, 12:24])                        
                        main_lefthand_vel = self.calculate_velocity_loss(Y[:, 24:36], Y_prev[:, 24:36], Y_hat[:, 24:36], Y_prev[:, 24:36])   
                    loss_vel = WEIGHT_VEL * ( ((1.0 - Confidence_Head_t) * main_head_vel).mean() +\
                                              ((1.0 - Confidence_RightHand_t) * main_righthand_vel).mean() +\
                                              ((1.0 - Confidence_LeftHand_t) * main_lefthand_vel).mean() )
        
                loss_acc = torch.tensor(0.0, device=self.device)
                if WEIGHT_ACC != 0:
                    if use_autoreg and len(Y_hat_list) >= 2:
                        main_head_acc = self.calculate_acceleration_loss(Y[:, 0:12], Y_list[-1][:, 0:12], Y_list[-2][:, 0:12], Y_hat[:, 0:12], Y_hat_list[-1][:, 0:12], Y_hat_list[-2][:, 0:12])
                        main_righthand_acc = self.calculate_acceleration_loss(Y[:, 12:24], Y_list[-1][:, 12:24], Y_list[-2][:, 12:24], Y_hat[:, 12:24], Y_hat_list[-1][:, 12:24], Y_hat_list[-2][:, 12:24])
                        main_lefthand_acc = self.calculate_acceleration_loss(Y[:, 24:36], Y_list[-1][:, 24:36], Y_list[-2][:, 24:36], Y_hat[:, 24:36], Y_hat_list[-1][:, 24:36], Y_hat_list[-2][:, 24:36])
                    elif use_autoreg and len(Y_hat_list) == 1:
                        main_head_acc = self.calculate_acceleration_loss(Y[:, 0:12], Y_list[-1][:, 0:12], Y_prev_prev[:, 0:12], Y_hat[:, 0:12], Y_hat_list[-1][:, 0:12], Y_prev_prev[:, 0:12])                    
                        main_righthand_acc = self.calculate_acceleration_loss(Y[:, 12:24], Y_list[-1][:, 12:24], Y_prev_prev[:, 12:24], Y_hat[:, 12:24], Y_hat_list[-1][:, 12:24], Y_prev_prev[:, 12:24])                        
                        main_lefthand_acc = self.calculate_acceleration_loss(Y[:, 24:36], Y_list[-1][:, 24:36], Y_prev_prev[:, 24:36], Y_hat[:, 24:36], Y_hat_list[-1][:, 24:36], Y_prev_prev[:, 24:36])                        
                    else:
                        main_head_acc = self.calculate_acceleration_loss(Y[:, 0:12], Y_prev[:, 0:12], Y_prev_prev[:, 0:12], Y_hat[:, 0:12], Y_prev[:, 0:12], Y_prev_prev[:, 0:12])
                        main_righthand_acc = self.calculate_acceleration_loss(Y[:, 12:24], Y_prev[:, 12:24], Y_prev_prev[:, 12:24], Y_hat[:, 12:24], Y_prev[:, 12:24], Y_prev_prev[:, 12:24])                        
                        main_lefthand_acc = self.calculate_acceleration_loss(Y[:, 24:36], Y_prev[:, 24:36], Y_prev_prev[:, 24:36], Y_hat[:, 24:36], Y_prev[:, 24:36], Y_prev_prev[:, 24:36])                        
                    loss_acc = WEIGHT_ACC * ( ((1.0 - Confidence_Head_t) * main_head_acc).mean() +\
                                              ((1.0 - Confidence_RightHand_t) * main_righthand_acc).mean() +\
                                              ((1.0 - Confidence_LeftHand_t) * main_lefthand_acc).mean() )
                
                # VQ
                loss_vq = WEIGHT_VQ_LOSS * losses_vq['total']
        
                loss = loss_rec + loss_rec_head + loss_rec_righthand + loss_rec_lefthand + loss_vel + loss_acc + loss_vq
                
                if mode=='TRAIN':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                Y_list.append(Y.detach())
                Y_hat_list.append(Y_hat.detach())
                
                if iter % 50 == 0:
                    if mode == 'TRAIN':
                        writer.add_scalar("Train", loss.item(), self.iter)
                        self.iter += 1
                
                pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, iter+1, len(dataloader)))
                pbar.set_postfix({"loss":((loss.item()))})     
                                  
                epoch_loss += loss.item()
                epoch_loss_rec += loss_rec.item()
                epoch_loss_rec_head += loss_rec_head.item()                
                epoch_loss_rec_righthand += loss_rec_righthand.item()
                epoch_loss_rec_lefthand += loss_rec_lefthand.item()
                epoch_loss_vel += loss_vel.item()
                epoch_loss_acc += loss_acc.item()                
                epoch_loss_vq += loss_vq.item()
                
        epoch_loss /= len(dataloader)
        epoch_loss_rec /= len(dataloader)
        epoch_loss_rec_head /= len(dataloader)        
        epoch_loss_rec_righthand /= len(dataloader)
        epoch_loss_rec_lefthand /= len(dataloader)
        epoch_loss_vel /= len(dataloader)
        epoch_loss_acc /= len(dataloader)        
        epoch_loss_vq /= len(dataloader)
        
        self.scheduler.step()
        
        return epoch_loss, epoch_loss_rec, epoch_loss_rec_head, epoch_loss_rec_righthand, epoch_loss_rec_lefthand, epoch_loss_vel, epoch_loss_acc, epoch_loss_vq
        
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', help='Model')
parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--directory_id', type=int, default=0, help='Directory ID')
parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--epochs', type=int, default=120, help='Epochs')
parser.add_argument('--c1', type=int, default=30, help='C1')
parser.add_argument('--c2', type=int, default=60, help='C2')
parser.add_argument('--batchSize', type=int, default=32, help='Batch Size')
parser.add_argument('--windowSizeBP_GTNet', type=int, default=30, help='Window Size (BP_GTNet)')
parser.add_argument('--layerSizeEncoder', type=int, default=4, help='Layer Size (Encoder)')
parser.add_argument('--headCodebookChannel', type=int, default=6, help='Head Codebook Channel')
parser.add_argument('--headCodebookDimension', type=int, default=6, help='Head Codebook Dimension')
parser.add_argument('--headTemperature', type=float, default=5, help='Head Temperature')
parser.add_argument('--handHiddenSize', type=int, default=12, help='Hand Hidden Size')
parser.add_argument('--handCodebookSize', type=int, default=64, help='Hand Codebook Size')
parser.add_argument('--handCodebookLayerSize', type=int, default=5, help='Hand Codebook Residual Layer Size')
parser.add_argument('--modelGating', default='Transformer', help='Gating (Network Model)')
parser.add_argument('--layerSizeGating', type=int, default=2, help='Layer Size (Gating Network)')
parser.add_argument('--hiddenSizeGating', type=int, default=64, help='Layer Size (Gating Network)')
parser.add_argument('--userPoseWindowSize', type=int, default=5, help='User Pose Window Size')
parser.add_argument('--clipSize', type=int, default=45, help='Clip Size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--optimizer', default='AdamW', help='Optimizer')
parser.add_argument('--expertNumber', type=int, default=24, help='Expert Number')
parser.add_argument('--expertNumberPart', type=int, default=15, help='Expert Number Part')
parser.add_argument('--rawDataSplit', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--weightVel', type=float, default=0, help='Weight (Velocity Loss)')
parser.add_argument('--weightAcc', type=float, default=0, help='Weight (Acceleration Loss)')
parser.add_argument('--weightVQLoss', type=float, default=1, help='Weight (VQ Loss)')
parser.add_argument('--weightHeadPos', type=float, default=1, help='Weight (Head Position)')
args = parser.parse_args()

# python train_BP_GTNet.py --gpu_id 1 --directory_id 1 --weightVQLoss 5; python train_BP_GTNet.py --gpu_id 1 --directory_id 1 --weightVQLoss 1;

if __name__ == '__main__':
    script_start = time.time()

    gc.collect()
    torch.cuda.empty_cache()
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%m%d_%H%M%S')
    
    MODEL = args.model
    GPU_ID = args.gpu_id
    DIRECTORY_ID = args.directory_id
    SHUFFLE = args.shuffle        
    EPOCHS = args.epochs
    C1 = args.c1
    C2 = args.c2
    BATCH_SIZE = args.batchSize
    WINDOW_SIZE_BP_GTNet = args.windowSizeBP_GTNet    
    LAYER_SIZE_ENCODER = args.layerSizeEncoder
    HEAD_CODEBOOK_CHANNEL = args.headCodebookChannel
    HEAD_CODEBOOK_DIMENSION = args.headCodebookDimension
    HEAD_TEMPERATURE = args.headTemperature 
    HAND_HIDDEN_SIZE = args.handHiddenSize     
    HAND_CODEBOOK_SIZE = args.handCodebookSize
    HAND_CODEBOOK_LAYER_SIZE = args.handCodebookLayerSize
    MODEL_GATING = args.modelGating
    LAYER_SIZE_GATING = args.layerSizeGating    
    HIDDEN_SIZE_GATING = args.hiddenSizeGating

    USER_POSE_WINDOW_SIZE = args.userPoseWindowSize
    CLIP_SIZE = args.clipSize
    T = CLIP_SIZE - WINDOW_SIZE_BP_GTNet + 1
    LR = args.lr    
    OPTIMIZER = args.optimizer
    WEIGHT_VEL = args.weightVel
    WEIGHT_ACC = args.weightAcc    
    WEIGHT_VQ_LOSS = args.weightVQLoss
    WEIGHT_HEAD_POS = args.weightHeadPos
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  
    
    NUM_EXPERTS = args.expertNumber 
    NUM_EXPERTS_PART = args.expertNumberPart
    INPUT_SIZE_ENCODER = 27
    INPUT_SIZE_GATING = 15
    INPUT_SIZE_MoE = 36
    
    model = BP_GTNet(rng=np.random.RandomState(23456), 
                  num_experts=NUM_EXPERTS, 
                  num_experts_part=NUM_EXPERTS_PART, 
                  window_size=WINDOW_SIZE_BP_GTNet,
                  user_pose_window_size=USER_POSE_WINDOW_SIZE,
                  layer_size_encoder=LAYER_SIZE_ENCODER,
                  input_size_encoder=INPUT_SIZE_ENCODER,
                  head_codebook_channels=HEAD_CODEBOOK_CHANNEL, 
                  head_codebook_dimension=HEAD_CODEBOOK_DIMENSION,
                  head_temperature=HEAD_TEMPERATURE,
                  hand_hidden_size=HAND_HIDDEN_SIZE,
                  hand_codebook_size=HAND_CODEBOOK_SIZE,
                  hand_codebook_layer_size=HAND_CODEBOOK_LAYER_SIZE,
                  model_gating=MODEL_GATING,
                  layer_size_gating=LAYER_SIZE_GATING,
                  input_size_gating=INPUT_SIZE_GATING,
                  hidden_size_gating=HIDDEN_SIZE_GATING,
                  input_size_moe=INPUT_SIZE_MoE, 
                  use_cuda=True)       
    
    runner = Runner(device=device, model=model, lr=LR)

    save_dir = "BP_GTNet"+","+str(EPOCHS)+","+str(C1)+","+ str(C2)+","+\
                           str(BATCH_SIZE)+","+str(WINDOW_SIZE_BP_GTNet)+","+\
                           str(LAYER_SIZE_ENCODER)+","+\
                           str(HEAD_CODEBOOK_CHANNEL)+","+str(HEAD_CODEBOOK_DIMENSION)+","+str(HEAD_TEMPERATURE)+","+\
                           str(HAND_HIDDEN_SIZE)+","+str(HAND_CODEBOOK_SIZE)+","+str(HAND_CODEBOOK_LAYER_SIZE)+","+\
                           str(MODEL_GATING)+","+str(LAYER_SIZE_GATING)+","+str(HIDDEN_SIZE_GATING)+","+str(USER_POSE_WINDOW_SIZE)+","+\
                           str(CLIP_SIZE)+","+str(LR)+","+str(OPTIMIZER)+","+str(NUM_EXPERTS)+","+str(NUM_EXPERTS_PART)+","+\
                           str(WEIGHT_VEL)+","+str(WEIGHT_ACC)+","+str(WEIGHT_VQ_LOSS)+","+str(WEIGHT_HEAD_POS)+","+nowDatetime
                                     
    writer = SummaryWriter('logs/'+save_dir+'/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_config = {"num_experts" : NUM_EXPERTS, 
                    "num_experts_part" : NUM_EXPERTS_PART, 
                    "window_size" : WINDOW_SIZE_BP_GTNet,
                    "user_pose_window_size" : USER_POSE_WINDOW_SIZE,
                    "layer_size_encoder" : LAYER_SIZE_ENCODER,
                    "input_size_encoder" : INPUT_SIZE_ENCODER,
                    "head_codebook_channels" : HEAD_CODEBOOK_CHANNEL, 
                    "head_codebook_dimension" : HEAD_CODEBOOK_DIMENSION,
                    "head_temperature" : HEAD_TEMPERATURE,
                    "hand_hidden_size" : HAND_HIDDEN_SIZE,
                    "hand_codebook_size" : HAND_CODEBOOK_SIZE,
                    "hand_codebook_layer_size" : HAND_CODEBOOK_LAYER_SIZE,
                    "model_gating" : MODEL_GATING,
                    "layer_size_gating" : LAYER_SIZE_GATING,
                    "input_size_gating" : INPUT_SIZE_GATING,                    
                    "hidden_size_gating" : HIDDEN_SIZE_GATING,
                    "input_size_moe" : INPUT_SIZE_MoE, 
                    "use_cuda" : True}

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

    # BP_GTNet Data    
    DATA_PARSER = data_parser(REPERTORIES, WINDOW_SIZE_BP_GTNet, CLIP_SIZE, "BP_GTNet", DIRECTORY_ID)
    if args.rawDataSplit == True:
        DATA_PARSER.delete_raw_data_dir()
        DATA_PARSER.split()
    DATA_PARSER.delete_data_dir()    
    DATA_PARSER.generate(0, False, False)

    train_data_set = Dataset(directory_id=DIRECTORY_ID, network="BP_GTNet")
    train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=4, pin_memory=True, drop_last=True)
    NUM_BATCHES = len(train_data_set) // BATCH_SIZE
    print('# of train example: {}, # of batches {}'.format(len(train_data_set), NUM_BATCHES))
    
    P = 1
    train_start = time.time()
    for epoch in range(1, EPOCHS+1):                                 
        if epoch <= C1:
            P = 1
        elif C1 < epoch <= C2:
            P = 1 - (epoch - C1) / float(C2 - C1)            
        else:
            P = 0                
        epoch_start = time.time()
        Bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(1-P, dtype=torch.float))                  
        total_train_loss, total_train_loss_rec, total_train_loss_rec_head, total_train_loss_rec_righthand, total_train_loss_rec_lefthand, total_train_loss_vel, total_train_loss_acc, total_train_loss_vq = runner.run(train_data_loader, epoch, Bernoulli, 'TRAIN')
        torch.save({'model_state_dict': model.state_dict(), 'config': model_config}, os.path.join(save_dir, f"BP_GTNet_{epoch}.pt"))
        training_time = time.time() - epoch_start
        log = "Train Time = %.1f, [Epoch %d/%d] [Loss: %.4f] [REC Loss: %.4f] [Head Loss: %.4f] [RHand Loss: %.4f] [LHand Loss: %.4f] [VEL Loss: %.4f] [ACC Loss: %.4f] [VQ Loss: %.4f]" % (training_time, epoch, EPOCHS, total_train_loss, total_train_loss_rec, total_train_loss_rec_head, total_train_loss_rec_righthand, total_train_loss_rec_lefthand, total_train_loss_vel, total_train_loss_acc, total_train_loss_vq)
        print(log)                           
    print("Total Time= %.2f" %(time.time()-train_start))                           
    print("\n")    
    writer.flush()
    writer.close()
    
    text_to_save = "Total Time= %.2f" %(time.time()-script_start)
    with open(save_dir+"/training time.txt", "w") as file:
        file.write(text_to_save)    