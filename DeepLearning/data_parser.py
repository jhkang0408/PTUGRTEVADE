import os
import shutil
import gc
import time
import math
import torch
import os.path as osp
import numpy as np
import random 
from torch.nn import functional as F 
from scipy import interpolate 

class data_parser:
    def __init__(self, repertories, window_size, clip_size, network, directory_id, TENet=None):
        self.repertories = repertories        
        self.WINDOW_SIZE = window_size
        self.CLIP_SIZE = clip_size
        self.NETWORK = network
        self.raw_data_dir = 'data(raw)/'+network+'/''train/'
        self.data_dir = 'data/train'+'_'+network+'_'+str(directory_id)+'/'             
        self.count = 0
        self.TENet = TENet
        self.data_mean = None
        self.data_std = None
        
    def __len__(self):
        print('# of Path='+str(len(os.listdir(self.data_dir)))) 
        
    def split(self):
        for repertorie in self.repertories:
            folder_raw = self.raw_data_dir + repertorie['repertorie'] + '/' + 'raw'
            folder_split = self.raw_data_dir  + repertorie['repertorie'] + '/' + 'split'
            for f in os.listdir(folder_split):
                os.remove(os.path.join(folder_split, f))
            input_data = np.loadtxt(osp.join(folder_raw, 'Data.txt')).astype(np.float32)
            sequences = np.loadtxt(osp.join(folder_raw, 'Sequences.txt')).astype(np.float32)
            for i in range(1, int(np.max(sequences))+1):
                np.savetxt(osp.join(folder_split, 'Data'+str(i)+'.txt'), input_data[np.where(sequences==i)], fmt='%1.4e')        
        
    def delete_raw_data_dir(self):
        for repertorie in self.repertories:
            folder_split = self.raw_data_dir + repertorie['repertorie'] + '/' + 'split'     
            for f in os.listdir(folder_split):
                os.remove(os.path.join(folder_split, f))                        

    def delete_data_dir(self):
        # Ensure the parent 'data' folder exists (created if missing, left alone if present),
        # then start this run's folder fresh: delete it if it exists, then (re)create it.
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir)

    def devide(self, repertorie_name, repertorie_num, folder_split, idx, num_targets=3, target_augmentation=True, joint_training=False):              
        if self.TENet is not None:
            self.TENet.eval()
            
        with torch.inference_mode():
            data = np.loadtxt(osp.join(folder_split, 'Data'+str(idx)+'.txt')).astype(np.float32)                                
            data = torch.tensor(data, dtype=torch.float32)                
            ####################################################################################################################################################################################
            ####################################################################################################################################################################################            
            if self.NETWORK == "TENet":                                         
                if repertorie_name=='Transition (Pointing)' or repertorie_name=='Transition (Pointing) (mirrored)' or\
                   repertorie_name=='Transition (Touching)' or repertorie_name=='Transition (Touching) (mirrored)':
                    head_target = data[:,12:15]
                    
                    # Find frames that have a target (frames that are not -10)
                    has_target = torch.any(head_target != 0, dim=1)
                    
                    # Find continuous segments
                    segments = []
                    start = None                    
                    for i in range(len(has_target)):
                        target_exists = has_target[i].item()
                        
                        if target_exists and start is None:
                            start = i  # Start of a new segment
                        elif not target_exists and start is not None:
                            segments.append((start, i-1))  # End of segment
                            start = None
                    
                    # Case where the last segment continues to the end
                    if start is not None:
                        segments.append((start, len(has_target)-1))
                    
                    # Extract the data of each segment
                    for start, end in segments:
                        segment_data = data[start:end+1]  # Full data of the corresponding segment
                        first_segment_data_frame = segment_data[0:1].repeat(15, 1)
                        segment_data = torch.cat([first_segment_data_frame, segment_data], dim=0)
                        
                        original_bool_data = segment_data[:, 9:12].clone()
                        continuous_bool_data = torch.zeros_like(original_bool_data)

                        for feature_idx in range(3):
                            cum_sum = torch.cumsum(segment_data[:, 9+feature_idx], dim=0)
                            continuous_bool_data[15:, feature_idx] = (cum_sum[15:] - cum_sum[:-15]) / 15                                            
                        segment_data[:, 9:12] = continuous_bool_data                          
                        Segment_Data = segment_data[15:]
                        
                        first_Segment_Data_frame = Segment_Data[0:1].repeat(self.WINDOW_SIZE, 1)
                        Segment_Data = torch.cat([first_Segment_Data_frame, Segment_Data], dim=0)
                                                    
                        Segment_Data_unfold = torch.transpose(Segment_Data.unfold(0, self.WINDOW_SIZE, 1), 1, 2)                    
                        for k in range(Segment_Data_unfold.shape[0]):
                            np.save(osp.join(self.data_dir, str(self.count)), Segment_Data_unfold[k].cpu().numpy())
                            self.count = self.count + 1                        
                else:                                                         
                    first_data_frame = data[0:1].repeat(15, 1)
                    data = torch.cat([first_data_frame, data], dim=0)
    
                    original_bool_data = data[:, 9:12].clone()
                    continuous_bool_data = torch.zeros_like(original_bool_data)
                    
                    for feature_idx in range(3):
                        cum_sum = torch.cumsum(data[:, 9+feature_idx], dim=0)
                        continuous_bool_data[15:, feature_idx] = (cum_sum[15:] - cum_sum[:-15]) / 15                                            
                    data[:, 9:12] = continuous_bool_data                          
                    Data = data[15:]

                    first_Data_frame = Data[0:1].repeat(self.WINDOW_SIZE, 1)
                    Data = torch.cat([first_Data_frame, Data], dim=0)
                                                
                    Data_unfold = torch.transpose(Data.unfold(0, self.WINDOW_SIZE, 1), 1, 2)                    
                    for k in range(Data_unfold.shape[0]):
                        np.save(osp.join(self.data_dir, str(self.count)), Data_unfold[k].cpu().numpy())
                        self.count = self.count + 1                        
                    
            ####################################################################################################################################################################################
            ####################################################################################################################################################################################                                    
            elif self.NETWORK == "BP_GTNet":   
                if repertorie_name=='Transition (Pointing)' or repertorie_name=='Transition (Pointing) (mirrored)' or\
                   repertorie_name=='Transition (Touching)' or repertorie_name=='Transition (Touching) (mirrored)':
                    head_target = data[:,0:3]
                    
                    # Find frames that have a target (frames that are not 0,0,0)
                    has_target = torch.any(head_target != 0, dim=1)
                    
                    # Find continuous segments
                    segments = []
                    start = None                    
                    for i in range(len(has_target)):
                        target_exists = has_target[i].item()
                        
                        if target_exists and start is None:
                            start = i  # Start of a new segment
                        elif not target_exists and start is not None:
                            segments.append((start, i-1))  # End of segment
                            start = None
                    
                    # Case where the last segment continues to the end
                    if start is not None:
                        segments.append((start, len(has_target)-1))
                    
                    # Extract the data of each segment
                    for start, end in segments:
                        segment_data = data[start:end+1]  # Full data of the corresponding segment

                        # Head Confidence
                        Confidence_Head = []
                        confidence_Head = 0.0
                        Confidence_Head.append(confidence_Head) 
                        for i in range(segment_data.shape[0] - 1):
                            if torch.equal(segment_data[i+1, 0:3], segment_data[i, 0:3]):
                                if torch.all(segment_data[i+1, 0:3]==0) == False:                                    
                                    if segment_data[i+1, 9].item() == 1:
                                        confidence_Head = min(1.0, confidence_Head + 1.0 / 15.0)
                                    else:  # if 0, decrease
                                        confidence_Head = max(0.0, confidence_Head - 1.0 / 15.0)
                            else:
                                confidence_Head = 0.0                  
                            Confidence_Head.append(confidence_Head)                        
                        Confidence_Head = torch.tensor(Confidence_Head, device=segment_data.device, dtype=segment_data.dtype)

                        # RightHand Confidence
                        Confidence_RightHand = []
                        confidence_RightHand = 0.0
                        Confidence_RightHand.append(confidence_RightHand) 
                        for i in range(segment_data.shape[0] - 1):
                            if torch.equal(segment_data[i+1, 3:6], segment_data[i, 3:6]):
                                if torch.all(segment_data[i+1, 3:6]==0) == False and torch.all(segment_data[i+1, 0:3]==0) == False:
                                    if segment_data[i+1, 9].item() == 1: 
                                        confidence_RightHand = min(1.0, confidence_RightHand + 1.0 / 15.0)
                                    else:  # if 0, decrease
                                        confidence_RightHand = max(0.0, confidence_RightHand - 1.0 / 15.0)
                                else:
                                    confidence_RightHand = max(0.0, confidence_RightHand - 1.0 / 15.0)
                            else:
                                confidence_RightHand = 0.0                        
                            Confidence_RightHand.append(confidence_RightHand)                        
                        Confidence_RightHand = torch.tensor(Confidence_RightHand, device=segment_data.device, dtype=segment_data.dtype)

                        # LeftHand Confidence
                        Confidence_LeftHand = []
                        confidence_LeftHand = 0.0
                        Confidence_LeftHand.append(confidence_LeftHand)
                        for i in range(segment_data.shape[0] - 1):
                            if torch.equal(segment_data[i+1, 6:9], segment_data[i, 6:9]):                                    
                                if torch.all(segment_data[i+1, 6:9]==0) == False and torch.all(segment_data[i+1, 0:3]==0) == False:    
                                    if segment_data[i+1, 9].item() == 1:  # if 1, increase
                                        confidence_LeftHand = min(1.0, confidence_LeftHand + 1.0 / 15.0)
                                    else:  # if 0, decrease
                                        confidence_LeftHand = max(0.0, confidence_LeftHand - 1.0 / 15.0)
                                else:
                                    confidence_LeftHand = max(0.0, confidence_LeftHand - 1.0 / 15.0)
                            else:
                                confidence_LeftHand = 0.0                        
                            Confidence_LeftHand.append(confidence_LeftHand)                        
                        Confidence_LeftHand = torch.tensor(Confidence_LeftHand, device=segment_data.device, dtype=segment_data.dtype)   
                            
                        # Compute the bool feature
                        first_segment_data_frame = segment_data[0:1].repeat(15, 1)
                        segment_data = torch.cat([first_segment_data_frame, segment_data], dim=0)                            
                        original_bool_data = segment_data[:, 9:15].clone()
                        continuous_bool_data = torch.zeros_like(original_bool_data)

                        for feature_idx in range(6):
                            cum_sum = torch.cumsum(segment_data[:, 9+feature_idx], dim=0)
                            continuous_bool_data[15:, feature_idx] = (cum_sum[15:] - cum_sum[:-15]) / 15                                            
                        segment_data[:, 9:15] = continuous_bool_data                          
                        segment_data = segment_data[15:]    
                        segment_data_clone = segment_data.clone() 
                        
                        # Save data
                        feature = segment_data_clone[:, :15]
                        motion_encoder = segment_data_clone[:, 15:15+27]                    
                        motion_decoder = segment_data_clone[:, 15+27:15+27+36]       
                        
                        Segment_Data =  torch.cat((feature, 
                                                   Confidence_Head.unsqueeze(-1), 
                                                   Confidence_RightHand.unsqueeze(-1), 
                                                   Confidence_LeftHand.unsqueeze(-1), 
                                                   motion_encoder, 
                                                   motion_decoder), -1)
                        first_Segment_Data_frame = Segment_Data[0:1].repeat(self.WINDOW_SIZE, 1)
                        Segment_Data = torch.cat([first_Segment_Data_frame, Segment_Data], dim=0)
                                                    
                        np.save(osp.join(self.data_dir, str(self.count)), Segment_Data[-self.CLIP_SIZE:, :].cpu().numpy())
                        self.count = self.count + 1            
                        Segment_Data_unfold = torch.transpose(Segment_Data.unfold(0, self.CLIP_SIZE, (self.CLIP_SIZE-self.WINDOW_SIZE)//2), 1, 2)             
                        for k in range(Segment_Data_unfold.shape[0]):
                            np.save(osp.join(self.data_dir, str(self.count)), Segment_Data_unfold[k].cpu().numpy())
                            self.count = self.count + 1                            
                     
                else:                        
                    # Head Confidence
                    Confidence_Head = []
                    confidence_Head = 0.0
                    Confidence_Head.append(confidence_Head) 
                    for i in range(data.shape[0] - 1):
                        if torch.equal(data[i+1, 0:3], data[i, 0:3]):
                            if torch.all(data[i+1, 0:3]==0) == False:                                  
                                if data[i+1, 9].item() == 1:
                                    confidence_Head = min(1.0, confidence_Head + 1.0 / 15.0)
                                else:  # if 0, decrease
                                    confidence_Head = max(0.0, confidence_Head - 1.0 / 15.0)
                        else:
                            confidence_Head = 0.0                  
                        Confidence_Head.append(confidence_Head)                        
                    Confidence_Head = torch.tensor(Confidence_Head, device=data.device, dtype=data.dtype)

                    # RightHand Confidence
                    Confidence_RightHand = []
                    confidence_RightHand = 0.0
                    Confidence_RightHand.append(confidence_RightHand) 
                    for i in range(data.shape[0] - 1):
                        if torch.equal(data[i+1, 3:6], data[i, 3:6]):
                            if torch.all(data[i+1, 3:6]==0) == False and torch.all(data[i+1, 0:3]==0) == False:
                                if data[i+1, 9].item() == 1: 
                                    confidence_RightHand = min(1.0, confidence_RightHand + 1.0 / 15.0)
                                else:  # if 0, decrease
                                    confidence_RightHand = max(0.0, confidence_RightHand - 1.0 / 15.0)
                            else:
                                confidence_RightHand = max(0.0, confidence_RightHand - 1.0 / 15.0)
                        else:
                            confidence_RightHand = 0.0                        
                        Confidence_RightHand.append(confidence_RightHand)                        
                    Confidence_RightHand = torch.tensor(Confidence_RightHand, device=data.device, dtype=data.dtype)

                    # LeftHand Confidence
                    Confidence_LeftHand = []
                    confidence_LeftHand = 0.0
                    Confidence_LeftHand.append(confidence_LeftHand)
                    for i in range(data.shape[0] - 1):
                        if torch.equal(data[i+1, 6:9], data[i, 6:9]):
                            if torch.all(data[i+1, 6:9]==0) == False and torch.all(data[i+1, 0:3]==0) == False:
                                if data[i+1, 9].item() == 1:  # if 1, increase
                                    confidence_LeftHand = min(1.0, confidence_LeftHand + 1.0 / 15.0)
                                else:  # if 0, decrease
                                    confidence_LeftHand = max(0.0, confidence_LeftHand - 1.0 / 15.0)
                            else:
                                confidence_LeftHand = max(0.0, confidence_LeftHand - 1.0 / 15.0)
                        else:
                            confidence_LeftHand = 0.0                        
                        Confidence_LeftHand.append(confidence_LeftHand)                        
                    Confidence_LeftHand = torch.tensor(Confidence_LeftHand, device=data.device, dtype=data.dtype)  
                                        
                    # Compute the bool feature
                    first_data_frame = data[0:1].repeat(15, 1) 
                    data = torch.cat([first_data_frame, data], dim=0)                                                    
                    original_bool_data = data[:, 9:15].clone()
                    continuous_bool_data = torch.zeros_like(original_bool_data)
                    
                    for feature_idx in range(6):
                        cum_sum = torch.cumsum(data[:, 9+feature_idx], dim=0)
                        continuous_bool_data[15:, feature_idx] = (cum_sum[15:] - cum_sum[:-15]) / 15                                            
                    data[:, 9:15] = continuous_bool_data                              
                    data = data[15:]
                    data_clone = data.clone()                     
                    
                    # Save data
                    feature = data_clone[:, :15]
                    motion_encoder = data_clone[:, 15:15+27]          
                    motion_decoder = data_clone[:, 15+27:15+27+36]                
                    
                    Data = torch.cat((feature, 
                                      Confidence_Head.unsqueeze(-1),
                                      Confidence_RightHand.unsqueeze(-1),
                                      Confidence_LeftHand.unsqueeze(-1),
                                      motion_encoder, 
                                      motion_decoder), -1)
                    first_Data_frame = Data[0:1].repeat(self.WINDOW_SIZE, 1)
                    Data = torch.cat([first_Data_frame, Data], dim=0)
                                                
                    np.save(osp.join(self.data_dir, str(self.count)), Data[-self.CLIP_SIZE:, :].cpu().numpy())
                    self.count = self.count + 1            
                    Data_unfold = torch.transpose(Data.unfold(0, self.CLIP_SIZE, (self.CLIP_SIZE-self.WINDOW_SIZE)//2), 1, 2)             
                    for k in range(Data_unfold.shape[0]):
                        np.save(osp.join(self.data_dir, str(self.count)), Data_unfold[k].cpu().numpy())
                        self.count = self.count + 1                        
                            
            ####################################################################################################################################################################################
            ####################################################################################################################################################################################                                                                            
  
    def generate(self, num_targets=3, target_augmentation=False, joint_training=False):
        start = time.time()
        self.count = 0
        for repertorie in self.repertories:
            folder_split = self.raw_data_dir + repertorie['repertorie'] + '/' + 'split'
            for i in range(repertorie['num']):
                self.devide(repertorie['repertorie'], repertorie['num'], folder_split, i+1, num_targets, target_augmentation, joint_training)

        end = time.time() - start
        print("data generation time="+str(end))
####################################################################################################################################
            
if __name__ == '__main__':
 
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
    
    WINDOW_SIZE = 30
    CLIP_SIZE = 40
    DIRECTORY_ID = 0
    DATA_PARSER = data_parser(REPERTORIES, WINDOW_SIZE, CLIP_SIZE, "BP_GTNet", DIRECTORY_ID)
    
    DATA_PARSER.delete_raw_data_dir()     
    DATA_PARSER.delete_data_dir()     
    # DATA_PARSER.split()    
    # DATA_PARSER.generate()
    # DATA_PARSER.__len__()