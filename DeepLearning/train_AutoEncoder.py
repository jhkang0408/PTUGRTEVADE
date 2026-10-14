import os
import os.path as osp
import gc
import time
import datetime 
import numpy as np
from tqdm import tqdm 

import torch
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from model import ConvolutionalAutoEncoder

class Dataset_AutoEncoder(Dataset):
    def __init__(self, train_data, clip_length, gpu_id):
        self.gpu_id = gpu_id
        self.train_data = train_data
        if train_data:
            self.folder_name = 'data/train_AutoEncoder_'+str(gpu_id)
        else:
            self.folder_name = 'data/test_AutoEncoder_'+str(gpu_id)          
        self.count = 0
        self.paths = []
        
        self.repertories_train = [{'repertorie': 'Pointing at single target (right)', 'num': 27},                   
                                  {'repertorie': 'Pointing at two targets (right)', 'num': 210},
                                  {'repertorie': 'Pointing at single target with gaze shift (right)', 'num': 180},
                                  {'repertorie': 'Pointing at single target with explanation (right)', 'num': 126},     
                                  {'repertorie': 'Pointing at single target with explanation shift (right)', 'num': 126},                   
                                  {'repertorie': 'Pointing at single target with explanation shift shift (right)', 'num': 126},                   
                                  {'repertorie': 'Transition (Pointing) (right)', 'num': 15},
                                  {'repertorie': 'Pointing at single target (left)', 'num': 27},
                                  {'repertorie': 'Pointing at two targets (left)', 'num': 210},
                                  {'repertorie': 'Pointing at single target with gaze shift (left)', 'num': 180},
                                  {'repertorie': 'Pointing at single target with explanation (left)', 'num': 126},
                                  {'repertorie': 'Pointing at single target with explanation shift (left)', 'num': 126},
                                  {'repertorie': 'Pointing at single target with explanation shift shift (left)', 'num': 126},                   
                                  {'repertorie': 'Transition (Pointing) (left)', 'num': 15},
                                  {'repertorie': 'Touching single target (right)', 'num': 30},                   
                                  {'repertorie': 'Touching two targets (right)', 'num': 210},
                                  {'repertorie': 'Touching single target with gaze shift (right)', 'num': 210},
                                  {'repertorie': 'Touching single target with explanation (right)', 'num': 126},     
                                  {'repertorie': 'Touching single target with explanation shift shift (right)', 'num': 126},                   
                                  {'repertorie': 'Transition (Touching) (right)', 'num': 15},
                                  {'repertorie': 'Touching single target (left)', 'num': 30},
                                  {'repertorie': 'Touching two targets (left)', 'num': 210},
                                  {'repertorie': 'Touching single target with gaze shift (left)', 'num': 210},
                                  {'repertorie': 'Touching single target with explanation (left)', 'num': 126},
                                  {'repertorie': 'Touching single target with explanation shift shift (left)', 'num': 126},                   
                                  {'repertorie': 'Transition (Touching) (left)', 'num': 15}]
        self.repertories_test1  = [{'repertorie': 'Pointing at single target', 'num': 27},
                                   {'repertorie': 'Pointing at two targets', 'num': 8},
                                   {'repertorie': 'Pointing at single target with gaze shift', 'num': 8},
                                   {'repertorie': 'Pointing at single target with explanation', 'num': 8},
                                   {'repertorie': 'Touching single target', 'num': 30},
                                   {'repertorie': 'Touching single target with gaze shift', 'num': 8},
                                   {'repertorie': 'Touching single target with explanation', 'num': 8},
                                   {'repertorie': 'Touching two targets', 'num': 8}]
        self.repertories_test2 = [{'repertorie': 'Pointing at single target with explanation shift shift', 'num': 8},
                                  {'repertorie': 'Touching single target with explanation shift shift', 'num': 8}]        
        self.L= clip_length
        self.raw_data_dir = 'data(raw)/'+'AutoEncoder/'
        
    def split(self):
        if self.train_data == True:
            for repertorie in self.repertories_train:
                folder_raw = self.raw_data_dir + 'train/' + repertorie['repertorie'] + '/' + 'raw'
                folder_split = self.raw_data_dir + 'train/' + repertorie['repertorie'] + '/' + 'split'
                for f in os.listdir(folder_split):    
                    os.remove(os.path.join(folder_split, f))                    
                input_data = np.loadtxt(osp.join(folder_raw, 'Data.txt')).astype(np.float32)
                sequences = np.loadtxt(osp.join(folder_raw, 'Sequences.txt')).astype(np.float32)
                for i in range(1, int(np.max(sequences))+1):
                    np.savetxt(osp.join(folder_split, 'Data'+str(i)+'.txt'), input_data[np.where(sequences==i)], fmt='%1.4e')        
        else:
            data_dir_list = ['test/161/', 'test/172/', 'test/173/', 'test/174/', 'test/180/',
                             'test/158(HJ)/', 'test/165(YN)/', 'test/166(HS)/', 'test/174(KB)/', 'test/175(RS)/', 'test/180(DH)/', 'test/187(ES)/']
            for data_dir in data_dir_list:
                if data_dir == 'test/161/' or data_dir == 'test/172/' or data_dir == 'test/173/' or data_dir == 'test/174/' or data_dir == 'test/180/':
                    repertories = self.repertories_test1
                else:
                    repertories = self.repertories_test2                       
                for repertorie in repertories:
                    folder_raw = self.raw_data_dir + data_dir + repertorie['repertorie'] + '/' + 'raw'
                    folder_split = self.raw_data_dir + data_dir + repertorie['repertorie'] + '/' + 'split'
                    for f in os.listdir(folder_split):    
                        os.remove(os.path.join(folder_split, f))                    
                    input_data = np.loadtxt(osp.join(folder_raw, 'Data.txt')).astype(np.float32)
                    sequences = np.loadtxt(osp.join(folder_raw, 'Sequences.txt')).astype(np.float32)
                    for i in range(1, int(np.max(sequences))+1):
                        np.savetxt(osp.join(folder_split, 'Data'+str(i)+'.txt'), input_data[np.where(sequences==i)], fmt='%1.4e')   
    
    def delete(self):        
        for f in os.listdir(self.folder_name):    
            os.remove(os.path.join(self.folder_name, f))  
                
    def generate(self):
        if self.train_data == True:
            for repertorie in self.repertories_train:
                folder_split = self.raw_data_dir + 'train/' + repertorie['repertorie'] + '/' + 'split'                        
                for i in range(len(os.listdir(folder_split))):                
                    data = np.loadtxt(osp.join(folder_split, 'Data'+str(i+1)+'.txt')).astype(np.float32)
                    data = torch.tensor(data, dtype=torch.float32)                 
                    data = torch.cat((data[:,42:51], data[:,54:63], data[:,66:75]), -1)
                    np.save(osp.join(self.folder_name, str(self.count)), data[-self.L:,:].numpy())
                    self.count = self.count + 1
                    
                    data = torch.transpose(data.unfold(0, self.L, int(self.L/2)), 1, 2)
                    for k in range(data.shape[0]):
                        np.save(osp.join(self.folder_name, str(self.count)), data[k].numpy())
                        self.count = self.count + 1                  
        else:
            data_dir_list = ['test/161/', 'test/172/', 'test/173/', 'test/174/', 'test/180/',
                             'test/158(HJ)/', 'test/165(YN)/', 'test/166(HS)/', 'test/174(KB)/', 'test/175(RS)/', 'test/180(DH)/', 'test/187(ES)/']
            for data_dir in data_dir_list:
                if data_dir == 'test/161/' or data_dir == 'test/172/' or data_dir == 'test/173/' or data_dir == 'test/174/' or data_dir == 'test/180/':
                    repertories = self.repertories_test1
                else:
                    repertories = self.repertories_test2                        
                for repertorie in repertories:
                    folder_split = self.raw_data_dir + data_dir + repertorie['repertorie'] + '/' + 'split'
                    for i in range(len(os.listdir(folder_split))):                
                        data = np.loadtxt(osp.join(folder_split, 'Data'+str(i+1)+'.txt')).astype(np.float32)
                        data = torch.tensor(data, dtype=torch.float32)               
                        data = torch.cat((data[:,42:51], data[:,54:63], data[:,66:75]), -1)
                        np.save(osp.join(self.folder_name, str(self.count)), data[-self.L:,:].numpy())
                        self.count = self.count + 1
                        
                        data = torch.transpose(data.unfold(0, self.L, int(self.L/2)), 1, 2)
                        for k in range(data.shape[0]):
                            np.save(osp.join(self.folder_name, str(self.count)), data[k].numpy())
                            self.count = self.count + 1                
        self.paths = [osp.join(self.folder_name, str(i)+'.npy') for i in range(self.count)] 
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):        
        X = torch.from_numpy(np.load(self.paths[idx]))
        return {'X': X}

def sixd2matrot(pose_6d):
    # pose_6d: (N, L, J, 6), the first 3 values are forward, the last 3 the (raw) up vector
    fwd = F.normalize(pose_6d[..., 0:3], dim=-1)                      # b1
    up_raw = pose_6d[..., 3:6]

    # orthogonalize up against forward, then normalize
    up = up_raw - (fwd * up_raw).sum(dim=-1, keepdim=True) * fwd      # b2'
    up = F.normalize(up, dim=-1)                                      # b2

    # right = up x forward (right-handed frame)
    right = torch.cross(up, fwd, dim=-1)
    right = F.normalize(right, dim=-1)

    # re-derive up to minimize the accumulated error
    up = torch.cross(fwd, right, dim=-1)

    # column vectors: (right, up, forward)
    pose_matrot = torch.stack([right, up, fwd], dim=-1)               # (..., 3, 3)
    return pose_matrot

def rot6_geodesic_deg(A_pose_6d, B_pose_6d):
    N, T, _ = A_pose_6d.shape
    J = 3  # Head / Right / Left
    A = A_pose_6d.reshape(N, T, J, 6)
    B = B_pose_6d.reshape(N, T, J, 6)

    Ra = sixd2matrot(A)                        # (N,T,J,3,3)
    Rb = sixd2matrot(B)                        # (N,T,J,3,3)

    Rrel = Ra.transpose(-1, -2) @ Rb                # (N,T,J,3,3)
    trace = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    ang = torch.acos(cos)                           # radians, 0~pi
    return torch.rad2deg(ang)                       # degrees, 0~180    

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--numEpoch', type=int, default=600, help='epoch size for training')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size for training')
parser.add_argument('--hiddenSize', type=int, default=512, help='Hidden Size')
parser.add_argument('--kernelSize', type=int, default=15, help='Hidden Size')
parser.add_argument('--L', type=int, default=30, help='Clip Length')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
args = parser.parse_args()

if __name__ == '__main__':

    gc.collect()
    torch.cuda.empty_cache() 
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%m%d_%H%M%S')   
    
    # Training setup.
    GPU_ID = args.gpu_id
    SHUFFLE = args.shuffle
    NUM_EPOCHS = args.numEpoch
    BATCH_SIZE = args.batchSize    
    L = args.L
    LR = args.lr  # learning rate
            
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')        

    INPUT_SIZE=27
    OUTPUT_SIZE=27
    HIDDEN_SIZE=args.hiddenSize
    KERNEL_SIZE=args.kernelSize
    model = ConvolutionalAutoEncoder(INPUT_SIZE, HIDDEN_SIZE, KERNEL_SIZE)
    model = model.to(device)
    
    save_dir = "AutoEncoder,"+str(HIDDEN_SIZE)+","+str(KERNEL_SIZE)+","+str(NUM_EPOCHS)+","+str(BATCH_SIZE)+","+str(LR)+","+nowDatetime
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_config = {"input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "kernel_size": KERNEL_SIZE}


    train_data_set = Dataset_AutoEncoder(True, L, GPU_ID) 
    train_data_set.delete()
    train_data_set.split()
    train_data_set.generate()    
    train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=16, pin_memory=True, drop_last=True)
    print('# of train example: {}, # of batches {}'.format(len(train_data_set), len(train_data_set) // BATCH_SIZE))

    test_data_set = Dataset_AutoEncoder(False, L, GPU_ID) 
    test_data_set.delete()
    test_data_set.split()
    test_data_set.generate()
    test_data_loader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    print('# of test example: {}, # of batches {}'.format(len(test_data_set), len(test_data_set) // BATCH_SIZE)) 
            
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.999)
    train_start = time.time()
    for epoch in range(1, NUM_EPOCHS+1): 
        # Train
        epoch_start = time.time()                
        model.train()    
        epoch_loss = 0    
        epoch_pe = 0
        epoch_re = 0
        pbar = tqdm(train_data_loader)
        for iter, X in enumerate(pbar):                         
            X = X['X'].to(device)                   
            X_hat = model(X, False)
            loss = F.l1_loss(X_hat, X)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()                            
            pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, iter+1, len(train_data_loader)))
            pbar.set_postfix({"loss":((loss))})  
            epoch_loss += loss.item()

            # position error
            X_pos_hat = torch.reshape(torch.cat((X_hat[:,:,6:9], X_hat[:,:,15:18], X_hat[:,:,24:27]), -1), (BATCH_SIZE, L, 3, 3))  
            X_pos = torch.reshape(torch.cat((X[:,:,6:9], X[:,:,15:18], X[:,:,24:27]), -1), (BATCH_SIZE, L, 3, 3))  
            pe = torch.mean(torch.sqrt(torch.sum(torch.square(X_pos-X_pos_hat),axis=-1)))*100
            epoch_pe += pe
            # rotation error
            X_rot_hat = torch.cat((X_hat[:,:,0:6], X_hat[:,:,9:15], X_hat[:,:,18:24]), -1)
            X_rot = torch.cat((X[:,:,0:6], X[:,:,9:15], X[:,:,18:24]), -1)
            rot_diff = rot6_geodesic_deg(X_rot_hat, X_rot)
            re = torch.mean(rot_diff)
            epoch_re += re
        
        scheduler.step()
        train_time = time.time() - epoch_start
        epoch_loss /= (len(train_data_loader))
        epoch_pe /= (len(train_data_loader))
        epoch_re /= (len(train_data_loader))
        log = "Train Time = %.1f, [Epoch %d/%d] [Train Loss: %.7f] [Position Error: %.3f] [Rotation Error: %.3f]" % (train_time, epoch, NUM_EPOCHS, epoch_loss, epoch_pe, epoch_re)
        print(log)     
        
        # Test
        epoch_start = time.time()       
        model.eval()
        epoch_loss = 0
        epoch_pe = 0
        epoch_re = 0
        pbar = tqdm(test_data_loader)
        with torch.no_grad():
            for iter, X in enumerate(pbar):            
                X = X['X'].to(device)                  
                X_hat = model(X, False)
                loss = F.l1_loss(X_hat, X)                         
                pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, iter+1, len(test_data_loader)))
                pbar.set_postfix({"loss":((loss))})  
                epoch_loss += loss.item()
                
                # position error
                X_pos_hat = torch.reshape(torch.cat((X_hat[:,:,6:9], X_hat[:,:,15:18], X_hat[:,:,24:27]), -1), (BATCH_SIZE, L, 3, 3))  
                X_pos = torch.reshape(torch.cat((X[:,:,6:9], X[:,:,15:18], X[:,:,24:27]), -1), (BATCH_SIZE, L, 3, 3))  
                pe = torch.mean(torch.sqrt(torch.sum(torch.square(X_pos-X_pos_hat),axis=-1)))*100
                epoch_pe += pe
                # rotation error
                X_rot_hat = torch.cat((X_hat[:,:,0:6], X_hat[:,:,9:15], X_hat[:,:,18:24]), -1)
                X_rot = torch.cat((X[:,:,0:6], X[:,:,9:15], X[:,:,18:24]), -1)
                rot_diff = rot6_geodesic_deg(X_rot_hat, X_rot)
                re = torch.mean(rot_diff)
                epoch_re += re
                
        test_time = time.time() - epoch_start
        epoch_loss /= (len(test_data_loader))
        epoch_pe /= (len(test_data_loader))
        epoch_re /= (len(test_data_loader))
        log = "Test Time = %.1f, [Epoch %d/%d] [Test Loss: %.7f] [Position Error: %.3f] [Rotation Error: %.3f]" % (test_time, epoch, NUM_EPOCHS, epoch_loss, epoch_pe, epoch_re)
        print(log)                         
        
        # Model Save
        torch.save({'model_state_dict': model.state_dict(), 'config': model_config}, os.path.join(save_dir, "AutoEncoder_"+str(epoch)+".pt"))
        
    print("Total Time= %.2f" %(time.time()-train_start))      