import zmq
import time
import torch
import numpy as np

from model import TENet, BP_GTNet

DEVICE = 'cuda:0' # 'cpu', 'cuda:0'
device = torch.device(DEVICE)
torch.set_float32_matmul_precision('high') 

##############################################################################################################
path = "model(trained)/TENet_Ours_10.pt"
checkpoint = torch.load(path, map_location=DEVICE)
model_config = checkpoint['config']
WINDOW_SIZE_TENet = model_config['window_dim'] 
TENet = TENet(head_depth=model_config['head_depth'],
              head_sz=model_config['head_sz'],
              feat_sz=model_config['feat_sz'],
              depth=model_config['depth'],
              condition_dim=model_config['condition_dim'],        
              input_dim=model_config['input_dim'],
              window_dim=model_config['window_dim'],
              target_dim=model_config['target_dim'],
              n_flow_layers=model_config['n_flow_layers'],
              n_encoder_layers=model_config['n_encoder_layers'],
              n_encoder_head=model_config['n_encoder_head'],
              use_null_token=model_config['use_null_token'])
TENet.load_state_dict(checkpoint['model_state_dict'])
TENet = TENet.to(device)
TENet.eval()
###########################################################################x###################################

##############################################################################################################
path = "model(trained)/BP_GTNet_Ours_90.pt"
checkpoint = torch.load(path, map_location=DEVICE)
model_config = checkpoint['config']
WINDOW_SIZE_BP_GTNet = model_config["window_size"]    
WINDOW_SIZE_USER_POSE = model_config["user_pose_window_size"]
BP_GTNet = BP_GTNet(rng=np.random.RandomState(23456),
              num_experts=model_config["num_experts"],
              num_experts_part=model_config["num_experts_part"],
              window_size=model_config["window_size"], 
              user_pose_window_size=model_config["user_pose_window_size"], 
              layer_size_encoder=model_config["layer_size_encoder"],
              input_size_encoder=model_config["input_size_encoder"],               
              head_codebook_channels=model_config["head_codebook_channels"],
              head_codebook_dimension=model_config["head_codebook_dimension"],           
              head_temperature=model_config["head_temperature"], 
              hand_hidden_size=model_config["hand_hidden_size"],           
              hand_codebook_size=model_config["hand_codebook_size"], 
              hand_codebook_layer_size=model_config["hand_codebook_layer_size"], 
              model_gating=model_config["model_gating"], 
              layer_size_gating=model_config["layer_size_gating"], 
              input_size_gating=model_config["input_size_gating"],       
              hidden_size_gating=model_config["hidden_size_gating"],       
              input_size_moe=model_config["input_size_moe"], 
              use_cuda=model_config["use_cuda"])
BP_GTNet.load_state_dict(checkpoint['model_state_dict'])
BP_GTNet = BP_GTNet.to(device)
BP_GTNet.eval()
############################################################################################################## 

server_address = 'tcp://*:12345'
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(server_address)
print("server initiated")

with torch.inference_mode():
    NUM_TARGET = 4
    TARGET_COMBO_SIZE = NUM_TARGET**3
    X_TENet_Progression = torch.ones(1,WINDOW_SIZE_TENet,3).to(device)
    X_BP_GTNet_Progression = torch.ones(1,WINDOW_SIZE_BP_GTNet,6).to(device)
    X_BP_GTNet_Progression[:,:,0:3] = 0
    Confidence_TENet = torch.ones(1,WINDOW_SIZE_BP_GTNet,1).to(device)    
    
    while True:
        start_time = time.time()
        msg = socket.recv()
        message = np.fromstring(msg, dtype=np.float32, sep=',')        
        X = torch.from_numpy(message).to(device=device, dtype=torch.float32, non_blocking=True)
        X = X.unsqueeze(0)
                
        X_TENet = (X[:,:WINDOW_SIZE_TENet*12]).reshape(1,WINDOW_SIZE_TENet,12)
        new_prog_tenet = torch.mean(X_TENet[0, -15:, -3:], dim=0)  # (3,)
        X_TENet_Progression = torch.roll(X_TENet_Progression, shifts=-1, dims=1)
        X_TENet_Progression[0, -1, :] = new_prog_tenet
        X_TENet[:, :, -3:] = X_TENet_Progression                     
        X_TENet_batch = X_TENet.repeat(TARGET_COMBO_SIZE, 1, 1) 
                
        Y_TENet = (X[:,-(TARGET_COMBO_SIZE*9):]).reshape(TARGET_COMBO_SIZE,9)
        # start_time = time.time()
        max_indice, confidence_TENet = TENet.infer_target(X_TENet_batch, Y_TENet)
        # print("Total: " + str(time.time()-start_time))                       
        
        Confidence_TENet = torch.roll(Confidence_TENet, shifts=-1, dims=1)
        Confidence_TENet[0, -1, 0] = confidence_TENet             
        
        X_BP_GTNet = (X[:,WINDOW_SIZE_TENet*12 : WINDOW_SIZE_TENet*12 + WINDOW_SIZE_BP_GTNet*21]).reshape(1,WINDOW_SIZE_BP_GTNet,21)
        X_Target_User = X_BP_GTNet[:,:,-3:]
        Confidence = X_BP_GTNet[:,:,-6:-3]
        X_BP_GTNet = X_BP_GTNet[:,:,:15]
        
        new_prog_um = torch.mean(X_BP_GTNet[0, -15:, -6:], dim=0)  # (6,)
        X_BP_GTNet_Progression = torch.roll(X_BP_GTNet_Progression, shifts=-1, dims=1)
        X_BP_GTNet_Progression[0, -1, :] = new_prog_um
        X_BP_GTNet[:, :, -6:] = X_BP_GTNet_Progression
        X_Target_Avatar = X_BP_GTNet
        X_Pose_User = (X[:, WINDOW_SIZE_TENet*12 + WINDOW_SIZE_BP_GTNet*21 : WINDOW_SIZE_TENet*12 + WINDOW_SIZE_BP_GTNet*21 + WINDOW_SIZE_USER_POSE*27]).reshape(1,WINDOW_SIZE_USER_POSE,27)        
        X_Pose_Avatar = (X[:, WINDOW_SIZE_TENet*12 + WINDOW_SIZE_BP_GTNet*21 + WINDOW_SIZE_USER_POSE*27 : WINDOW_SIZE_TENet*12 + WINDOW_SIZE_BP_GTNet*21 + WINDOW_SIZE_USER_POSE*27 + 36]).reshape(1,36)
        # start_time = time.time()
        output_data = BP_GTNet(X_Target_User, X_Target_Avatar, X_Pose_User, X_Pose_Avatar, Confidence, False)
        # print("Total: " + str(time.time()-start_time))                
        output_data = torch.cat((output_data, X_Target_Avatar[:,-1,-6:], Confidence_TENet[:,-1,:], max_indice.unsqueeze(0).unsqueeze(0)), -1)        
        output_data = output_data.detach().cpu().numpy()
        output_data = np.squeeze(output_data)
        output = ','.join(str(x) for x in output_data)
                                 
        socket.send_string(output)
        time.sleep(0.018) # Target 4
        # time.sleep(0.012) # Target 5
        # time.sleep(0.006) # Target 6
        print("Total: " + str(time.time()-start_time))                       
        
        