import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
import numpy as np
import scipy.linalg
from torch.nn.parameter import Parameter
import math
import random
from random import randrange

# ============================================================================
# Body-part Translation Network (BP_GTNet)
# ============================================================================
class BP_GTNet(nn.Module):
    def __init__(self,
                 rng, 
                 num_experts,
                 num_experts_part,
                 window_size,
                 user_pose_window_size,
                 layer_size_encoder,
                 input_size_encoder,                 
                 head_codebook_channels, 
                 head_codebook_dimension,
                 head_temperature,
                 hand_hidden_size,
                 hand_codebook_size,
                 hand_codebook_layer_size,   
                 model_gating,
                 layer_size_gating,
                 input_size_gating,
                 hidden_size_gating,
                 input_size_moe,
                 use_cuda=False,
                 **kwargs):
        super(BP_GTNet, self).__init__()       
        head_hidden_size = head_codebook_channels * head_codebook_dimension
        self.encoder = Encoder_BP_GTNet(window_size = user_pose_window_size,
                                     layer_size = layer_size_encoder,
                                     input_size = input_size_encoder,                     
                                     head_codebook_channels = head_codebook_channels,
                                     head_codebook_dimension = head_codebook_dimension,
                                     head_temperature = head_temperature,
                                     hand_hidden_size = hand_hidden_size,
                                     hand_codebook_size = hand_codebook_size,
                                     hand_codebook_layer_size = hand_codebook_layer_size)
        if model_gating == 'Transformer':
            self.gating_network = GatingNetwork_BP_GTNet_Transformer(layer_size_gating, input_size_gating+3, hidden_size_gating, num_experts)        
            self.gating_network_head = GatingNetwork_BP_GTNet_Transformer(layer_size_gating, 5, hidden_size_gating//2, num_experts_part)
            self.gating_network_righthand = GatingNetwork_BP_GTNet_Transformer(layer_size_gating, 5, hidden_size_gating//2, num_experts_part)
            self.gating_network_lefthand = GatingNetwork_BP_GTNet_Transformer(layer_size_gating, 5, hidden_size_gating//2, num_experts_part)
        elif model_gating == 'GRU':
            self.gating_network = GatingNetwork_BP_GTNet_GRU(layer_size_gating, input_size_gating+3, hidden_size_gating, num_experts)        
            self.gating_network_head = GatingNetwork_BP_GTNet_GRU(layer_size_gating, 5, hidden_size_gating//2, num_experts_part)
            self.gating_network_righthand = GatingNetwork_BP_GTNet_GRU(layer_size_gating, 5, hidden_size_gating//2, num_experts_part)
            self.gating_network_lefthand = GatingNetwork_BP_GTNet_GRU(layer_size_gating, 5, hidden_size_gating//2, num_experts_part)
            
        self.moe_network = ExpertMLP_BP_GTNet(rng=rng,
                                           num_experts = num_experts,
                                           input_size = 27 + head_hidden_size + 2*hand_hidden_size + (input_size_gating+3)*window_size,
                                           hidden_size = (27 + head_hidden_size + 2*hand_hidden_size + (input_size_gating+3)*window_size)//2,
                                           output_size = 36)
        self.moe_network_head = ExpertMLP_BP_GTNet(rng=rng,
                                                num_experts = num_experts_part,
                                                input_size = 9 + head_hidden_size + 5*window_size,
                                                hidden_size = (9 + head_hidden_size + 5*window_size)//2,
                                                output_size = 9)
        self.moe_network_righthand = ExpertMLP_BP_GTNet(rng=rng,
                                                num_experts = num_experts_part,
                                                input_size = 9 + hand_hidden_size + 5*window_size,
                                                hidden_size = (9 + hand_hidden_size + 5*window_size)//2,
                                                output_size = 9)
        self.moe_network_lefthand = ExpertMLP_BP_GTNet(rng=rng,
                                                    num_experts = num_experts_part,
                                                    input_size = 9 + hand_hidden_size + 5*window_size,
                                                    hidden_size = (9 + hand_hidden_size + 5*window_size)//2,
                                                    output_size = 9)               

        self.head_target_null_token = nn.Parameter(torch.randn(1, 3))
        self.righthand_target_null_token = nn.Parameter(torch.randn(1, 3))
        self.lefthand_target_null_token = nn.Parameter(torch.randn(1, 3))
        
    def forward(self, X_Target_User, X_Target_Avatar, X_Pose_User, X_Pose_Avatar, Confidence, IsTrain):             
        B, L, _ = X_Target_Avatar.shape
        
        ####################################################################################################################################################################################
        head_target_avatar = X_Target_Avatar[:,:,0:3]
        righthand_target_avatar = X_Target_Avatar[:,:,3:6]
        lefthand_target_avatar = X_Target_Avatar[:,:,6:9]
        
        # Null Token
        head_target_null_token_expanded = self.head_target_null_token.expand(B, L, -1)
        righthand_target_null_token_expanded = self.righthand_target_null_token.expand(B, L, -1)
        lefthand_target_null_token_expanded = self.lefthand_target_null_token.expand(B, L, -1)     

        new_head_target_avatar = torch.where(torch.all(head_target_avatar == 0, dim=2).unsqueeze(-1), head_target_null_token_expanded.detach(), head_target_avatar)
        new_righthand_target_avatar = torch.where(torch.all(righthand_target_avatar == 0, dim=2).unsqueeze(-1), righthand_target_null_token_expanded.detach(), righthand_target_avatar)
        new_lefthand_target_avatar = torch.where(torch.all(lefthand_target_avatar == 0, dim=2).unsqueeze(-1), lefthand_target_null_token_expanded.detach(), lefthand_target_avatar)
        
        # Refined Target      
        new_X_Target_Avatar = torch.cat((new_head_target_avatar, new_righthand_target_avatar, new_lefthand_target_avatar, X_Target_Avatar[:,:,9:].clone()), dim=2)
        ####################################################################################################################################################################################

        ####################################################################################################################################################################################        
        Z_head, Z_righthand, Z_lefthand, losses_vq = self.encoder(X_Pose_User[:,:,0:9], 
                                                                  X_Pose_User[:,-1,9:18], 
                                                                  X_Pose_User[:,-1,18:27],
                                                                  X_Target_User,
                                                                  IsTrain)      
        ####################################################################################################################################################################################                        
               
        if IsTrain:
            Z_Pose_User = torch.cat((Z_head.detach(), Z_righthand.detach(), Z_lefthand.detach()), -1)
        else:
            Z_Pose_User = torch.cat((Z_head, Z_righthand, Z_lefthand), -1)
            
        Y = self.moe_network(torch.cat((X_Pose_Avatar[:,0:9],
                                        X_Pose_Avatar[:,12:21],
                                        X_Pose_Avatar[:,24:33],
                                        Z_Pose_User,
                                        Confidence.reshape(B,-1),
                                        new_X_Target_Avatar.reshape(B,-1)),-1),
                             self.gating_network(torch.cat((new_X_Target_Avatar, Confidence), -1)))
                      
        if IsTrain==True:
            head_target_avatar = X_Target_Avatar[:,:,0:3]
            righthand_target_avatar = X_Target_Avatar[:,:,3:6]
            lefthand_target_avatar = X_Target_Avatar[:,:,6:9]
            
            # Null Token   
            new_head_target_avatar = torch.where(torch.all(head_target_avatar == 0, dim=2).unsqueeze(-1), head_target_null_token_expanded, head_target_avatar)            
            new_righthand_target_avatar = torch.where(torch.all(righthand_target_avatar == 0, dim=2).unsqueeze(-1), righthand_target_null_token_expanded, righthand_target_avatar)
            new_lefthand_target_avatar = torch.where(torch.all(lefthand_target_avatar == 0, dim=2).unsqueeze(-1), lefthand_target_null_token_expanded, lefthand_target_avatar)
                     
            X_Target_Avatar_Head = torch.cat((new_head_target_avatar, X_Target_Avatar[:,:,[9,12]]), -1)
            X_Target_Avatar_RightHand = torch.cat((new_righthand_target_avatar, X_Target_Avatar[:,:,[10,13]]), -1)
            X_Target_Avatar_LeftHand = torch.cat((new_lefthand_target_avatar, X_Target_Avatar[:,:,[11,14]]), -1)
            
            Y_Head = self.moe_network_head(torch.cat((X_Pose_Avatar[:,0:9], 
                                                      Z_head,
                                                      X_Target_Avatar_Head.reshape(B,-1)), -1),                                           
                                           self.gating_network_head(X_Target_Avatar_Head))
            Y_RightHand = self.moe_network_righthand(torch.cat((X_Pose_Avatar[:,12:21], 
                                                                Z_righthand,
                                                                X_Target_Avatar_RightHand.reshape(B,-1)), -1), 
                                                     self.gating_network_righthand(X_Target_Avatar_RightHand))
            Y_LeftHand = self.moe_network_lefthand(torch.cat((X_Pose_Avatar[:,24:33], 
                                                              Z_lefthand,
                                                              X_Target_Avatar_LeftHand.reshape(B,-1)), -1), 
                                                   self.gating_network_lefthand(X_Target_Avatar_LeftHand))
            
            return Y, Y_Head, Y_RightHand, Y_LeftHand, losses_vq
        else:
            return Y

class Encoder_BP_GTNet(nn.Module):
    def __init__(self, 
                 window_size, 
                 layer_size, 
                 input_size, 
                 head_codebook_channels, 
                 head_codebook_dimension, 
                 head_temperature,
                 hand_hidden_size, 
                 hand_codebook_size, 
                 hand_codebook_layer_size,                                  
                 **kwargs):
        super(Encoder_BP_GTNet, self).__init__()         
        self.C = head_codebook_channels
        self.D = head_codebook_dimension       
        self.temperature = head_temperature
        self.head_token = nn.Parameter(torch.randn(1, self.C*self.D))
        self.righthand_token = nn.Parameter(torch.randn(1, hand_hidden_size))
        self.lefthand_token = nn.Parameter(torch.randn(1, hand_hidden_size))
        
        self.head_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.C*self.D, nhead=1, dim_feedforward=self.C*self.D, batch_first=True, norm_first=True), layer_size)
        self.hand_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hand_hidden_size, nhead=1, dim_feedforward=hand_hidden_size, batch_first=True, norm_first=True), layer_size)

        self.head_default_token = nn.Parameter(torch.randn(1, self.C*self.D))         
        self.head_transition_token = nn.Parameter(torch.randn(1, self.C*self.D))
        
        self.head_embedding = nn.Linear(input_size//3, self.C*self.D)
        self.righthand_embedding = nn.Linear(input_size//3, hand_hidden_size)
        self.lefthand_embedding = nn.Linear(input_size//3, hand_hidden_size)
        
        self.righthand_vqvae = ResidualVQVAE(codebook_dim=hand_hidden_size,
                                             num_codebooks=hand_codebook_size,
                                             num_residual_layers=hand_codebook_layer_size)
        self.lefthand_vqvae = ResidualVQVAE(codebook_dim=hand_hidden_size,
                                            num_codebooks=hand_codebook_size,
                                            num_residual_layers=hand_codebook_layer_size)        

    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.reshape(-1,1,1,1) #This is noise scale between 0 and 1
        noise = torch.rand_like(tensor) - 0.5 #This is random noise between -0.5 and 0.5
        samples = scale * noise + 0.5 #This is noise rescaled between 0 and 1 where 0.5 is default for 0 noise
        return -torch.log(-torch.log(samples + eps) + eps)        
    
    def gumbel_softmax_sample(self, logits, scale):
        y = logits + self.sample_gumbel(logits, scale)
        return F.softmax(y / self.temperature, dim=-1)
    
    def gumbel_softmax(self, logits, scale):
        y = self.gumbel_softmax_sample(logits, scale)
        y_soft = y.view(logits.shape)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        y_hard = y_hard.view(logits.shape)
        return y_soft, y_hard 
        
    def sample(self, z, knn):        
        z = z.reshape(-1, self.C, self.D)
        z = z.unsqueeze(0).repeat(knn.size(0), 1, 1, 1)
        z_soft, z_hard = self.gumbel_softmax(z, knn)
        z_soft = z_soft.reshape(-1, self.C*self.D)
        z_hard = z_hard.reshape(-1, self.C*self.D)
        return z_soft, z_hard  
                
    def forward(self, 
                X_Pose_User_head, 
                X_Pose_User_righthand, 
                X_Pose_User_lefthand,
                X_Target_User,
                is_training):
        
        B, L, _ = X_Pose_User_head.shape
        
        head_token = self.head_token.expand(B, -1)  
        head_default_token = self.head_default_token.expand(B, -1)                
        head_transition_token = self.head_transition_token.expand(B, -1)                
                
        # Head        
        X_head = []
        for i in range(L):                  
            head_target_user_change = torch.any(X_Target_User[:, -L+i, :] != X_Target_User[:, -L+i-1, :], dim=1)
            is_head_target_user_zero = torch.all(X_Target_User[:, -L+i, :] == 0, dim=1)
            X_head_ = torch.where(head_target_user_change.unsqueeze(1),
                                  head_transition_token,
                                  torch.where(is_head_target_user_zero.unsqueeze(1),
                                              head_default_token,
                                              self.head_embedding(X_Pose_User_head[:, i]))).unsqueeze(1)
            X_head.append(X_head_)
        X_head = torch.cat(X_head, dim=1)    
        X_head = torch.cat((head_token.unsqueeze(1), X_head), dim=1)  
        head_token = self.head_transformer(X_head)[:,0]
        
        # Hand
        righthand_token = self.righthand_token.expand(B, -1)
        lefthand_token = self.lefthand_token.expand(B, -1)            
        X_righthand = self.righthand_embedding(X_Pose_User_righthand)
        X_lefthand = self.lefthand_embedding(X_Pose_User_lefthand)        
        X_hand = torch.cat((righthand_token.unsqueeze(1), lefthand_token.unsqueeze(1), X_righthand.unsqueeze(1), X_lefthand.unsqueeze(1)), 1)  
        hand_token = self.hand_transformer(X_hand)[:,:2]
        
        # Apply Residual VQ-VAE to each body part
        if is_training:
            # During training, apply VQ-VAE and get quantized results            
            _, Z_head = self.sample(head_token, torch.ones(1, device=X_Pose_User_head.device))    
            Z_righthand, vq_loss_righthand = self.righthand_vqvae(hand_token[:,0], is_training=True)
            Z_lefthand, vq_loss_lefthand = self.lefthand_vqvae(hand_token[:,1], is_training=True)
            
            # Collect VQ losses
            losses_vq = {'righthand': vq_loss_righthand,
                         'lefthand': vq_loss_lefthand,
                         'total': vq_loss_righthand + vq_loss_lefthand}
        else:
            # During inference, just return continuous representations
            Z_head, _ = self.sample(head_token, torch.zeros(1, device=X_Pose_User_head.device))    
            Z_righthand, _ = self.righthand_vqvae(hand_token[:,0], is_training=False)
            Z_lefthand, _ = self.lefthand_vqvae(hand_token[:,1], is_training=False)
            losses_vq = None
        
        return Z_head, Z_righthand, Z_lefthand, losses_vq

class ResidualVQVAE(nn.Module):
    """Residual VQ-VAE with multiple codebook layers"""
    def __init__(self, codebook_dim, num_codebooks=128, num_residual_layers=4, commitment_cost=0.25):
        super(ResidualVQVAE, self).__init__()
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        self.num_residual_layers = num_residual_layers
        self.commitment_cost = commitment_cost
        
        # Create residual layers with codebooks
        self.residual_layers = nn.ModuleList([
            VectorQuantizer(codebook_dim, num_codebooks, commitment_cost)
            for _ in range(num_residual_layers)
        ])
        
    def forward(self, z, is_training=True):
        """
        Args:
            z: Input features [B, codebook_dim]
            is_training: Whether in training mode
        Returns:
            output: Quantized representation if training, continuous if inference
            vq_loss: Total VQ loss (only if training)
        """
        
        if is_training:
            # During training, perform actual quantization
            residual = z
            quantized_layers = []
            losses_vq = []
            
            for i, layer in enumerate(self.residual_layers):
                # Quantize the residual
                quantized_residual, vq_loss = layer(residual, is_training=True)
                
                quantized_layers.append(quantized_residual)
                losses_vq.append(vq_loss)
                
                # Compute residual for next layer
                residual = residual - quantized_residual.detach()
            
            # Sum all quantized layers to get final representation
            quantized = sum(quantized_layers)
            
            # Sum VQ losses from all layers
            total_vq_loss = sum(losses_vq)
            
            return quantized, total_vq_loss
        else:
            # During inference, return continuous representation without quantization
            return z, None

class VectorQuantizer(nn.Module):
    """Vector Quantizer module for VQ-VAE"""
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs, is_training=True):
        """
        Args:
            inputs: Input tensor [B, embedding_dim]
            is_training: Whether in training mode
        Returns:
            quantized: Quantized tensor
            vq_loss: Vector quantization loss (only if training)
        """
        B = inputs.shape[0]
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to all codebook entries
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(B, self.embedding_dim)
        
        if is_training:
            # Calculate VQ loss
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
            
            # Straight-through estimator
            quantized = inputs + (quantized - inputs).detach()
            
            return quantized, vq_loss
        else:
            return quantized, None

class GatingNetwork_BP_GTNet_Transformer(nn.Module):
    def __init__(self, layer_size_gating, input_size, hidden_size, output_size, **kwargs):
        super(GatingNetwork_BP_GTNet_Transformer, self).__init__()    
        self.linear_input = nn.Linear(input_size, hidden_size)
        self.linear_output = nn.Linear(hidden_size, output_size)
        self.token = nn.Parameter(torch.randn(1, hidden_size))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, nhead=1, dim_feedforward=hidden_size, batch_first=True, norm_first=True), layer_size_gating)
        
    def forward(self, X):
        token = self.token.expand(X.size(0), -1)          
        X = self.linear_input(X)
        X = torch.cat((token.unsqueeze(1), X), 1)
        X = self.transformer(X)[:,0,:]
        X = self.linear_output(X)
        Y = nn.Softmax(dim=-1)(X)
        return Y

class GatingNetwork_BP_GTNet_GRU(nn.Module):
    def __init__(self, layer_size_gating, input_size, hidden_size, output_size, **kwargs):
        super(GatingNetwork_BP_GTNet_GRU, self).__init__()    
        self.linear_input = nn.Linear(input_size, hidden_size)
        self.linear_output = nn.Linear(hidden_size, output_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=layer_size_gating, batch_first=True)
        
    def forward(self, X):
        X = self.linear_input(X)
        X = self.gru(X)
        X = X[0][:,-1,:]
        X = self.linear_output(X)
        Y = nn.Softmax(dim=-1)(X)
        return Y

class ExpertMLP_BP_GTNet(nn.Module):
    def __init__(self, rng, num_experts, input_size, hidden_size, output_size, use_cuda=False, **kwargs):
        super(ExpertMLP_BP_GTNet, self).__init__()
        self.rng = rng
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.use_cuda = use_cuda             
        self.act = nn.ELU()                     
        
        self.w_l1, self.b_l1 = self.init_params(self.num_experts, self.input_size, self.hidden_size)
        self.w_l2, self.b_l2 = self.init_params(self.num_experts, self.hidden_size, self.hidden_size)
        self.w_l3, self.b_l3 = self.init_params(self.num_experts, self.hidden_size, self.output_size)
                                
    def init_params(self, num_experts, input_size, output_size):
        w_bound = np.sqrt(6. / np.prod([input_size, output_size]))
        w = np.asarray(self.rng.uniform(low=-w_bound, high=w_bound, size=[num_experts, input_size, output_size]), dtype=np.float32)
        if self.use_cuda:
            w = nn.Parameter(torch.cuda.FloatTensor(w), requires_grad=True)
            b = nn.Parameter(torch.cuda.FloatTensor(num_experts, output_size).fill_(0), requires_grad=True)
        else:
            w = nn.Parameter(torch.FloatTensor(w), requires_grad=True)
            b = nn.Parameter(torch.FloatTensor(num_experts, output_size).fill_(0), requires_grad=True)
        return w, b

    def linearlayer(self, inputs, weights, bias):
        return torch.sum(inputs[..., None] * weights, dim=1) + bias

    def forward(self, X, blending_coef):
        w_l1 = torch.sum(blending_coef[..., None, None] * self.w_l1[None], dim=1)
        b_l1 = torch.matmul(blending_coef, self.b_l1)
        w_l2 = torch.sum(blending_coef[..., None, None] * self.w_l2[None], dim=1)
        b_l2 = torch.matmul(blending_coef, self.b_l2)
        w_l3 = torch.sum(blending_coef[..., None, None] * self.w_l3[None], dim=1)
        b_l3 = torch.matmul(blending_coef, self.b_l3) 
        
        # Layer 1
        X = self.linearlayer(X, w_l1, b_l1)
        X = self.act(X)        
        # Layer 2  
        X = self.linearlayer(X, w_l2, b_l2)
        X = self.act(X)        
        # layer 3
        Y = self.linearlayer(X, w_l3, b_l3)                        
        return Y


# ============================================================================
# Target Estimation Network (TENet)
# ============================================================================
class TENet(nn.Module):
    def __init__(self, 
                 head_depth,
                 head_sz,  
                 feat_sz,
                 depth,
                 condition_dim, 
                 input_dim,
                 window_dim,
                 target_dim,                         
                 n_flow_layers,
                 n_encoder_layers,
                 n_encoder_head,
                 encoder_dropout=0,
                 use_null_token=True):
        super().__init__()
        self.target_dim = target_dim
        self.window_dim = window_dim      
        self.use_null_token = use_null_token
        self.encoder = Encoder_TENet(input_dim=input_dim, 
                                     output_dim=condition_dim, 
                                     window_size=window_dim, 
                                     n_encoder_layers=n_encoder_layers,
                                     n_encoder_head=n_encoder_head,
                                     encoder_dropout=encoder_dropout)
        
        # Flow configuration
        flow_config = {"param_sz": target_dim,
                       "cond_sz": condition_dim,
                       "n_block": n_flow_layers}
        
        # Configure for Glow (yes ActNorm, yes invertible conv, no permutation, no batch norm)
        flow_config["block"] = {"head_depth": head_depth,
                                "head_sz": head_sz,
                                "depth": depth,
                                "feat_sz": feat_sz,
                                "cond_sz": condition_dim}
        self.flow = InvertibleNet(flow_config)                
        self.head_target_null_token = nn.Parameter(torch.randn(1, 3))
        self.righthand_target_null_token = nn.Parameter(torch.randn(1, 3))
        self.lefthand_target_null_token = nn.Parameter(torch.randn(1, 3))
        
    def replace_null_target(self, target):
        head_target = target[:,:3]
        righthand_target = target[:,3:6]
        lefthand_target = target[:,6:9]
        
        is_head_null      = torch.all(head_target==0, dim=1)
        is_righthand_null = torch.all(righthand_target==0, dim=1)
        is_lefthand_null  = torch.all(lefthand_target==0, dim=1)
        
        batch_size, _ = target.shape
        head_null_expanded      = self.head_target_null_token.expand(batch_size, -1)
        righthand_null_expanded = self.righthand_target_null_token.expand(batch_size, -1)
        lefthand_null_expanded  = self.lefthand_target_null_token.expand(batch_size, -1)
        
        new_head_target      = torch.where(is_head_null.unsqueeze(-1),      head_null_expanded,      head_target)
        new_righthand_target = torch.where(is_righthand_null.unsqueeze(-1), righthand_null_expanded, righthand_target)
        new_lefthand_target  = torch.where(is_lefthand_null.unsqueeze(-1),  lefthand_null_expanded,  lefthand_target)
        
        target = torch.cat((new_head_target, new_righthand_target, new_lefthand_target), dim=-1)
         
        return target
        
    def forward(self, x, target):
        # Encode the input data
        cond = self.encoder(x)
        
        # Process null target
        if self.use_null_token:
            target = self.replace_null_target(target)

        # Training mode: transform targets to latent space and compute log-likelihood
        z, log_det_J = self.flow(target, cond)
        
        # Compute log-likelihood using change of variables formula
        log_prob = torch.sum(-0.5 * z.pow(2) - 0.5 * np.log(2 * np.pi), dim=-1)
        log_prob = log_prob + log_det_J
        
        return z, log_det_J, log_prob
    
    def posterior_probability(self, x, target):        
        # Encode the input data
        cond = self.encoder(x)

        # Process null target
        if self.use_null_token:
            target = self.replace_null_target(target)
        
        # Transform to latent space
        z, log_det_J = self.flow(target, cond)
        
        # Compute log-likelihood
        log_prob = torch.sum(-0.5 * z.pow(2) - 0.5 * np.log(2 * np.pi), dim=-1)
        log_prob = log_prob + log_det_J
        
        return log_prob
    
    def infer_target(self, x, candidate_targets):
        # Get log probabilities for all candidates
        log_probs = self.posterior_probability(x, candidate_targets)
        
        # Find the most likely target
        max_indices = torch.argmax(log_probs, dim=-1)
        
        # Compute confidence (softmax of log probs)
        probs = F.softmax(log_probs, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]
        
        return max_indices, confidences

class InvertibleNet(nn.Module):
    """
    An invertible network for normalizing flows.
    This class supports different types of flows:
    - Glow: blocks include (act_norm, inv_1x1_conv, affine_block)
    - Real NVP: blocks include (affine_block_with_permutation, batch_norm)
    """
    def __init__(self, config):
        super().__init__()
        n_block = config["n_block"]
        self.n_param = config["param_sz"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        modules = list()
        for i in range(n_block):
            modules += [ActNormLayer(self.n_param)]
            modules += [Inv1x1ConvLayer(self.n_param, LU_decomposed=True)]
            modules += [ConditionalAffineBlock(self.n_param, config["block"])]
        self.cINNs = nn.ModuleList(modules)

    def forward(self, param, cond, inverse=False):
        """
        Apply the invertible network (forward or inverse) on the input tensor.
        """
        log_det_Js = []
        if not inverse:
            z = param
            for cINN in self.cINNs:
                z, log_det_J = cINN(z, cond)
                log_det_Js.append(log_det_J)
            log_det_J = sum(log_det_Js)
            return z, log_det_J
        else:
            theta = param
            for cINN in reversed(self.cINNs):
                theta, log_det_J = cINN(theta, cond, inverse=True)
                log_det_Js.append(log_det_J)
            log_det_J = sum(log_det_Js)
            return theta, log_det_J

class ActNormLayer(nn.Module):
    """
    Activation Normalization layer for Glow, performing feature-wise scaling and biasing.
    """
    def __init__(self, n_param):
        super().__init__()
        self.n_param = n_param
        self.mu = nn.Parameter(torch.zeros(n_param, dtype=torch.float))
        self.log_sigma = nn.Parameter(torch.zeros(n_param, dtype=torch.float))
        self.register_buffer("inited", torch.tensor(0, dtype=torch.bool))

    def initialize_parameters(self, input):
        """
        Initialize the parameters (mu and log_sigma) based on the input tensor.
        """
        if not self.training:
            return

        with torch.no_grad():
            # Calculate mu and log_sigma from input data
            mu = -torch.mean(input.clone(), dim=0)
            vars = torch.mean((input.clone() + mu) ** 2, dim=0)
            log_sigma = torch.log(1.0 / (torch.sqrt(vars) + 1e-6))
            # Update parameters
            self.mu.data.copy_(mu.data)
            self.log_sigma.data.copy_(log_sigma.data)

        self.inited.fill_(1)

    def forward(self, param, cond=None, inverse=False):
        """
        Apply activation normalization (forward or inverse) on the input tensor.
        """
        if not self.inited and self.training:
            self.initialize_parameters(param)

        if not inverse:
            z = (param + self.mu) * torch.exp(self.log_sigma)
            log_det = torch.sum(self.log_sigma)
            return z, log_det
        else:
            x = param * torch.exp(-self.log_sigma) - self.mu
            log_det = -torch.sum(self.log_sigma)
            return x, log_det

class Inv1x1ConvLayer(nn.Module):
    """
    1x1 Convolution for Glow
    Invertibility guaranteed by LU-decomposition
    """
    def __init__(self, n_param, LU_decomposed=True):
        super().__init__()
        self.n_param = n_param
        W = scipy.linalg.qr(np.random.randn(n_param, n_param))[0]
        self.LU_decomposed = LU_decomposed

        if not self.LU_decomposed:
            self.W = nn.Parameter(torch.tensor(W, dtype=torch.float))
        else:
            P, L, U = scipy.linalg.lu(W)
            self.register_buffer("P", torch.tensor(P, dtype=torch.float))
            self.L = nn.Parameter(torch.tensor(L, dtype=torch.float))
            self.S = nn.Parameter(torch.tensor(np.diag(U), dtype=torch.float))
            self.U = nn.Parameter(torch.triu(torch.tensor(U, dtype=torch.float), diagonal=1))
            self.register_buffer("eye", torch.eye(self.n_param))

    def forward(self, param, cond=None, inverse=False):
        """
        Apply the 1x1 convolution (forward or inverse) on the input tensor.
        """
        if not inverse:
            if not self.LU_decomposed:
                z = param @ self.W
                log_det = torch.slogdet(self.W)[1]
            else:
                L = torch.tril(self.L, diagonal=-1) + self.eye
                U = torch.triu(self.U, diagonal=1)
                z = param @ self.P @ L @ (U + torch.diag(self.S))
                log_det = torch.sum(torch.log(torch.abs(self.S)))
            return z, log_det
        else:
            if not self.LU_decomposed:
                x = param @ torch.inverse(self.W)
                log_det = -torch.slogdet(self.W)[1]
            else:
                L = torch.tril(self.L, diagonal=-1) + self.eye
                U = torch.triu(self.U, diagonal=1)
                W = self.P @ L @ (U + torch.diag(self.S))
                x = param @ torch.inverse(W)
                log_det = -torch.sum(torch.log(torch.abs(self.S)))
            return x, log_det

class ConditionalLayer(nn.Module):
    """
    A conditional layer that combines input tensor and conditioning tensor.
    """
    def __init__(self, in_sz, out_sz, config):
        super().__init__()
        # Process conditioning information
        head_seq = []
        for i in range(config["head_depth"]):
            layer_in = config["cond_sz"]
            layer_out = config["head_sz"] if i == config["head_depth"] - 1 else config["cond_sz"]                                                               
            head_seq.append(nn.Linear(layer_in, layer_out))                        
            if i < config["head_depth"] - 1:                
                head_seq.append(nn.Tanh())            
                
        self.head_layers = nn.Sequential(*head_seq)

        # Main processing layers
        seq = []
        first_in_sz = in_sz
        first_in_sz += config["head_sz"] if config["head_depth"] > 0 else config["cond_sz"]
        for i in range(config["depth"]):
            layer_in = first_in_sz if i == 0 else config["feat_sz"]
            layer_out = out_sz if i == config["depth"] - 1 else config["feat_sz"]            
            seq.append(nn.Linear(layer_in, layer_out))
            if i == config["depth"] - 1:
                seq.append(nn.Tanh())

        self.layers = nn.Sequential(*seq)

    def forward(self, param, cond):
        """
        Apply the conditional layer on the input tensor.
        """
        cond = self.head_layers(cond)
        x = torch.cat([param, cond], dim=-1)
        return self.layers(x)

class ConditionalAffineBlock(nn.Module):
    """
    Glow-style conditional affine coupling layer.
    Only transforms one half of the input (single coupling).
    """
    def __init__(self, n_param, config):
        super().__init__()
        self.sz_unchanged = n_param // 2  # First half: unchanged
        self.sz_transform = n_param - self.sz_unchanged  # Second half: transformed
        
        # Single coupling: use first half to transform second half
        self.scale_net = ConditionalLayer(self.sz_unchanged, self.sz_transform, config)
        self.translation_net = ConditionalLayer(self.sz_unchanged, self.sz_transform, config)

    def forward(self, param, cond, inverse=False):
        """
        Apply Glow-style single coupling (forward or inverse).
        
        Args:
            param: Input tensor [..., n_param]
            cond: Conditioning tensor
            inverse: If True, apply inverse transformation
        
        Returns:
            output: Transformed tensor
            log_det: Log determinant of Jacobian
        """
        if not inverse:
            return self._forward(param, cond)
        else:
            return self._inverse(param, cond)
    
    def _forward(self, param, cond):
        """
        Forward: x -> z
        x_a stays unchanged, x_b is transformed using x_a as condition
        """
        # Split into two parts
        x_a = param[..., :self.sz_unchanged]  # Unchanged part
        x_b = param[..., self.sz_unchanged:]  # Part to transform
        
        # Compute scale and translation from unchanged part
        log_s = self.scale_net(x_a, cond)
        t = self.translation_net(x_a, cond)
        
        # Affine transformation on second part only
        z_b = x_b * torch.exp(log_s) + t
        
        # Concatenate (first part unchanged)
        z = torch.cat([x_a, z_b], dim=-1)
        
        # Log determinant (only from transformed part)
        log_det = log_s.sum(dim=-1)
        
        return z, log_det
    
    def _inverse(self, param, cond):
        """
        Inverse: z -> x
        z_a stays unchanged, z_b is inverse-transformed using z_a as condition
        """
        # Split into two parts
        z_a = param[..., :self.sz_unchanged]  # Unchanged part
        z_b = param[..., self.sz_unchanged:]  # Part to inverse-transform
        
        # Compute scale and translation from unchanged part
        log_s = self.scale_net(z_a, cond)
        t = self.translation_net(z_a, cond)
        
        # Inverse affine transformation on second part only
        x_b = (z_b - t) * torch.exp(-log_s)
        
        # Concatenate (first part unchanged)
        x = torch.cat([z_a, x_b], dim=-1)
        
        # Log determinant (negative for inverse)
        log_det = -log_s.sum(dim=-1)
        
        return x, log_det

class Encoder_TENet(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, n_encoder_layers, n_encoder_head, encoder_dropout):
        super().__init__()
        self.window_size = window_size        
        self.embedding = nn.Linear(4, output_dim)
        self.time_token = nn.Parameter(torch.randn(self.window_size, output_dim))
        self.part_token = nn.Parameter(torch.randn(3, output_dim))
        self.separator_token = torch.zeros_like(torch.randn(2, output_dim))  # learnable register tokens at the 2 part boundaries
        
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(output_dim,
                                                                            nhead=n_encoder_head,
                                                                            dim_feedforward=output_dim*4,
                                                                            dropout=encoder_dropout,
                                                                            batch_first=True,
                                                                            norm_first=False), n_encoder_layers)
        self.linear = nn.Linear(output_dim*3, output_dim)
        
        self.register_buffer('head_indices', torch.tensor([0, 1, 2, 9]))
        self.register_buffer('righthand_indices', torch.tensor([3, 4, 5, 10]))      
        self.register_buffer('lefthand_indices', torch.tensor([6, 7, 8, 11]))     
                
    def forward(self, x):
        x_head = x[:, :, self.head_indices]
        x_righthand = x[:, :, self.righthand_indices]
        x_lefthand = x[:, :, self.lefthand_indices]

        x_head = self.embedding(x_head) + self.part_token[0] + self.time_token
        x_righthand = self.embedding(x_righthand) + self.part_token[1] + self.time_token
        x_lefthand = self.embedding(x_lefthand) + self.part_token[2] + self.time_token
                
        B = x_head.shape[0]
        separator_token = self.separator_token.to(x_head.device)
        sep0 = separator_token[0].view(1, 1, -1).expand(B, 1, -1)
        sep1 = separator_token[1].view(1, 1, -1).expand(B, 1, -1)

        x = torch.cat((x_head, sep0, x_righthand, sep1, x_lefthand), 1)
        x = self.transformer(x)
        x_head = x[:, self.window_size-1]
        x_righthand = x[:, 2*self.window_size]
        x_lefthand = x[:, 3*self.window_size+1]

        x = torch.cat((x_head, x_righthand, x_lefthand), -1)
        x = self.linear(x)
        
        return x

def Get_Model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():        
        buffer_size += buffer.nelement() * buffer.element_size()    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.5f}MB'.format(size_all_mb))    
    print(param_size)

def Get_Model_size_excluding(model, exclude_names):
    """
    Measure the model size excluding specific components
    
    Args:
        model: PyTorch model
        exclude_names: list of component names to exclude
    """
    param_size = 0
    excluded_param_size = 0
    
    for name, param in model.named_parameters():
        size = param.nelement() * param.element_size()
        # Check whether it is a component to exclude
        if any(exclude_name in name for exclude_name in exclude_names):
            excluded_param_size += size
        else:
            param_size += size
    
    buffer_size = 0
    excluded_buffer_size = 0
    
    for name, buffer in model.named_buffers():
        size = buffer.nelement() * buffer.element_size()
        if any(exclude_name in name for exclude_name in exclude_names):
            excluded_buffer_size += size
        else:
            buffer_size += size
    
    total_mb = (param_size + buffer_size) / 1024**2
    excluded_mb = (excluded_param_size + excluded_buffer_size) / 1024**2
    
    print(f'Included model size: {total_mb:.5f}MB')
    print(f'Excluded component size: {excluded_mb:.5f}MB')
    print(f'Number of included parameters: {param_size // 4}')  # based on float32
    
    return total_mb

if __name__ == '__main__':
    
    print("torch.cuda.is_available()="+str(torch.backends.cudnn.is_available()))
    print("torch.backends.cudnn.is_available()="+str(torch.backends.cudnn.is_available())) 
    
    # BP_GTNet   
    model = BP_GTNet(rng=np.random.RandomState(23456), 
                  num_experts=24, 
                  num_experts_part=15,
                  window_size=30,
                  user_pose_window_size=5,
                  layer_size_encoder=4,
                  input_size_encoder=27,
                  head_codebook_channels=6, 
                  head_codebook_dimension=6,
                  head_temperature=5,
                  hand_hidden_size=12,
                  hand_codebook_size=128,
                  hand_codebook_layer_size=5,                          
                  model_gating='Transformer',
                  layer_size_gating=2,
                  input_size_gating=15,
                  hidden_size_gating=64,
                  input_size_moe=36,
                  use_cuda=False)      
    model.eval()
    output = model(torch.ones(32,30,3), torch.ones(32,30,15), torch.ones(32,5,27), torch.ones(32,36), torch.ones(32,30,3), False)
    print("BP_GTNet output.shape="+str(output.shape))
    output, _,  _, _, _ = model(torch.ones(32,30,3), torch.ones(32,30,15), torch.ones(32,5,27), torch.ones(32,36), torch.ones(32,30,3), True)
    print("BP_GTNet output.shape="+str(output.shape))    
    exclude_list = [
        'moe_network_head',
        'moe_network_righthand', 
        'moe_network_lefthand',
        'gating_network_head',
        'gating_network_righthand',
        'gating_network_lefthand'
    ]    
    print("=== Total model size ===")
    Get_Model_size(model)    
    print("\n=== Size after excluding specific components ===")
    Get_Model_size_excluding(model, exclude_list)
    print("") 
    
    # TENet     
    model = TENet(head_depth=3,
                  head_sz=128,                 
                  feat_sz=128,
                  depth=3,
                  condition_dim=256, 
                  input_dim=12,
                  window_dim=45,
                  target_dim=9,                  
                  n_flow_layers=5,
                  n_encoder_layers=6,
                  n_encoder_head=8,
                  encoder_dropout=0,
                  use_null_token=False)
    model.eval()
    z, log_det_J, log_prob = model.forward(torch.ones(32,45,12), torch.ones(32,9))
    print("TENet output z.shape="+str(z.shape))
    print("TENet output log_det_J.shape="+str(log_det_J.shape))
    print("TENet output log_prob.shape="+str(log_prob.shape))
    # print("TENet output="+str(output))
    print("")    
    max_indices, confidences = model.infer_target(torch.rand(32,45,12), torch.ones(32,9))
    print("TENet output max_indices="+str(max_indices))
    print("TENet output confidences="+str(confidences))    
    probability = model.posterior_probability(torch.rand(2,45,12), torch.ones(2,9))
    print("TENet output probability="+str(probability[0]))
    print("=== Total model size ===")
    Get_Model_size(model)        