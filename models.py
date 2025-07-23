import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, TopKPooling, global_max_pool
import torchvision.models as models
import torch_geometric.nn as pyg_nn
import torchvision.models as models
import torch.nn as nn
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision import models, transforms
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from .utils import *
from .subject import *

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, TopKPooling, global_max_pool
import torchvision.models as models
import torch_geometric.nn as pyg_nn
import torchvision.models as models
import torch.nn as nn
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision import models, transforms
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

class TrajectoryGAT(torch.nn.Module):
    def __init__(self, input_dim = 9, hidden_dim = 128, output_dim = 2):
        super(TrajectoryGAT, self).__init__()
        self.spatial_gat1  = GATConv(input_dim, hidden_dim)
        self.temporal_gat1 = GATConv(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.classifier  = nn.Linear(128, 2)
        self.fc          = nn.Linear(hidden_dim , 2)
        self.fc_adjust   = nn.Linear(256, 2)
        self.time_embedding = nn.Embedding(3000, 128)
    def _initialize_weights(self):
        for layer in [self.spatial_gat1, self.temporal_gat1]:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        for layer in [self.classifier, self.fc, self.fc_adjust]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        for layer in [self.layer_norm1, self.layer_norm2]:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
    def forward(self, data):
        x, edge_index_spatial, edge_weight_spatial, edge_index_temporal, edge_weight_temporal = data.x, data.edge_index_spatial, data.spatial_weights,data.edge_index_temporal, data.temporal_weights
        batch = data.batch
        x = self.spatial_gat1(x, edge_index_spatial, edge_weight_spatial)
        x = torch.relu(x)
        x = self.temporal_gat1(x, edge_index_temporal, edge_weight_temporal)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x = F.softmax(x, dim = -1)
        return x
class TrajectoryGAT(torch.nn.Module):
    def __init__(self, input_dim = 9, hidden_dim = 128, output_dim = 2):
        super(TrajectoryGAT, self).__init__()
        self.spatial_gat1  = GATConv(input_dim, hidden_dim)
        self.temporal_gat1 = GATConv(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.classifier  = nn.Linear(128, 2)
        self.fc          = nn.Linear(hidden_dim , 2)
        self.fc_adjust   = nn.Linear(256, 2)
        # self._initialize_weights()
        self.time_embedding = nn.Embedding(3000, 128)
    def _initialize_weights(self):

        for layer in [self.spatial_gat1, self.temporal_gat1]:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        for layer in [self.classifier, self.fc, self.fc_adjust]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        for layer in [self.layer_norm1, self.layer_norm2]:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
    def forward(self, data):
        x, edge_index_spatial, edge_weight_spatial, edge_index_temporal, edge_weight_temporal = data.x, data.edge_index_spatial, data.spatial_weights,data.edge_index_temporal, data.temporal_weights
        batch = data.batch
        
        x = self.spatial_gat1(x, edge_index_spatial, edge_weight_spatial)
        x = torch.relu(x)
        # y = x
        x = self.temporal_gat1(x, edge_index_temporal, edge_weight_temporal)
        # x = torch.relu(x)
        
        # x = self.temporal_gat1(x, edge_index_spatial, edge_weight_spatial)
        # x = x + y 
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x = F.log_softmax(x, dim = -1)
        return x
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from einops import rearrange

# Placeholder embedding classes
class PositionEmbeddingSine1d(nn.Module):
    def __init__(self, max_len, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(max_len, hidden_dim)
    def forward(self, x):
        B, T, C = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.embed(pos)

class PositionEmbeddingSine1dShort(nn.Module):
    def __init__(self, num_positions, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(num_positions, hidden_dim)
    def forward(self, x):
        B, N, C = x.size()
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        return x + self.embed(pos)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from einops import rearrange

class Generactor(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.feat_proj = nn.Linear(1, hidden_dim)
        self.cue_prob_proj = nn.Linear(3, hidden_dim)  # cue(1) + prob(2)

        self.token_pos_embedding = PositionEmbeddingSine1d(max_len=100, hidden_dim=hidden_dim)
        self.image_pos_embedding = PositionEmbeddingSine1dShort(num_positions=4, hidden_dim=hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=512, batch_first=True)
        self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.image_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.actor_head = nn.Linear(hidden_dim, 1)
        self.patch_head = nn.Linear(hidden_dim, 1)
        self.fix_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )
        self.sac_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )
        self.value_head_img = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.value_head_patch = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.value_head_sac = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.value_head_fix = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def extract_img_feats_crop10x10(self, high_feats, screen_mask_index = screen_mask_index.view(27, 48)):
        B, C, H, W = high_feats.shape
        crops = []
        for i in range(4):
            mask = (screen_mask_index == i)
            coords = mask.nonzero(as_tuple=False)  
            # print(coords)
            y1, x1 = coords.min(dim=0).values.tolist()
            y2, x2 = coords.max(dim=0).values.tolist()
            # print(y1,x1, y2,x2)
            crop = high_feats[:, :, y1:y2+1, x1:x2+1] 
            crops.append(crop)
        out = torch.stack(crops, dim=1)
        return out

    def forward(self, 
                states, 
                cues, 
                probs,
                screen_mask_index = screen_mask_index.view(27,48),
                time_record = None,
                ior_map = None):
        states = [states[:,:-3,:, :], states[:, -2:, :,:]][1]
        states = states.sum(1).unsqueeze(1)
        B, _, H, W = states.shape
        img_feats = self.extract_img_feats_crop10x10(states)  # [B, 4, 1, 10, 10]
        img_feats = rearrange(img_feats, 'b n c h w -> b n (h w) c')  # [B, 4, 100, 1]
        img_feats = self.feat_proj(img_feats)  # [B, 4, 100, hidden_dim]

        cues = cues.float().unsqueeze(-1)  # [B, 4, 1]
        probs = probs.unsqueeze(1).expand(B, 4, 2)  # [B, 4, 2]
        cue_prob_feat = torch.cat([cues, probs], dim=-1)  # [B, 4, 3]
        cue_cond_embed = self.cue_prob_proj(cue_prob_feat).unsqueeze(2)  # [B, 4, 1, hidden_dim]

        token_input = img_feats + cue_cond_embed  # [B, 4, 100, hidden_dim]
        token_input = token_input.view(B * 4, 100, -1)
        token_input = self.token_pos_embedding(token_input)
        token_output = self.token_encoder(token_input).view(B, 4, 100, -1)

        img_tokens = token_output.mean(dim=2)  # [B, 4, hidden_dim]
        img_tokens = self.image_pos_embedding(img_tokens)
        img_tokens = self.image_encoder(img_tokens)

        act_img_logit = self.actor_head(img_tokens).squeeze(-1)  # [B, 4]
        act_img_prob = F.softmax(act_img_logit, dim=-1)
        top_img_idx = act_img_prob.argmax(dim=-1)
        
        '''img '''
        eps = 1e-6
        if time_record is not None:
            for b in range(B):
                # print(time_record[b][screen_mask_index!=-1].sum().item())
                if time_record[b][screen_mask_index!=-1].sum() == 0:
                    continue
                ior_time = calculate_ior_temporal(time_record[b])
                ior_4_img = []
                for j in range(4):
                    ior_time_temp = ((screen_mask_index == j) * ior_map)
                    ior_4_img.append(ior_time_temp.unsqueeze(0)) 
                ior_4_img = torch.cat(ior_4_img)
                # ior_time_temp = ((screen_mask_index == top_img_idx[b]) * ior_time)
                top_idx = top_img_idx[b].item()
                prob = act_img_prob[b].clone()
                ior_vals = (1-ior_4_img.view(4, -1).max(dim=1).values.clamp(0,1))
                # print(ior_vals)
                # if ior_vals[top_idx].max() > 0:
                #     act_img_prob[b] = F.one_hot(torch.tensor(top_idx), num_classes=4).float().to(act_img_prob.device)
                # elif ior_vals[top_idx].max() == 0 and ior_vals[top_idx].min() != 0:
                act_img_prob[b] = act_img_prob[b] * ior_vals.to(act_img_prob.device)
                    
            row_sum = act_img_prob.sum(dim=1, keepdim=True)
            act_img_prob = act_img_prob / (row_sum + eps)
            top_img_idx = act_img_prob.argmax(dim=-1)
        # print(top_img_idx)
        patch_logits_all = self.patch_head(token_output).squeeze(-1)   # [B, 4, 100]
        patch_prob_all = F.softmax(patch_logits_all, dim=-1)
        
        patch_prob = patch_prob_all[torch.arange(B), top_img_idx]
        selected_token = token_output[torch.arange(B), top_img_idx]  # [B, 100, hidden_dim]
        # patch_logit = self.patch_head(selected_token).squeeze(-1)  # [B, 100]
        
        '''patch ior'''
        # patch_prob = F.softmax(patch_logit, dim=-1)
        if time_record is not None:
            for b in range(B):
                if time_record[b][screen_mask_index!=-1].sum() == 0:
                    continue
                else:
                    # print(top_img_idx[b].shape, top_img_idx[b])
                    # print((screen_mask_index == top_img_idx[b]).shape)
                    ior = ior_map[(screen_mask_index == top_img_idx[b])]
                    patch_prob[b] = ior_map[b] * patch_prob[b]
        row_sum = patch_prob.sum(dim=1, keepdim=True)
        patch_prob = patch_prob / (row_sum + 1e-8)
        top_patch_idx = patch_prob.argmax(dim=-1)
        
        '''fix & sac'''
        token_repr = selected_token.mean(dim=1)  # [B, hidden_dim]
        fix_params = self.fix_head(token_repr)  # [B, 2]
        sac_params = self.sac_head(token_repr)  # [B, 2]

        fix_mu, fix_std = fix_params[:, 0], fix_params[:, 1].exp()
        sac_mu, sac_std = sac_params[:, 0], sac_params[:, 1].exp()

        fix_dist = Normal(fix_mu, fix_std)
        sac_dist = Normal(sac_mu, sac_std)
        '''Critic Net'''
        value_img = self.value_head_img(img_tokens.mean(dim=1)).squeeze(-1)     # [B]
        value_patch = self.value_head_patch(selected_token.mean(dim=1)).squeeze(-1)  # [B]
        selected_feat = selected_token[torch.arange(B), top_patch_idx]
        value_sac = self.value_head_sac(selected_feat).squeeze(-1)             # [B]
        value_fix = self.value_head_fix(selected_feat).squeeze(-1)             # [B]

        return [act_img_prob, patch_prob, sac_dist, fix_dist], [value_img+value_patch, value_sac, value_fix]

    def select_action(self, 
                      states, 
                      cues,
                      labels,
                      fix_range=[50, 600], 
                      sac_range=[20, 300], 
                      image_index = [(10,16),(28,16),(28,1),(10,1)],
                      time_record = None,
                      ior_map = None,
                      is_eval = False):
        actions_dist, values = self.forward(states, cues, labels, time_record = time_record, ior_map = ior_map)
        # act_img_prob, patch_prob, fix_dist, sac_dist = self.forward(high_feats, screen_mask_index)
        
        B = states.shape[0]

        x_sac = actions_dist[2].rsample()
        x_fix = actions_dist[3].rsample()

        y_sac = torch.tanh(x_sac)
        y_fix = torch.tanh(x_fix)

        fix = ((y_fix + 1) / 2) * (fix_range[1] - fix_range[0]) + fix_range[0]
        sac = ((y_sac + 1) / 2) * (sac_range[1] - sac_range[0]) + sac_range[0]
        if is_eval:
            img_idx = act_img_prob.argmax(dim=-1)
            patch_idx = patch_prob.argmax(dim=-1)
        else:
            img_sampler = Categorical(actions_dist[0])
            img_idx = img_sampler.sample()
            patch_sampler = Categorical(actions_dist[1])
            patch_idx = patch_sampler.sample()
        batch_actions = []
        for b in range(B):
            img_id = img_idx[b].item()
            patch_id = patch_idx[b].item()
            base_x, base_y = image_index[img_id]
            patch_x = base_x + (patch_id % 10)
            patch_y = base_y + (patch_id // 10)
            action_id = int(48 * patch_y + patch_x)
            batch_actions.append([action_id, fix[b].item(), sac[b].item()])
        actions = torch.tensor(batch_actions, device=states.device).double()
        # print(actions.shape)
        # log_probs = torch.stack([
        #     img_sampler.log_prob(img_idx) if not is_eval else torch.log(act_img_prob.gather(1, img_idx.unsqueeze(1)).squeeze(1)),
        #     patch_sampler.log_prob(patch_idx) if not is_eval else torch.log(patch_prob.gather(1, patch_idx.unsqueeze(1)).squeeze(1)),
        #     fix_dist.log_prob(x_fix),
        #     sac_dist.log_prob(x_sac)
        # ], dim=1)
        log_probs = torch.stack([
            img_sampler.log_prob(img_idx) + patch_sampler.log_prob(patch_idx) if not is_eval else torch.log(act_img_prob.gather(1, img_idx.unsqueeze(1)).squeeze(1)) + torch.log(patch_prob.gather(1, patch_idx.unsqueeze(1)).squeeze(1)),
            actions_dist[2].log_prob(x_fix),
            actions_dist[3].log_prob(x_sac)
        ], dim=1)
        return actions, log_probs, torch.stack(values, dim=1)
    
    def get_log_prob(self, 
                     states, 
                     cues,
                     labels,
                     actions, 
                     img_index = [(10,16),(28,16),(28,1),(10,1)],
                     fixation_range = [50.0, 600.0],
                     saccade_range = [20.0, 300.0],
                     screen_mask_index = screen_mask_index.view(27,48),
                     time_record = None,
                     ior_map = None):
        # act_img_prob, patch_prob, fix_dist, sac_dist = self.forward(high_feats, screen_mask_index)
        actions_dist, values = self.forward(states, cues, labels, time_record = None, ior_map = None)      
        act_img_prob, patch_prob, fix_dist, sac_dist = actions_dist
        screen_mask_index = screen_mask_index.to(actions.device)
        patch_y = (actions[:, 0] // 48).long()
        patch_x = (actions[:, 0] % 48).long()
        B = actions[:0].shape[0]
        img_id = screen_mask_index[patch_y, patch_x]
        # img_id = screen_mask_index[actions[:, 0].long()]
        img_index_tensor = torch.tensor(img_index, device=actions.device)  # [4, 2]
        img_index_selected = img_index_tensor[img_id]  # [B, 2]
        
        rel_x = patch_x - img_index_selected[:, 0]
        rel_y = patch_y - img_index_selected[:, 1]
        patch_id = rel_y * 10 + rel_x
        # print(actions[:, 0].cpu())
        # print(patch_y, patch_x)
        # print(actions[:, 1].cpu())
        # print(f'img_id:{img_id}')
        # print(f'patch_id:{patch_id}')
        log_img = torch.log(act_img_prob.gather(1, img_id.unsqueeze(1)).clamp(min=1e-6)).squeeze(1)
        log_patch = torch.log(patch_prob.gather(1, patch_id.unsqueeze(1)).clamp(min=1e-6)).squeeze(1)
        log_pos = log_img + log_patch
        
        mean_sac = actions_dist[2].mean
        mean_fix = actions_dist[3].mean
        mean_sac_log_std = actions_dist[2].stddev
        mean_fix_log_std = actions_dist[3].stddev
        # print(actions[:,1])
        # print(actions[:,2])
        # log_probs_pos = torch.log(torch.clamp(actions_dist[0].gather(1, actions[:, 0].long().unsqueeze(1)), min=1e-6)).squeeze(1)
        # clamp saccade and fixation durations to valid ranges
        clamped_sac = torch.clamp(actions[:, 1], min=saccade_range[0] + 1e-3, max=saccade_range[1] - 1e-3)
        clamped_fix = torch.clamp(actions[:, 2], min=fixation_range[0] + 1e-3, max=fixation_range[1] - 1e-3)

        # compute log probs using clamped values
        log_sac = self.gauss_log_prob((mean_sac, mean_sac_log_std), clamped_sac, saccade_range)
        log_fix = self.gauss_log_prob((mean_fix, mean_fix_log_std), clamped_fix, fixation_range)

        # log_sac = self.gauss_log_prob((mean_sac, mean_sac_log_std), torch.clamp(actions[:, 1], saccade_range)
        # log_fix = self.gauss_log_prob((mean_fix, mean_fix_log_std), actions[:, 2], fixation_range)

        # log_sac = self.gauss_log_prob(sac_dist, clamped_sac, saccade_range)
        # log_fix = self.gauss_log_prob(fix_dist, clamped_fix, fixation_range)
        
        # --- Entropies ---
        entropy_img = -torch.sum(act_img_prob * torch.log(act_img_prob.clamp(min=1e-6)), dim=1)
        entropy_patch = -torch.sum(patch_prob * torch.log(patch_prob.clamp(min=1e-6)), dim=1)
        # if is_eval:
        #     img_idx = act_img_prob.argmax(dim=-1)
        #     patch_idx = patch_prob.argmax(dim=-1)
        # else:
        #     img_sampler = Categorical(actions_dist[0])
        #     img_idx = img_sampler.sample()
        #     patch_sampler = Categorical(actions_dist[1])
        #     patch_idx = patch_sampler.sample()
        # probs_pos = actions_dist[0] * actions_dist[1]
        entropy_pos = entropy_img + entropy_patch
        # entropy_pos = -torch.sum(probs_pos * torch.log(probs_pos.clamp(min=1e-6)), dim=1)
        entropy_sac = actions_dist[2].entropy()
        entropy_fix = actions_dist[3].entropy()
        entropy = torch.stack([entropy_pos, entropy_sac, entropy_fix], dim=1).sum(dim=1, keepdim=True)
        return torch.stack([log_pos, log_sac, log_fix], dim=1), torch.stack(values, dim=1), entropy
        # return log_probs, torch.stack(values, dim=1), entropy
    def gauss_log_prob(self, 
                      params, 
                      x,
                      x_range):
        

        # if x.numel() == 0:   
        #     print(x)# 🚩 空 batch
        #     raise ValueError("Empty batch passed to gauss_log_prob")
        min_x, max_x = x_range
        x = torch.atanh(torch.clamp(2 * (x - min_x) / (max_x - min_x) - 1, min=-0.9999, max=0.9999)).unsqueeze(-1)
        mean, log_diag_std = params
        N, d = mean.unsqueeze(-1).shape
        cov =  torch.square(torch.exp(log_diag_std))
        diff = x - mean.unsqueeze(-1)
        exp_term = - 0.5 * torch.sum(torch.square(diff)/cov, axis=1)
        norm_term = -0.5 * d * torch.log(torch.clamp(2 * torch.pi * torch.ones(N).to(x.device), min=1e-6))
        var_term = - 0.5 * torch.sum(torch.log(cov.unsqueeze(-1)), axis=1)
        log_probs = norm_term + var_term + exp_term
        # print(f'params:{params}')
        # print(f'x:{x}')
        # print(f'log_probs:{log_probs}')
        return log_probs 
    # def gauss_log_prob(self, dist, value, value_range):
    #     min_x, max_x = value_range
    #     # y = torch.atanh(torch.clamp(2 * (value - min_x) / (max_x - min_x) - 1, min=-0.9999, max=0.9999))
    #     # print(f'dist:{dist}')
    #     # print(f'value:{value}')
    #     y = torch.atanh(torch.clamp(2 * (value - min_x) / (max_x - min_x) - 1, min=-0.9999, max=0.9999))
    #     # print(f'y:{y}')
    #     # if torch.isnan(y).any() or torch.isinf(y).any():
    #     #     print("Invalid y detected!", y)
    #     log_prob = dist.log_prob(y) - torch.log((1 - y.pow(2)) * (max_x - min_x) / 2 + 1e-6)
    #     return log_prob

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Discriminator(nn.Module):
    def __init__(self, hidden_dim=256, nhead=8, num_layers=2, gamma=0.99):
        super().__init__()
        self.gamma = gamma

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.feat_proj = nn.Linear(1, hidden_dim)
        self.action_proj = nn.Linear(3, hidden_dim)     # [img_patch_idx, sac_time, fix_time]
        self.label_proj = nn.Linear(2, hidden_dim)      # classifier probs
        self.cue_proj   = nn.Linear(4, hidden_dim)      # cue + one-hot img index (optional)

        self.mlp_reward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.mlp_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def extract_img_feats_crop10x10(self, high_feats, screen_mask_index = screen_mask_index.view(27,48)):
        B, C, H, W = high_feats.shape
        crops = []
        for i in range(4):
            mask = (screen_mask_index == i)
            coords = mask.nonzero(as_tuple=False)
            y1, x1 = coords.min(dim=0).values.tolist()
            y2, x2 = coords.max(dim=0).values.tolist()
            crop = high_feats[:, :, y1:y2+1, x1:x2+1]
            crops.append(crop)
        out = torch.stack(crops, dim=1)
        return out

    def encode(self, states, actions, labels, cues, screen_mask_index = screen_mask_index.view(27,48)):
        states = [states[:,:-3,:, :], states[:, -2:, :,:]][1]  # use ior
        states = states.sum(1).unsqueeze(1)
        B, _, H, W = states.shape

        img_feats = self.extract_img_feats_crop10x10(states, screen_mask_index)
        img_feats = rearrange(img_feats, 'b n c h w -> b n (h w) c')
        img_feats = self.feat_proj(img_feats)  # [B, 4, 100, hidden_dim]
        x = rearrange(img_feats, 'b n p c -> b (n p) c')

        a_embed = self.action_proj(actions).unsqueeze(1)
        l_embed = self.label_proj(labels).unsqueeze(1)
        c_embed = self.cue_proj(cues).unsqueeze(1)

        x = torch.cat([a_embed, l_embed, c_embed, x], dim=1)
        x = self.token_encoder(x)
        return x[:, 0]  # CLS token表示

    def forward(self, states, cues, actions, labels, next_states, log_pis):
        z_s = self.encode(states, actions, labels, cues)
        z_ns = self.encode(next_states, actions, labels, cues)

        r = self.mlp_reward(z_s).squeeze(-1)
        v = self.mlp_value(z_s).squeeze(-1)
        v_next = self.mlp_value(z_ns).squeeze(-1)
        
        advantage = r + self.gamma * v_next - v
        # print(advantage.shape, log_pis.shape)
        logits = advantage - log_pis
        return logits

    def calculate_reward(self, states, cues, actions, labels, next_states, log_pis):
        with torch.no_grad():
            logits = self.forward(states, cues, actions, labels, next_states, log_pis)
            return -F.logsigmoid(-logits)  # AIRL reward