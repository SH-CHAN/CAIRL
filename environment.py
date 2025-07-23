from .subject import *
from .utils import *

# from subject import *
# from utils import *
import torch.nn.functional as F
class Addiction_Env:
    def __init__(self,
                 mask_size=1,
                 max_time=3000,
                 grid=[27, 48],
                 visual_degree=40,
                 inhibit_return=True,
                 finished_state=0.0,
                 device=torch.device("cpu")):
        self.grid = grid
        self.vd = visual_degree
        self.max_time = max_time
        self.mask_size = mask_size
        self.inhibit_return = inhibit_return
        self.max_step = 8
        self.device = device
        # self.screen_mask_index = screen_mask_index
    def observe(self, accumulate=True):
        active_indices = self.is_active
        if torch.sum(active_indices) == 0:
            print('active_indices:', torch.sum(active_indices))
            return None
        if self.step_id > 0:
            lastest_fixation_on_feats = self.fixations[:, self.step_id].to(dtype=torch.float32)
            px = lastest_fixation_on_feats[:, 0]
            py = lastest_fixation_on_feats[:, 1]
            fix = lastest_fixation_on_feats[:, 2]
            masks = []
            # ior_mask = > 0
            for i in range(self.batch_size):
                # self.ior_time[i, ior_mask[i]]
                mask = foveal2mask(px[i].item(), py[i].item(), 2, self.states[0].size(-2), self.states[0].size(-1))
                mask = torch.from_numpy(mask).to(self.device)
                mask = mask.unsqueeze(0).repeat(self.states[0].size(1), 1, 1)
                masks.append(mask)
            masks = torch.stack(masks)
            if accumulate:
                self.states[0] = (1 - masks) * self.states[0] + masks * self.high_feats
                self.states[1] = (1 - masks[:, 0:2]) *self.states[1] + masks[:, 0:2] * self.cam_feats_high
            self.history_map = (1 - masks[:, 0]) * self.history_map + masks[:, 0]
            # self.saliency_map = 
        ext_states = [self.states[0].clone(), self.states[1].clone(), self.ior_time.clone().unsqueeze(1)]
        return ext_states
    def get_reward(self):
        # reward = torch.zeros(self.batch_size, 3, device=self.device)
#         done_mask = self.status.bool()
#         if done_mask.sum() == 0:
#             return reward
#         real_visit = self.visit_index[done_mask].bool()
#         agent_visit = self.visit_matrix[done_mask].bool()
        
#         missed = real_visit & (~agent_visit)
#         missed_penalty = missed.sum(dim=1).float() * (-100.0)

        # extra = (~real_visit) & agent_visit
        # extra_penalty = extra.sum(dim=1).float() * (-1.0)

        # total_reward = missed_penalty + extra_penalty
        # reward[done_mask, 0] += missed_penalty
        reward = self.reward.clone()
        return reward

    def status_update(self):
        done = ((self.time > 3000).squeeze() + (self.step_id >= self.max_step)).bool()
        self.status = (self.status + done).bool()
        self.status = self.status.to(torch.uint8)

    def step(self, action_batch):
        self.step_id += 1
        assert self.step_id < self.max_step, "Error: Exceeding maximum step!"
        position, saccade, fixation = action_batch[:, 0].unsqueeze(-1).to(self.device), action_batch[:, 1].long().unsqueeze(-1).to(self.device), action_batch[:, 2].long().unsqueeze(-1).to(self.device)
        # self.reward[(screen_mask_index[position.squeeze().long()] == -1), 0] += -100.0
        # same_location = (self.fixations[:, self.step_id, 0:2] == self.fixations[:, self.step_id-1, 0:2]).all(dim=1) & \
        #             (self.fixations[:, self.step_id-1, 0:2] == self.fixations[:, self.step_id-2, 0:2]).all(dim=1)
        # self.reward[same_location, 0] += -100.0
        pos_index = position.squeeze().long()
        ior_temp_maps = self.ior_time_temp
        screen_ids = screen_mask_index[pos_index]
        screen_mask_index_2d = screen_mask_index.view(27, 48)
        for i in range(self.batch_size):
            sid = screen_ids[i].item()
            if sid == -1:
                # self.reward[i, 0] += -50  # fix 在空白屏幕也可选择惩罚
                continue
            ior_map = ior_temp_maps[i]
            avg_ior_by_img = []
            for img_id in range(4):
                img_mask = (screen_mask_index_2d == img_id).to(self.device)  # [27, 48]
                avg_ior_by_img.append(ior_map[img_mask].mean())
            # avg_ior_by_img = torch.stack(avg_ior_by_img) 
            # cur_ior = avg_ior_by_img[sid]
            # max_ior = avg_ior_by_img.max()
            # if cur_ior < max_ior and cur_ior < 1.0:
            #     self.reward[i, 0] += -100 * (1 - cur_ior)
                
        self.time_mask[:, :, 0] = self.time_mask[:, :, 0] + (self.time_idx.unsqueeze(0) < self.time + saccade) & (self.time_idx.unsqueeze(0) >= self.time)
        self.time += saccade
        self.time_mask[:, :, 1] = self.time_mask[:, :, 1] + (self.time_idx.unsqueeze(0) < self.time + fixation) & (self.time_idx.unsqueeze(0) >= self.time)
        self.time += fixation
        # print(self.time)
        # update fixation
        py, px = position // self.grid[1], position % self.grid[1]
        # print(py, px)
        self.fixations[:, self.step_id, 1] = py.squeeze()
        self.fixations[:, self.step_id, 0] = px.squeeze()
        self.fixations[:, self.step_id, 2] = fixation.squeeze()
        # update action mask
        before_action_mask = self.action_mask.clone()
        # ior_mask = self.ior_map > 0
        if self.inhibit_return:
            action_idx = torch.arange(0, self.batch_size, device=self.device, dtype=torch.long)
            if self.mask_size == 0:
                self.action_mask[action_idx, position.long()] = 1
            else:
                batch_size = self.action_mask.size(0)
                px, py = px.to(dtype=torch.long), py.to(dtype=torch.long)
                self.action_mask = self.action_mask.view(self.batch_size, self.grid[0], self.grid[1])
                self.action_time = self.action_time.view(self.batch_size, self.grid[0], self.grid[1])
                self.ior_map = self.ior_map.view(self.batch_size, self.grid[0], self.grid[1])
                self.ior_time = self.ior_time.view(self.batch_size, self.grid[0], self.grid[1])
                for i in range(self.batch_size):
                    self.action_mask[i, max(py[i] - self.mask_size, 0): py[i] + self.mask_size + 1,
                    max(px[i] - self.mask_size, 0): px[i] + self.mask_size + 1] = 1
                    self.action_time[i, py[i].item(), px[i].item()] = fixation.squeeze()[i]
                    duration = saccade.squeeze()[i] + fixation.squeeze()[i]
                    mask = (screen_mask_index!=-1).view(self.grid[0], self.grid[1]).double()
                    mask_time = (self.ior_time[i] > 0) & (mask == True)
                    self.ior_time[i][mask_time] += duration
                    temp_index = position.long()[i].detach().cpu()
                    if screen_mask_index[temp_index] != -1:
                        mask_temp = (screen_mask_index == temp_index).view(self.grid[0], self.grid[1])
                        tty, ttx = min(py[i].item(), 26), min(px[i].item(), 47)
                        temp_ior = templates[(tty, ttx)].to(self.device) * mask_temp.double()
                        self.ior_spat[i] = self.ior_spat[i] + temp_ior
                        self.ior_time[i][mask_temp] += fixation.squeeze()[i]
                        self.saliency_map[i] += templates[(py[i].item(), px[i].item())].to(self.device) *  fixation.squeeze()[i]
                    ior_temp = calculate_ior_temporal(self.ior_time[i]) 
                    self.ior_map[i, py[i].item(), px[i].item()] = 1e-8
                    self.ior_map[i] = self.ior_spat[i] * (1 + ior_temp) 
                self.action_mask = self.action_mask.view(self.batch_size, -1)
                self.action_time = self.action_time.view(self.batch_size, -1)
                self.ior_map = self.ior_map.view(self.batch_size, -1)
                # print(self.ior_map)
                # plt.imshow(self.ior_map.cpu().numpy())
                # plt.show()
            if self.action_mask.sum() - before_action_mask.sum() == 0:
                print('error!!!')
                action_mask = before_action_mask.view(batch_size, self.grid[0], -1)  
        obs = self.observe()
        self.status_update()
        return obs, self.status
    def reset(self):
        self.time_idx = torch.arange(3000).to(self.device)
        self.step_id = 0
        # self.label_coding = get_label_coding(self.cam_feats_low.clone(), self.low_feats.clone())
        self.time = torch.zeros((self.batch_size, 1), dtype=torch.long, device=self.device)
        self.time = self.time + self.init_fix[:, 2].unsqueeze(-1)
        self.time_mask = torch.zeros((self.batch_size, self.max_time, 2), dtype=torch.long, device=self.device)
        self.time_mask[:, :, 0] = (self.time_idx.unsqueeze(0) < self.init_fix[:, -1].long().unsqueeze(1)) & (self.time_idx.unsqueeze(0) >= self.time)
        self.fixations = torch.zeros((self.batch_size, 15, 3), dtype=torch.long, device=self.device)
        self.status = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)
        self.is_active = torch.ones(self.batch_size, dtype=torch.uint8, device=self.device)
        self.states = [self.low_feats.clone(), self.cam_feats_low.clone()]
        self.action_mask = self.init_action_mask.clone()
        self.action_time = self.init_action_time.clone()
        self.ior_spat = torch.zeros((self.batch_size, 27, 48), dtype=torch.float64, device=self.device)
        self.ior_map = torch.ones((self.batch_size, 27, 48), dtype=torch.float64, device=self.device)
        self.ior_time = torch.zeros((self.batch_size, 27, 48), dtype=torch.float64, device=self.device)
        self.ior_time_temp = torch.ones((self.batch_size, 27, 48), dtype=torch.float64, device=self.device)
        self.history_map = self.init_history_map.clone()
        self.visit_matrix = torch.zeros((self.batch_size, 4), dtype=torch.uint8, device=self.device)
        self.fixations[:, 0, 0] = self.init_fix[:, 0].to(self.device)
        self.fixations[:, 0, 1] = self.init_fix[:, 1].to(self.device)
        self.fixations[:, 0, 2] = self.init_fix[:, 2].to(self.device)
        self.reward = torch.zeros(self.batch_size, 3, device=self.device)
    def normalize_cam_feats_by_region(self, cam_feats, screen_mask_index = screen_mask_index):
        for i in range(cam_feats.size(0)):
            for img_id in range(4):
                mask = (screen_mask_index == img_id).to(cam_feats.device)
                for c in range(cam_feats.size(1)):
                    region = cam_feats[i, c][mask]
                    mean = region.mean()
                    std = region.std() + 1e-6
                    cam_feats[i, c][mask] = (region - mean) / std
        return cam_feats
    def set_data(self, data):
        self.init_fix = torch.stack(data['init_fix'], dim=1).to(self.device)
        self.init_action_mask = data['action_mask'].to(self.device)
        self.init_action_time = data['action_time'].to(self.device)
        self.init_history_map = data['history_map'].to(self.device)
        self.saliency_map = data['saliency_map'].to(self.device)
        self.cam_feats_high = data['cam_feats_high'].double().to(self.device) * (screen_mask_index !=-1).view(27,48)
        self.cam_feats_low = data['cam_feats_low'].double().to(self.device) * (screen_mask_index !=-1).view(27,48)
        self.cam_feats_high = self.normalize_cam_feats_by_region(self.cam_feats_high, screen_mask_index.view(27, 48))
        self.cam_feats_low  = self.normalize_cam_feats_by_region(self.cam_feats_low,  screen_mask_index.view(27, 48)) 
        self.low_feats = data['low_feats'].double().to(self.device)
        self.high_feats = data['high_feats'].double().to(self.device)
        # self.action_mask = data['action_mask'].to(self.device)
        self.batch_size = self.high_feats.size(0)
        self.cues = torch.stack(data['cues']).to(self.device)
        # self.label = data['label'].to(self.device)
        self.subject = data['subject'].to(self.device)
        self.label = data['label'].to(self.device)
        self.soft_label = data['soft_label'].to(self.device)
        self.img_index = torch.stack(data['img_index'], dim = 1).to(self.device)
        self.trail_index = data['trail_index'].to(self.device)
        self.visit_index = data['visit_index'].to(self.device)
        if self.inhibit_return:
            self.action_mask = data['action_mask'].to(self.device).view(self.batch_size, -1)
            self.action_time = data['action_time'].to(self.device).view(self.batch_size, -1)
        else:
            self.action_mask = torch.zeros(self.batch_size, 1296, dtype=torch.uint8)
        self.reset()