import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, TopKPooling
import torchvision.models as models
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Rectangle
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, KFold
from .utils import *
from .subject import *
import warnings
import gc
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-scatter'")

class TrajectoryDatasets():
    def __init__(self,
                 root = '/root/autodl-tmp/Drug/Drug_Datasets_pkl',
                 kind='ice_drug',
                 subject_drug = subject_icedrug,
                 subject_control = control_icedrug,
                 cues=[3, 4, 5],
                 test_ratio=0.2,
                 random_state=42,
                 k_fold=5):
        self.kind = kind
        self.cues = cues
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.k_fold = k_fold
        self.root_path = root
        self.subject_drug = subject_drug
        self.subject_control = subject_control

    def split_datasets(self, subject):
        train_val_set, test_set = train_test_split(subject, test_size=self.test_ratio, random_state=self.random_state)
        kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=self.random_state)
        train_val_set_fold = list(kf.split(train_val_set))
        combined_folds = []
        for i in range(self.k_fold):
            train_subjects, val_subjects = [], []
            train_subjects.extend(list(np.array(train_val_set)[train_val_set_fold[i][0]]))
            val_subjects.extend(list(np.array(train_val_set)[train_val_set_fold[i][1]]))
            combined_folds.append((train_subjects, val_subjects, test_set))
        return combined_folds

    def generate_learning_datasets(self, subject_drug, stage=3, control=False, fold = 1):
        combined_folds = self.split_datasets(subject_drug)
        cues = self.cues[0:stage]
        abnoramal_sets = []
        combined_sets_img = {3: [], 4: [], 5: []}
        combined_sets_gaze = {3: [], 4: [], 5: []}
        # combined_sets_traj = {3: [], 4: [], 5: []}
        # combined_sets_cdf = {3: [], 4: [], 5: []}
        trajs_cdf = []
        for cue in cues:
            count = 0
            for train_set, val_set, test_set in combined_folds:
                if count == fold:
                    train_data, train_gaze, abnoraml_train = self.read_subjects(train_set, cue, control)
                    val_data, val_gaze, abnoraml_val = self.read_subjects(val_set, cue, control)
                    test_data, test_gaze, abnoraml_test = self.read_subjects(test_set, cue, control)

                    combined_sets_img[cue].append((train_data, val_data, test_data))
                    combined_sets_gaze[cue].append((train_gaze, val_gaze, test_gaze))
                    # combined_sets_traj[cue].append((train_traj, val_traj, test_traj))
                    # trajs_cdf.append(val_cdf)
                    abnoramal_sets = abnoraml_train + abnoraml_val + abnoraml_test
                    break
                count += 1
        # gc.collect() 
        return combined_sets_img, combined_sets_gaze, abnoramal_sets
        # return combined_sets_img, combined_sets_gaze, None, None, abnoramal_sets

    def read_subjects(self, subject_set, cue, control=False):
        data_normal, data_abnormal = {}, []
        gaze_normal = []
        traj_normal = []
        traj_cdf = []
        kind = f'control_{self.kind}' if control else self.kind
        folder_path = f'{self.root_path}/{cue}/{self.kind}/{kind}'
        subjects = [os.path.join(folder_path, subject) for subject in subject_set]
        for folder_name in tqdm(subjects, desc=f'process {kind} {cue} data'):
            for i in range(10):
                path_data = os.path.join(folder_name, f'{i}.pkl')
                data = self.preprocess(path_data, f'{os.path.basename(folder_name)}_{cue}_{i}')
                # print(data[0])
                if data[-1]:
                    data_normal.update(data[0])
                    gaze_normal.extend(data[1])
                    # traj_normal.append(process_data(data[2]))
                    # traj_cdf.extend(process_data(data[3]))
                else:
                    data_abnormal.append(path_data)
        # print(data_normal)
        return data_normal, gaze_normal, data_abnormal

    def get_datasets(self, fold = 0):
        subject_drug, subject_control = self.subject_drug, self.subject_control
        s_img, s_gaze, _ = self.generate_learning_datasets(subject_drug, stage=3, fold = fold)
        c_img, c_gaze, _ = self.generate_learning_datasets(subject_control, stage=3, control=True ,fold = fold)

        train_sets_img, val_sets_img, test_sets_img = {}, {}, {}
        train_sets_gaze, val_sets_gaze, test_sets_gaze = [], [], []
        # trajs_valid_noraml, trajs_valid_addict = [], []
        for cue in [3, 4, 5]:
            train_img_data, val_img_data, test_img_data = {}, {}, {}
            train_img_data.update(s_img[cue][fold][0])
            train_img_data.update(c_img[cue][fold][0])
            val_img_data.update(c_img[cue][fold][1])
            val_img_data.update(c_img[cue][fold][1])
            test_img_data.update(c_img[cue][fold][2])
            test_img_data.update(c_img[cue][fold][2])
            # train_img_data = s_img[cue][fold][0] + c_img[cue][fold][0]
            # val_img_data = s_img[cue][fold][1] + c_img[cue][fold][1]
            # test_img_data = s_img[cue][fold][2] + c_img[cue][fold][2]

            train_gaze_data = s_gaze[cue][fold][0] + c_gaze[cue][fold][0]
            val_gaze_data = s_gaze[cue][fold][1] + c_gaze[cue][fold][1]
            test_gaze_data = s_gaze[cue][fold][2] + c_gaze[cue][fold][2]

            train_sets_img.update(train_img_data)
            val_sets_img.update(val_img_data)
            test_sets_img.update(test_img_data)

            train_sets_gaze.extend(train_gaze_data)
            val_sets_gaze.extend(val_gaze_data)
            test_sets_gaze.extend(test_gaze_data)
            
            # train_sets_traj.extend(process_data(train_traj_data))
            # val_sets_traj.extend(process_data(val_traj_data))
            # test_sets_traj.extend(process_data(test_traj_data))
            
            # trajs_valid_noraml.extend(process_data(s_gaze[cue][fold][1]))
            # trajs_valid_addict.extend(process_data(c_gaze[cue][fold][1]))
            print('check data finished!')
            
        # human_mean_cdf_noraml, _ = compute_search_cdf(trajs_valid, 15)
        # human_mean_cdf_addict, _ = compute_search_cdf(trajs_valid, 15)
        
        return {
            'train_img_data': Image_Data(train_sets_img),
            'val_img_data': Image_Data(val_sets_img),
            'test_img_data': Image_Data(test_sets_img),
            'train_gaze_data': Gaze_Data(train_sets_gaze),
            'val_gaze_data':Gaze_Data(val_sets_gaze),
            'test_gaze_data':Gaze_Data(test_sets_gaze),
            # 'train_gaze_data': Traj_Data(train_sets_traj),
            # 'val_gaze_data': Traj_Data(val_sets_traj),
            # 'test_gaze_data': Traj_Data(test_sets_traj),
            # 'noraml_mean_cdf': human_mean_cdf_noraml,
            # 'addict_mean_cdf': human_mean_cdf_addict,
        }
    
    def preprocess(self, data_path, name):
        flag = True
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        # print(data['subject'])
        time, px, py, mask, state, label = np.array(data['trajectory_data']['time']), np.array(
            data['trajectory_data']['px']), np.array(data['trajectory_data']['py']), np.array(
            data['trajectory_data']['mask'], dtype=np.float32), np.array(data['trajectory_data']['state']), np.array(
            data['label'])

        if np.sum(state) == 0:
            flag = False
            return data, flag

        final_fixation_peaks = data['final_fixation_peaks']
        velocity = np.insert(calculate_velocity(px, py), 0, 0)
        acceleration = np.insert(calculate_acceleration(velocity), 0, 0)
        if final_fixation_peaks is None or len(final_fixation_peaks) <= 1:
            flag = False
            return data, flag
        # TODO
        # fix_no_bg = [item[-1] for item in final_fixation_peaks if item[-1] != -1]
        # # fixs_steps[data['label']].append(len(final_fixation_peaks))
        # fixs_steps_nobg[data['label']].append(len(fix_no_bg))
        # node feature : x, y, v, a
        # state = F.one_hot(state, num_classes=3).to(device).float()
        # print(state.shape)
        state_vector = np.eye(3)
        node_features = torch.FloatTensor(np.column_stack([time, px, py, velocity, acceleration, state_vector[state], mask]))
        node_features[:, 0] /= 3000  # 归一化
        node_features[:, 1] /= 1920  # 归一化
        node_features[:, 2] /= 1080  # 归一化
        node_features[:, 3] = (node_features[:, 3] - torch.mean(node_features[:, 3])) / torch.std(node_features[:, 3])
        node_features[:, 4] = (node_features[:, 4] - torch.mean(node_features[:, 4])) / torch.std(node_features[:, 4])
        # print(node_features.shape)
        edge_index = torch.tensor([[i, i + 1] for i in range(node_features.shape[0] - 1)], dtype=torch.long).T
        graph_data = Data()
        graph_data.x = node_features.double()

        graph_data.edge_index_spatial = edge_index.long()
        graph_data.spatial_weights = torch.Tensor(velocity[1:]).double()

        graph_data.edge_index_temporal = edge_index.long()
        graph_data.temporal_weights = torch.ones(len(edge_index[-1])).double()
        init_fix = (final_fixation_peaks[0][2], final_fixation_peaks[0][3], final_fixation_peaks[0][1] - final_fixation_peaks[0][0])
        index = int(data_path.split('.')[0][-1])
        fix_index = torch.tensor([any(a[-2] == i for a in final_fixation_peaks if a[-2] != -1) for i in range(4)], dtype=torch.bool)
        
        temp_scan = []
        for i, scan in enumerate(final_fixation_peaks):
            if scan[-1] != -1 or i == 0:
                x, y, t = scan[2], scan[3], scan[1] - scan[0]
                action = pos_to_action(x, y)         
                cx, cy = action_to_pos(action) 
                temp_scan.append([cx, cy, t])
        sub = subjects.index(data['subject'])
        human_scanpaths[(sub, index, sum(list(data['cue_position'].values())))] = np.array(temp_scan)
        
        # print(fix_index)
        # for i in range(len(final_fixation_peaks)):
        #     if data['label'] == 'ice_drug':
        #         fix['addict'].append(final_fixation_peaks[i][1] - final_fixation_peaks[i][0])
        #         if i == 0:
        #             sac['addict'].append(final_fixation_peaks[i+1][0] - final_fixation_peaks[i][1])
        #         if i < len(final_fixation_peaks) - 1:
        #             sac['addict'].append(final_fixation_peaks[i+1][0] - final_fixation_peaks[i][1])
        #     else:
        #         fix['normal'].append(final_fixation_peaks[i][1] - final_fixation_peaks[i][0])
        #         if i == 0:
        #             sac['normal'].append(final_fixation_peaks[i+1][0] - final_fixation_peaks[i][1])
        #         # if i < len(final_fixation_peaks) - 1 and final_fixation_peaks[i][-2] != final_fixation_peaks[i-1][-2] and i != 0 :
        #         if i < len(final_fixation_peaks) - 1:
        #             sac['normal'].append(final_fixation_peaks[i+1][0] - final_fixation_peaks[i][1])
        # count = 0
        # for i in range(len(final_fixation_peaks)):
        #     if final_fixation_peaks[i-1][-2] != -1:
        #         if i == 0:
        #             count += 1
        #         if i != 0 and i <= len(final_fixation_peaks) - 1:
        #             count += 1
        # if data['label'] == 'ice_drug':
        #     fixs_steps['ice_drug'].append(count)
        # else:
        #     fixs_steps['control_ice_drug'].append(count) 
            
        selected_keys = ['cam_feats_high', 'cam_feats_low', 'high_feats', 'low_feats', 'label', 'img_index', 'subject', 'cue_position', 'init_fix']
        img_dict = {k: data[k] for k in selected_keys if k in data}
        # print(data['subject'], index, np.sum(list(data['cue_position'].values())))
        img_data = {(data['subject'], index, np.sum(list(data['cue_position'].values()))):(img_dict, init_fix, index, fix_index, graph_data)}
        # print(img_data)
        selected_keys = ['cam_feats_high', 'cam_feats_low', 'high_feats', 'low_feats', 'label', 'img_index', 'subject', 'cue_position', 'init_fix']
        gaze_dict = {k: data[k] for k in selected_keys if k in data}
        fixs_label = process_fixations(gaze_dict, index, final_fixation_peaks, remove_bg = True)
        return img_data, fixs_label, flag   
    
class Image_Data(Dataset):
    def __init__(self, img_data):
        self.img_data = list(img_data.values())
    
    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        # graph_data = self.img_data[idx][0]
        traj_data = self.img_data[idx][4]
        # print(traj_data)
        data = process_data(self.img_data[idx])[0]
        # print(data)
        init_fix = self.img_data[idx][1]
        index = self.img_data[idx][2]
        visit_index = self.img_data[idx][3]
        crh = data['cam_feats_high']
        crl = data['cam_feats_low']
        hr = data['high_feats']
        lr = data['low_feats']
         
        px, py, delta = init_fix[0], init_fix[1], init_fix[2]
        mask = torch.from_numpy(foveal2mask(px, py, 2 * 40, 1080, 1920))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=[27, 48]).squeeze(0).repeat(256, 1, 1)
        lr = lr * (1 - mask) + mask * hr
        crl = crl * (1 - mask[0:2]) + crh * mask[0:2]
        
        history_map = torch.zeros((27, 48))
        history_map = (1 - mask[0]) * history_map + mask[0] * 1

        action_mask = torch.zeros((27, 48), dtype=torch.uint8)
        action_time = torch.zeros((27, 48), dtype=torch.float32)
        action_mask[int(py // 40) - 1:int(py // 40) + 1 + 1, int(px // 40) - 1:int(px // 40) + 1 + 1] = 1
        action_time[int(py // 40), int(px // 40)] = delta
        saliency_map = templates[(int(py // 40), int(px // 40))] * delta
        return {
            'subject': subjects.index(data['subject']),
            'label': data['label'] == 'ice_drug',
            'trail_index': index,
            'img_index': data['img_index'],
            'cues': list(data['cue_position'].values()),
            'low_feats': lr.double(),
            'high_feats': hr.double(),
            'cam_feats_low': crl.double(),
            'cam_feats_high': crh.double(),
            'history_map': history_map,
            'action_mask': action_mask,
            'action_time': action_time,
            'visit_index':visit_index,
            'saliency_map': saliency_map,
            'init_fix': init_fix,
            'traj_data': traj_data
        }


class Gaze_Data(Dataset):
    def __init__(self, gaze_data, img_data):
        self.gaze_data = process_data(gaze_data)
        self.img_data = img_data
    def __len__(self):
        return len(self.gaze_data)

    def __getitem__(self, idx):
        # print(self.gaze_data[0])
        # print(self.gaze_data[idx])
        i, end = self.gaze_data[idx][0][0], self.gaze_data[idx][0][1]
        subject = self.gaze_data[idx][1]
        cues = self.gaze_data[idx][2]
        index = self.gaze_data[idx][3]
        class_names = self.gaze_data[idx][4]
        bg = self.gaze_data[idx][5]
        fixs = self.gaze_data[idx][6]
        actions = self.gaze_data[idx][7]
        # print(self.img_data[(subject, index)])
        data = process_data(self.img_data[(subject, index, np.sum(cues))])[0]
        crh = data['cam_feats_high'].unsqueeze(0).double()
        crl = data['cam_feats_low'].unsqueeze(0).double()
        hr = data['high_feats'].unsqueeze(0).double()
        lr = data['low_feats'].unsqueeze(0).double()
        states = [lr, crl]
        spent_time = 0.0
        history_map = torch.zeros((27, 48))  
        saliency_map = torch.zeros((27, 48))  
        fixation_map = torch.zeros((27, 48)) 
        ior_map = torch.zeros((27, 48))
        mask_screen = (screen_mask_index!=-1).view(27,48).double()
        for i in range(len(fixs)):
            px, py, delta_t, delta_s = fixs[i][0], fixs[i][1], fixs[i][2], fixs[i][3]
            tx, ty = int(px//40), int(py//40)
            spent_time += delta_t
            
            temp_ior = templates[(min(ty, 26), min(tx, 47))]
            ior_map = (ior_map + temp_ior)*mask_screen
            ior_map[temp_ior > 0] += (delta_t + delta_s)
            
            masks = foveal2mask(px, py, 2, states[0].size(-2), states[0].size(-1))
            masks = torch.from_numpy(masks)
            masks = masks.unsqueeze(0).repeat(states[0].size(1), 1, 1).unsqueeze(0)
            states[0] = (1 - masks) * states[0] + masks * hr
            states[1] = (1 - masks[:,0:2]) * states[1] + masks[:,0:2] * crh
            history_map = (1 - masks[:,0]) * history_map + masks[:,0] * 1
            fixation_map[(ty, tx)] = delta_t
            saliency_map += templates[(ty, tx)] * delta_t
        saliency_map = saliency_map / saliency_map.max()
        next_states = [states[0], states[1]]
        pos, sac, dur = actions
        py, px = pos // 27, pos % 27
        temp_ior = templates[(min(int(py), 26), min(int(px), 47))]
        ior_map_next = ior_map + temp_ior
        ior_map_next[temp_ior > 0] += (sac+dur)
        masks = foveal2mask(px, py, 2, states[0].size(-2), states[0].size(-1))
        masks = torch.from_numpy(masks)
        # print(masks.shape)
        masks = masks.unsqueeze(0).repeat(states[0].size(1), 1, 1).unsqueeze(0)
        next_states[0] = (1 - masks) * next_states[0] + masks * hr
        next_states[1] = (1 - masks[:,0:2]) * next_states[1] + masks[:,0:2] * crh
        # screen_img_id_masks = encode_image_id_with_cue_map(torch.tensor(cues, dtype=torch.long).unsqueeze(0), torch.tensor(data['img_index'], dtype=torch.long).unsqueeze(0))
        # screen_cues_masks = encode_image_attribute_map(torch.tensor(cues, dtype=torch.long).unsqueeze(0))
        # coords = get_coord_channels(bs = 1)
        # print(states[0].shape, states[1].shape, coords.shape, screen_cues_masks.shape, screen_img_id_masks.shape)
        # print(torch.cat([states[0], states[1], coords, screen_cues_masks, screen_img_id_masks],dim = 1).shape)
        # print(torch.cat([next_states[0], next_states[1], coords, screen_cues_masks, screen_img_id_masks],dim = 1).shape)
        # self.x_coords = self.x_coords.to(self.device)
        # self.y_coords = self.y_coords.to(self.device)
        
        # print(next_states[0].shape , ior_map.shape, ior_map_next.shape)
        ret = {
            'subject': subjects.index(data['subject']),
            'i':i,
            'cues': list(data['cue_position'].values()),
            'true_labels': data['label'] == 'ice_drug',
            'trail_index': index,
            'img_index': data['img_index'],
            'subject_trail': index,
            'true_states':torch.cat([states[0], states[1], ior_map.unsqueeze(0).unsqueeze(0)],dim = 1).squeeze(0),
            'true_actions':actions,
            'true_times': dur,
            'true_next_states':torch.cat([next_states[0], next_states[1],ior_map_next.unsqueeze(0).unsqueeze(0)],dim = 1).squeeze(0),
            'history_map':history_map, 
            'spent_times': spent_time,
            'saliency_map':saliency_map,
            'fixation_map':fixation_map,
            'end':end
        }
        return ret

class FakeDataRollout:
    def __init__(self, 
                 trajs_all, 
                 minibatch_size, 
                 shuffle=True, 
                 drop_last=True):
        self.subjects = torch.cat([traj['subjects'] for traj in trajs_all])
        self.trail_index = torch.cat([traj['trail_index'] for traj in trajs_all])
        # self.ior_time = torch.cat([traj['ior_time'] for traj in trajs_all])
        # self.ior_map = torch.cat([traj['ior_map'] for traj in trajs_all])
        self.states = torch.cat([traj['curr_states'] for traj in trajs_all])
        self.actions = torch.cat([traj['actions'] for traj in trajs_all]).unsqueeze(1)
        # self.saliency_map = torch.cat([traj['saliency_map'] for traj in trajs_all]).unsqueeze(1)
        self.times = torch.cat([traj['times'][:-1] for traj in trajs_all])
        self.spent_times = torch.cat([traj['spent_times'][:-1] for traj in trajs_all])
        self.label = torch.cat([traj['label'] for traj in trajs_all])
        self.soft_label = torch.cat([traj['soft_label'] for traj in trajs_all])
        self.next_states = torch.cat([traj['next_states'] for traj in trajs_all]) if 'next_states' in trajs_all[0] else None
        self.log_probs = torch.cat([traj["log_probs"] for traj in trajs_all if traj["log_probs"] is not None]).unsqueeze(1)
        self.cues =  torch.cat([torch.stack(traj['cues']) for traj in trajs_all])
        self.sample_num = self.states.size(0)
        self.shuffle = shuffle
        self.batch_size = min(minibatch_size, self.sample_num)
        # print(f'mini:{minibatch_size}, {self.sample_num}, {self.states.shape}')
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            return self.sample_num // self.batch_size
        else:
            return (self.sample_num + self.batch_size - 1) // self.batch_size  # ceil division

    def get_generator(self):
        indices = torch.randperm(self.sample_num) if self.shuffle else torch.arange(self.sample_num)
        
        # 生成 mini-batch 索引
        batches = [indices[i:i + self.batch_size] for i in range(0, self.sample_num, self.batch_size)]
        
        # 处理 drop_last
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        for batch_idx in batches:
            batch = {
                "subject": self.subjects[batch_idx],
                "trail_index": self.trail_index[batch_idx],
                "states": self.states[batch_idx],
                "actions": self.actions[batch_idx],
                "times": self.times[batch_idx],
                # "ior_time": self.ior_time[batch_idx],
                # "ior_map": self.ior_map[bacth_idx],
                "spent_times": self.spent_times[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "labels": self.label[batch_idx],
                "soft_labels": self.soft_label[batch_idx],
                "cues": self.cues[batch_idx]
            }
            if self.next_states is not None:
                batch["next_states"] = self.next_states[batch_idx]
            yield batch
            

class RolloutStorage(object):
    def __init__(self, trajs_all, shuffle=True, norm_adv=True):
        self.subjects = torch.cat([traj['subjects'] for traj in trajs_all])
        self.trail_index = torch.cat([traj['trail_index'] for traj in trajs_all])
        self.cues =  torch.cat([torch.stack(traj['cues']) for traj in trajs_all])
        self.states = torch.cat([traj["curr_states"] for traj in trajs_all])
        self.actions = torch.cat([traj["actions"] for traj in trajs_all])
        self.times = torch.cat([traj['times'][:-1] for traj in trajs_all])
        self.spent_times = torch.cat([traj['spent_times'][:-1] for traj in trajs_all])
        self.next_states = torch.cat([traj["next_states"] for traj in trajs_all])
        self.lprobs = torch.cat([traj['log_probs'] for traj in trajs_all])
        self.labels = torch.cat([traj['label'] for traj in trajs_all])
        self.slabels = torch.cat([traj['soft_label'] for traj in trajs_all]).view(self.labels.shape[0],-1)
        self.rewards = torch.cat([traj['rewards'] for traj in trajs_all])
        self.returns = torch.cat([traj['acc_rewards'] for traj in trajs_all])
        self.values = torch.cat([traj['values'][:-1] for traj in trajs_all])
        self.advs = torch.cat([traj['advantages'] for traj in trajs_all])
        if norm_adv:
            self.advs = (self.advs - self.advs.mean(0)) / (self.advs.std(0) + 1e-8)

        self.sample_num = self.states.size(0)
        self.shuffle = shuffle

    def get_generator(self, minibatch_size):
        indices = torch.randperm(self.sample_num) if self.shuffle else torch.arange(self.sample_num)

        # 生成 mini-batch 索引
        batches = [indices[i:i + minibatch_size] for i in range(0, self.sample_num, minibatch_size)]

        # 处理 drop_last
        if hasattr(self, "drop_last") and self.drop_last and len(batches[-1]) < minibatch_size:
            batches = batches[:-1]

        for batch_idx in batches:
            batch = {
                "subject": self.subjects[batch_idx],
                "trail_index": self.trail_index[batch_idx],
                "states": self.states[batch_idx],
                "actions": self.actions[batch_idx],
                "times": self.times[batch_idx],
                "cues":self.cues[batch_idx],
                "spent_times": self.spent_times[batch_idx],
                "log_probs": self.lprobs[batch_idx], 
                "labels": self.labels[batch_idx],  
                "soft_labels": self.slabels[batch_idx], 
                "next_states": self.next_states[batch_idx],
                "rewards": self.rewards[batch_idx],
                "returns": self.returns[batch_idx],
                "values": self.values[batch_idx],
                "advantages": self.advs[batch_idx]
            }
            yield batch