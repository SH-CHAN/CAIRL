import os
import torch
import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader
from torch.utils.tensorboard import SummaryWriter
from .ppo import PPO
from .airl import AIRL
# from . import utils
from .utils import *
from .subject import *
from .data import *
from .environment import *
from .models import *
from .ppo import *

import torch.nn.functional as F
import torch.nn as nn
import logging
import os
import torch
import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

import os
import torch
import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime


class Trainer(object):
    def __init__(self,
                 train_img_loader = None,
                 valid_img_loader = None,
                 test_img_loader = None,
                 train_gaze_loader = None,
                 valid_gaze_loader = None,
                 test_gaze_loader = None,
                 fold = 'fold1',
                 gamma = 0.9,
                 num_epoch = 80,
                 num_step = 4,
                 num_critic = 1,
                 batch_size = 32,
                 max_traj_length = 8,
                 tau = 0.96,
                 env = None,
                 policy = None,
                 discriminator = None,
                 traj_net = None,
                 device = device, 
                 check_every = 100,
                 evaluate_every = 20, 
                 max_checkpoints = 5,
                 resume_from = None):
        self.log_dir = '/root/autodl-tmp/addiction/log/'
        self.checkpoints_dir = '/root/autodl-tmp/addiction/log/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.loaded_step = 0
        self.device = device
        self.checkpoint_every = check_every
        self.max_checkpoints = max_checkpoints
        self.batch_size = batch_size
        self.n_epoches = num_epoch
        self.n_steps = num_step
        self.n_critic = num_critic
        self.gamma = gamma
        self.max_traj_len = max_traj_length
        self.env = env
        self.env_valid = env
        self.tau = tau
        self.generator = policy
        self.discriminator = discriminator
        self.traj_net = traj_net
        self.optimizer_traj = torch.optim.Adam(self.traj_net.parameters(), lr=0.0001)
        
        
        self.eval_every = 20
        
        self.train_img_loader = train_img_loader
        self.valid_img_loader = valid_img_loader
        self.test_img_loader  = test_img_loader
        
        self.train_gaze_loader = train_gaze_loader
        self.valid_gaze_loader = valid_gaze_loader
        self.test_gaze_loader  = test_gaze_loader
        self.evaluate_every = 10
        
        self.fold = fold
        # self.log_dir = f'/root/autodl-tmp/Scanpath_Prediction/addiction/log/{self.fold}'
        self.log_dir = f'/root/autodl-tmp/addiction/log/fold1'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger(self.log_dir, f'{self.fold}_log_{timestamp}')
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.fold))
        self.ppo = PPO(policy = self.generator, logger = self.logger)
        self.airl = AIRL(self.discriminator, self.generator, logger = self.logger)
        # state_dict = torch.load('/root/autodl-tmp/Scanpath_Prediction/addiction/checkpoint/fold0/fold_0_pretrained_epoch30_1e2.pth', weights_only=True)
        # self.traj_net.load_state_dict(state_dict['traj_net'])
        # self.discriminator.load_state_dict(state_dict['discriminator'])
        # self.generator.load_state_dict(state_dict['generator'])
        self.global_step = 0
        self.loaded_epoch = 0
        self.loaded_step = 0
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        else:
            state_dict = torch.load('/root/autodl-tmp/addiction/checkpoint/fold1/fold_1_pretrained_epoch30_1e2.pth', weights_only=True)
            self.traj_net.load_state_dict(state_dict['traj_net'])
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.traj_net.load_state_dict(checkpoint['traj_net'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        # self.optimizer_traj.load_state_dict(checkpoint['optimizer_traj'])
        # self.ppo.load_state_dict(checkpoint['ppo'])  # 同样你要在 PPO 类里写 load_state_dict()
        self.loaded_epoch = checkpoint['epoch']
        self.loaded_step = checkpoint['global_step']
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}, continue from epoch {self.loaded_epoch}, global_step {self.global_step}")

    def train(self,label_tables = label_tables):
        self.generator.train()
        self.discriminator.train()
        self.traj_net.train()
        self.global_step = 212
        # created_folder = create_experiment_folder(self.log_dir)
        # logger = setup_logger(os.path.join(self.log_dir, self.fold), 'train_log.log')
        
        self.logger.info('calculate probs for trajectories')
        # label_tables = cal_labels(self.traj_net, self.train_img_loader, self.device)
        self.logger.info('finised calculate probs for trajectories')
        
        for i_epoch in range(0, self.n_epoches):
            for i_batch, batch in enumerate(self.train_img_loader):
                if i_epoch % 10 == 0 and i_epoch != 0:
                    probs_label = self.traj_net(batch['traj_data'].to(self.device))
                    self.optimizer_traj.zero_grad()
                    loss_classfier = nn.CrossEntropyLoss()(probs_label, batch['label'].long().to(self.device))
                    loss_classfier.backward()
                    self.optimizer_traj.step()
                    self.logger.info('calculate probs for trajectories')
                    label_tables = cal_labels(self.traj_net, self.train_img_loader, self.device)
                    self.logger.info('finised calculate probs for trajectories')
                batch_soft_label = torch.stack([label_tables[(subj.item(), idx.item(), cue.item())]
                        for subj, idx, cue in zip(batch['subject'], batch['trail_index'], torch.sum(torch.stack(batch['cues']), dim = 0))
                        if (subj.item(), idx.item(), cue.item()) in label_tables
                    ], dim = -1).T
                
                batch['soft_label'] = batch_soft_label
                trajs_all = []
                self.env.set_data(batch)
                return_train = 0.0
                self.logger.info('collect trajectories!')
                for i_step in range(self.n_steps):
                    with torch.no_grad():
                        self.env.reset()
                        trajs = collect_trajs(self.env, self.generator, self.max_traj_len)
                        trajs_all.extend(trajs)
                smp_num = np.sum(list(map(lambda x: x['length'], trajs_all)))
                # logger.info('collect trajectories!')
                self.logger.info(f"[epoch:{i_epoch} step:{i_batch}] Collected {smp_num} state-action pairs...")
                
                fake_data = FakeDataRollout(trajs_all,  self.batch_size)
                D_loss = self.airl.update(self.train_gaze_loader, fake_data, label_tables)
                
                if i_batch % self.n_critic == 0:
                    self.ppo.num_epoch = 10
                    if i_batch % (4 * self.n_critic) == 0 and i_batch != 0:
                        self.ppo.num_epoch = 20
                    if i_batch % (20 * self.n_critic) == 0 and i_batch != 0:
                        self.ppo.num_epoch = 40
                # if i_batch % 1 == 0:
                    with torch.no_grad():
                        for i in range(len(trajs_all)):
                            state = trajs_all[i]['curr_states'].to(self.device)
                            action = trajs_all[i]["actions"].squeeze(1).to(self.device)
                            label  = trajs_all[i]['soft_label'].to(self.device)
                            time   = trajs_all[i]['times'][:-1].to(self.device)
                            # spent_time   = trajs_all[i]['spent_time'][:-1]
                            log_pi = trajs_all[i]['log_probs'].to(self.device)
                            next_state = trajs_all[i]['next_states'].to(self.device)
                            # print(state.shape, action.shape, label.shape, time.shape)
                            # print(labels.shape)
                            if label.shape[0] != state.shape[0]:
                                label = label.view(state.shape[0], -1).to(self.device)
                                # spent_time = spent_time.unsqueeze(0)
                            '''calculate rewards'''
                            rewards = discriminator.calculate_reward(state, action, label, next_state,  log_pi)
                            # rewards += se
                            trajs_all[i]['rewards'] = trajs_all[i]['rewards'].to(self.device)
                            trajs_all[i]['rewards'] += rewards
                    return_trajs_train = compute_trajs_returns(trajs_all, 0.99, mtd='GAE', tau=0.96)
                    print(torch.mean(return_trajs_train).item())
                    rollouts = RolloutStorage(trajs_all, shuffle=True, norm_adv=True)
                    loss_ppo = self.ppo.update(rollouts)
                    # self.writer.add_scalar('Reward/train_return', torch.mean(return_train), self.global_step)
                    # self.writer.add_scalar('ppo loss', loss_ppo, self.global_step)
                    # print()
                    self.logger.info("Done updating policy")
                checkpoint = {
                    'epoch': i_epoch,
                    'global_step': self.global_step,
                    'traj_net':self.traj_net.state_dict(),
                    'generator': self.generator.state_dict(),
                    'discriminator':self.discriminator.state_dict()
                }
                torch.save(checkpoint, f'/root/autodl-tmp/addiction/log/fold1/{self.fold}_checkpoint_epoch_{i_epoch}_{self.global_step}.pth')
                # torch.save(checkpoint, f'{os.path.join(self.checkpoints_dir, self.fold)}/{self.fold}_checkpoint_epoch_{i_epoch}_{self.global_step}.pth')
                self.global_step += 1
        self.writer.close()
