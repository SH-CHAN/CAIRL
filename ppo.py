import torch
from torch.distributions import Categorical
import torch.optim as optim
from addiction.models import *
from .utils import *
from .subject import *

import torch.optim as optim
class PPO():
    def __init__(self,
                 policy = None,
                 lr = 0.00001,
                 betas = [0.9,0.99],
                 batch_size = 64,
                 num_epoch = 10,
                 epilson = 0.2, 
                 value_coef=1.,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 device = device,
                 logger = None,
                 writer = None):
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy = policy
        self.betas = betas
        self.num_epoch = num_epoch
        self.minibatch_size = batch_size
        self.epilson = epilson
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas) 
        self.logger = logger
        self.device = device
    def update(self, rollouts):
        for i_epoch in range(self.num_epoch):    
            avg_loss = 0.0
            data_generator = rollouts.get_generator(self.minibatch_size)
            value_loss_epoch = 0.0
            action_loss_epoch = 0.0
            dist_entropy_epoch = 0.0
            loss_epoch = 0.0
            for i, batch in enumerate(data_generator):
                # print(batch['times'])
                # print(batch['soft_labels'].shape)
                states, actions, fix_time, labels  = batch['states'].to(self.device), batch['actions'].to(self.device), batch['times'], batch['soft_labels'].to(self.device)
                log_probs, returns, advantages = batch['log_probs'].to(self.device), batch['returns'].to(self.device), batch['advantages'].to(self.device)
                # spent_time = batch['spent_times'][:, 0]
                # fix_time = times[:, 0]
                # time_remaining = (3000 - spent_time).squeeze()
                # print(labels.shape, states.shape[0])
                if labels.shape[0] != states.shape[0]:
                    labels = labels.view(states.shape[0], -1)
                # print(labels.shape,states.shape)
                    # time_remaining = time_remaining.unsqueeze(0)
                # labels = F.one_hot(labels.long(), num_classes=2).float()
                # print(states.shape, fix_time.shape, labels.shape, time_remaining.shape, actions.shape)
                action_log_probs, values, entropy  = self.policy.get_log_prob(states, labels, actions)
                values = values.squeeze(-1)
                ratio = torch.exp(action_log_probs -  log_probs)
                surr1 = ratio * advantages 
                surr2 = torch.clamp(ratio, 1.0 - self.epilson, 1.0 + self.epilson) * advantages
                loss_actor = -torch.min(surr1, surr2).mean(0)
                # print(returns.shape, values.shape)
                loss_critic = F.smooth_l1_loss(returns, values, reduction='none').mean(0)
                loss_entropy = (-1.0 * entropy.squeeze(-1) * self.entropy_coef).mean(0)

                loss = (loss_actor  + loss_critic + loss_entropy).mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                value_loss_epoch += loss_critic.mean().item()
                action_loss_epoch += loss_actor.mean().item()
                dist_entropy_epoch += loss_entropy.mean().item()
                loss_epoch += loss.item()
                
            value_loss_epoch /= i + 1
            action_loss_epoch /= i + 1
            dist_entropy_epoch /= i + 1
            loss_epoch /= i + 1
            if self.logger is not None:
                self.logger.info(f'[{i_epoch}, {self.num_epoch}] :policy loss: {action_loss_epoch}, value loss : {value_loss_epoch}, entropy loss : {dist_entropy_epoch}, loss:{loss_epoch}')
            avg_loss += loss_epoch
        avg_loss /= self.num_epoch
        return avg_loss