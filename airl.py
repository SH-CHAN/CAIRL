import torch
from torch import autograd
import torch.optim as optim
import torch.nn.functional as F
from .utils import *
from .subject import *
from .models import *
import torch.optim as optim

class AIRL():
    def __init__(self, 
                 discriminator = None, 
                 generator = None,
                 milestones = [10000], 
                 state_enc = None, 
                 device = device, 
                 lr = 0.00005,
                 betas = [0.9, 0.999],
                 logger = None):
        self.discriminator = discriminator
        self.policy = generator
        self.state_enc = state_enc
        self.device = device
        self.optimizer = optim.Adam(self.discriminator.parameters(),lr=lr,betas=betas)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        self.update_counter = 0
        self.logger = logger
        # self.logger
    def update(self,
               true_data_loader,
               fake_data,
               label_tables,
               iter_num=1,
               noisy_label_ratio = 0.01):
        running_loss = 0.0
        print_every = fake_data.sample_num // (5 * fake_data.batch_size) + 1
        avg_loss = 0.0
        D_real, D_fake = 0.0, 0.0
        fake_sample_num = 0
        real_sample_num = 0
        
        state_enc = None
        for i_iter in range(iter_num):
            fake_data_generator = iter(fake_data.get_generator())
            # print(fake_data_generator)
            for i_batch, true_batch in enumerate(true_data_loader):
                if i_batch == len(fake_data):
                    break
                fake_batch = next(fake_data_generator)
                with torch.no_grad():
                    if state_enc is None:
                        real_S = true_batch['true_states'].to(self.device).squeeze(1)
                        real_NS = true_batch['true_next_states'].to(self.device).squeeze(1)
                    else:
                        real_S = state_enc(true_batch['true_states'].to(self.device))
                        real_NS = state_enc(true_batch['true_next_states'].to(self.device))
                    
                    real_A = torch.stack(true_batch['true_actions'], dim=1).to(self.device)
                    # real_labels = F.one_hot(true_batch['true_labels'].long(), num_classes=2).to(self.device).float()
                    # real_labels = label_tables[()]
                    real_labels = torch.stack([label_tables[(subj.item(), idx.item(), cue.item())]
                        for subj, idx, cue in zip(true_batch['subject'], true_batch['trail_index'], torch.sum(torch.stack(true_batch['cues']), dim = 0))
                        if (subj.item(), idx.item(), cue.item()) in label_tables
                    ], dim = -1).T.to(self.device)
                    fake_labels = torch.stack([label_tables[(subj.item(), idx.item(), cue.item())]
                        for subj, idx, cue in zip(fake_batch['subject'], fake_batch['trail_index'], torch.sum(fake_batch['cues'], dim = 1))
                        if (subj.item(), idx.item(), cue.item()) in label_tables
                    ], dim = -1).T.to(self.device)
                    real_P, _, _ = self.policy.get_log_prob(real_S, real_labels, real_A)
                    
                fake_S, fake_A, fake_times,  = fake_batch['states'].to(self.device), fake_batch['actions'].squeeze(1).to(self.device), fake_batch['times'].to(self.device)
                fake_P, fake_NS = fake_batch['log_probs'].squeeze(1).to(self.device), fake_batch['next_states'].to(self.device)
                fake_num, real_num = fake_S.size(0), real_S.size(0)
                if fake_num == 0 or real_num == 0:
                        break
                        
                x_real = (real_S, real_A, real_labels, real_NS, real_P)
                x_fake = (fake_S, fake_A, fake_labels, fake_NS, fake_P)
                # print(real_labels.shape, fake_labels.shape)
                real_outputs = self.discriminator(*x_real)
                fake_outputs = self.discriminator(*x_fake)
                if noisy_label_ratio > 0:
                    flip_num = int(real_outputs.size(0) * noisy_label_ratio)
                    ind = torch.randint(real_outputs.size(0), (flip_num, ))
                    real_outputs[ind] = 0
                    fake_outputs[ind] = 1
                    
                loss_fake = -F.logsigmoid(-fake_outputs).mean(-1)
                loss_real = -F.logsigmoid(real_outputs).mean(-1)

                loss_airl = loss_fake.mean() + loss_real.mean()
                
                self.optimizer.zero_grad()
                loss_airl.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                if i_iter == iter_num - 1:
                    avg_loss += loss_airl.item()
                    real_sample_num += real_num
                    fake_sample_num += fake_num
                    
                # rewards = self.discriminator.calculate_reward(*x_real)
                # advantages, returns = estimate_advantages(rewards, masks, bad_masks, values, next_values, gamma, tau, device)
                
                running_loss += loss_airl.item()
                if i_batch % print_every == print_every - 1:
                    self.logger.info(f'i_bacth: {i_batch + 1}, AIRL loss: {running_loss / print_every}')
                    running_loss = 0.0

                self.update_counter += 1
        # return (avg_loss / fake_data.sample_num, D_real / real_sample_num, D_fake / fake_sample_num)
        return avg_loss / fake_data.sample_num