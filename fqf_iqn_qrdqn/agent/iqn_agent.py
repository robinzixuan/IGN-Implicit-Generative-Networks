import torch
from torch.optim import Adam
import torch.nn as nn

import numpy as np

from fqf_iqn_qrdqn.model import IQN
from fqf_iqn_qrdqn.utils import calculate_quantile_huber_loss, disable_gradients, evaluate_quantile_at_action, update_params

from .base_agent import BaseAgent
import torch.optim as optim

from torch.autograd import Variable
from torch import autograd
from fqf_iqn_qrdqn.network import DQNBase







class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        self.dqn_net = DQNBase(num_channels=num_channels)
        '''
        self.model = nn.Sequential(
            nn.Linear(64, 512),
            nn.LeakyLinear(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyLinear(0.2, inplace=True),
            nn.Linear(256, 64),
        )
        '''

        
    def forward(self, img, states = None, action = None):
        img_flat = img.view(img.shape[0], -1)
        batch_size = states.shape[0]
        action = torch.unsqueeze(action, dim=1).repeat(1, 64, 1)
        action= action.reshape(batch_size * 64, *action.shape[2:])
        states = torch.unsqueeze(states, dim=1).repeat(1, 64, 1, 1, 1)
        states = states.reshape(batch_size * 64, *states.shape[2:])
        state_embeddings = self.dqn_net(states)
        img = img.reshape(batch_size * 64, *img.shape[2:])

        print(img_flat.shape)
        print(img.shape)
        print(state_embeddings.shape)
        print(action.shape)
        
        #validity = self.model(img_flat)
        validity = evaluate_quantile_at_action(state_embeddings, action).transpose(1, 2)
        return validity



class IQNAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, N=64, N_dash=64, K=32, num_cosines=64,
                 kappa=1.0, lr=5e-5, memory_size=10**6, gamma=0.99,
                 multi_step=1, update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0):
        super(IQNAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, dueling_net, noisy_net, use_per, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed)

        # Online network.
        self.online_net = IQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, K=K, num_cosines=num_cosines,
            dueling_net=dueling_net, noisy_net=noisy_net).to(self.device)   #generator
        # Target network.
        self.target_net = IQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, K=K, num_cosines=num_cosines,
            dueling_net=dueling_net, noisy_net=noisy_net).to(self.device)

        self.discriminator = Discriminator(num_channels=env.observation_space.shape[0]).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        
        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.num_cosines = num_cosines
        self.kappa = kappa
        self.lr = lr
        self.n_critic = 5
        self.gamma = gamma
        self.discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=self.lr * 3)
        self.generator_optim = optim.Adam(self.online_net.parameters(),lr=self.lr)

    def learn(self):
        self.learning_steps += 1
        self.online_net.sample_noise()
        self.target_net.sample_noise()

        if self.use_per:
            (states, actions, rewards, next_states, dones), weights =\
                self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones =\
                self.memory.sample(self.batch_size)
            weights = None

        #print(self.memory.sample)
        # Calculate features of states.
        state_embeddings = self.online_net.calculate_state_embeddings(states)

        quantile_loss, mean_q = self.calculate_loss(states,
            state_embeddings,actions, rewards, next_states, dones, weights)
        '''
        update_params(
            self.generator_optim, quantile_loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)
        '''
        
        if 4*self.steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                4*self.steps)
            self.writer.add_scalar('stats/mean_Q', mean_q, 4*self.steps)
        

    def calculate_loss(self, states, state_embeddings, actions, rewards, next_states,
                       dones, weights, lamda  = 10):
        
        self.lamda = lamda
        taus = torch.rand(
            self.batch_size, self.N, dtype=state_embeddings.dtype,
            device=state_embeddings.device)

        # Calculate quantile values of current states and actions at tau_hats.
        current_sa_quantiles = evaluate_quantile_at_action(
            self.online_net.calculate_quantiles(
                taus, state_embeddings=state_embeddings), actions)
        

        with torch.no_grad():
            # Calculate Q values of next states.
            if self.double_q_learning:
                # Sample the noise of online network to decorrelate between
                # the action selection and the quantile calculation.
                self.online_net.sample_noise()
                next_q = self.online_net.calculate_q(states=next_states)
            else:
                next_state_embeddings =\
                    self.target_net.calculate_state_embeddings(next_states)
                next_q = self.target_net.calculate_q(
                    state_embeddings=next_state_embeddings)

            #greedy 
            next_actions =  torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate features of next states.
            if self.double_q_learning:
                next_state_embeddings =\
                    self.target_net.calculate_state_embeddings(next_states)

            # Sample next fractions.
            tau_dashes = torch.rand(
                self.batch_size, self.N_dash, dtype=state_embeddings.dtype,
                device=state_embeddings.device)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_net.calculate_quantiles(
                    tau_dashes, state_embeddings=next_state_embeddings
                ), next_actions).transpose(1, 2)
            #assert next_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)
        
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles


        target_sa_quantiles=target_sa_quantiles[:,torch.randperm(target_sa_quantiles.size(1))]
        current_sa_quantiles = current_sa_quantiles[:,torch.randperm(current_sa_quantiles.size(1))]
        assert current_sa_quantiles.shape == (self.batch_size, self.N, 1)
        assert target_sa_quantiles.shape == (self.batch_size, 1,self.N)
        

        current_sa_quantiles_d = self.discriminator(current_sa_quantiles, states, actions)
        target_sa_quantiles_d = self.discriminator(target_sa_quantiles,next_states, next_actions)
        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (self.batch_size, self.N, self.N_dash)

        for p in self.discriminator.parameters():
            p.requires_grad = True
        
        torch.autograd.set_detect_anomaly(True)
        #for i in range(self.n_critic):
        self.discriminator.zero_grad()
        GAN_loss = (current_sa_quantiles_d - target_sa_quantiles_d).mean()
        GAN_loss.backward(retain_graph=True)
        self.discriminator_optim.step() 
        
        
        for p in self.discriminator.parameters():
            p.requires_grad = False  # to avoid computation
        self.online_net.zero_grad()
        Q_loss = -1. * self.discriminator(current_sa_quantiles).mean()
        Q_loss.backward(retain_graph=True)
        self.generator_optim.step()
        return GAN_loss, td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)




