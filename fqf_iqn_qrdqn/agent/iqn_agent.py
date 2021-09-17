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








class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

        
    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        
        validity = self.model(img_flat)
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

        self.discriminator = Discriminator().to(self.device)

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

        print(self.memory.sample)
        # Calculate features of states.
        state_embeddings = self.online_net.calculate_state_embeddings(states)

        quantile_loss, mean_q = self.calculate_loss(
            state_embeddings)
       
        update_params(
            self.optim, quantile_loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

       
        if 4*self.steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                4*self.steps)
            self.writer.add_scalar('stats/mean_Q', mean_q, 4*self.steps)

    def calculate_loss(self, state_embeddings, lamda = 10):

        
        
        self.lamda = lamda

        for _ in range(self.n_critic):
            states, actions, rewards, next_states, _ =\
                self.memory.sample(self.batch_size)
            #next_reward = self.exploit(next_states.to('cpu'))
            epsilon = torch.FloatTensor(1, self.batch_size).uniform_(0, 1)
            #z = torch.normal(0,1,size=(1,self.batch_size))
            #z_1 = torch.normal(0,1,size=(1,self.batch_size))
            z = z_1 = taus = torch.rand(
                self.batch_size, self.N, dtype=state_embeddings.dtype,
                device=state_embeddings.device)
            x=self.online_net.calculate_quantiles(z,states)
            print(x.shape)
            x_1 = rewards + self.gamma * self.online_net.calculate_quantiles(z_1,next_states)
            interpolated = epsilon * x + (1-epsilon)*x_1
            
            interpolated = Variable(interpolated, requires_grad=True).to(self.device)
            prob_interpolated = self.discriminator(interpolated)

            
            gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]


            GAN_loss = self.discriminator(x) - self.discriminator(x_1) + self.lamda * ((gradients.norm(2, dim=1) - 1) ** 2).mean() 

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


            adam = optim.Adam(self.discriminator.parameters(), lr=self.lr) 
            adam.step() 
            return GAN_loss,next_q.detach().mean().item(), \
            td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)



