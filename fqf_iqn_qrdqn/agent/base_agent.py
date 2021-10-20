from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from fqf_iqn_qrdqn.env import make_pytorch_env

from fqf_iqn_qrdqn.memory import LazyMultiStepMemory, \
    LazyPrioritizedMultiStepMemory
from fqf_iqn_qrdqn.utils import RunningMeanStats, LinearAnneaer


class BaseAgent(ABC):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000, save_interval = 50000,
                 max_episode_steps=27000, grad_cliping=5.0, cuda=True, seed=0, 
                 agent=None, env_online=None):

        self.env = env
        self.env_online = env_online
        self.test_env = test_env
        print("env online:", env_online)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        if self.env_online:
            self.env_online.seed(seed)
        self.test_env.seed(2**31-1-seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False     # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.online_net = None
        self.target_net = None

        # Replay memory which is memory-efficient to store stacked frames.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.device, gamma, multi_step, beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)
        self.train_return_online = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.double_q_learning = double_q_learning
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        self.use_per = use_per

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.save_interval = save_interval
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.epsilon_train = LinearAnneaer(
            1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping
        self.agent = agent
        self.min_steps = 0

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def is_random(self, eval=False):
        # Use e-greedy for evaluation.
        if self.steps < self.start_steps:
            return True
        if eval:
            return np.random.rand() < self.epsilon_eval
        if self.noisy_net:
            return False
        return np.random.rand() < self.epsilon_train.get()

    def update_target(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict())

    def explore(self):
        # Act with randomness.
        action = self.env.action_space.sample()
        return action

    def exploit(self, state, online=False):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.

        if online:   # added
            return self.online_net.calculate_q(states=state).argmax().item()

        with torch.no_grad():
            if self.agent != None:
                action = self.agent.online_net.calculate_q(states=state).argmax().item()
            else:
                action = self.online_net.calculate_q(states=state).argmax().item()
        return action

    @abstractmethod
    def learn(self):
        pass

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.online_net.state_dict(),
            os.path.join(save_dir, 'online_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))

    def load_models(self, save_dir):
        self.online_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'online_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))

    def train_episode(self):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0
        episode_return_online = 0

        done = False
        state = self.env.reset()
        state_online = self.env_online.reset()

        # while (not done) and episode_steps <= self.max_episode_steps:
        while (not done) and self.steps <= self.start_steps:
            # fix net
            action = self.exploit(state)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            # added: online next
            # self.online_net.sample_noise()
            # action_online = self.exploit(state_online, online=True)
            # next_state_online, reward_online, done_online, _online = self.env_online.step(action_online)
            # episode_return_online += reward_online
            # state_online = next_state_online
            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

        if self.steps <= self.start_steps:
            self.train_return.append(episode_return)
            self.train_return_online.append(episode_return_online)

            # We log evaluation results along with training frames = 4 * steps.
            # if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'return/train-fixed', self.train_return.get(), 4 * self.steps)
            self.writer.add_scalar(
                'return/train-online', self.train_return_online.get(), 4 * self.steps)

            print(f'Episode: {self.episodes:<4}  '
                  f'episode steps: {episode_steps:<4}  '
                  f'return: {episode_return:<5.1f}',
                  f'return_online: {episode_return_online:<5.1f}')

        if self.steps > self.start_steps and self.min_steps == 0:
            self.min_steps = self.steps
            print(f"sample period is done: {self.min_steps} samples")

        if self.steps > self.start_steps:
            # print("train", end=",")
            self.train_step_interval()
            self.steps += 1


    def train_step_interval(self):
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        if self.is_update():
            self.learn()

        # if self.steps % self.eval_interval == 0:
        if self.steps % 5000 == 0:
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, 'final'))
            self.online_net.train()

    def evaluate(self):
        print("start evaluate")
        self.online_net.eval()
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                if self.is_random(eval=True):
                    action = self.explore()
                else:
                    action = self.exploit(state, online=True)

                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            # if num_steps > self.num_eval_steps:
            #     break
            if num_steps > 6000:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        # We log evaluation results along with training frames = 4 * steps.
        self.writer.add_scalar(
            'return/test-online', mean_return, 4 * self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()