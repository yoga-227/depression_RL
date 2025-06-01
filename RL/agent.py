import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from RL.memory import Memory_Buffer
from model.BLSTM import LSTMModel, Noisy_LSTMModel, depression_LSTM, depression_CNN
from model.RCNN import RCNN


class DQNAgent:
    def __init__(self, action_space=[], USE_CUDA=False, memory_size=10000, epsilon=1, lr=1e-4):
        self.epsilon = epsilon
        self.action_space = action_space
        self.memory_buffer = Memory_Buffer(memory_size)
        self.DQN = depression_LSTM(num_classes=action_space.n)
        # self.DQN.load_state_dict(torch.load("../test/D3QN_imb4.pth"))
        self.DQN_target = depression_LSTM(num_classes=action_space.n)
        self.DQN_target.load_state_dict(self.DQN.state_dict())

        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
        # self.optimizer = optim.RMSprop(self.DQN.parameters(), lr=lr, eps=0.001, alpha=0.95)
        self.optimizer = optim.Adam(self.DQN.parameters(), lr=lr)

    def observe(self, lazyframe):
        if isinstance(lazyframe, tuple):
            lazyframe = np.array(lazyframe[0])
        state = torch.from_numpy(lazyframe.__array__()[None]).float()
        # state = torch.from_numpy(lazyframe._force().transpose(2, 0, 1)[None] / 255).float()
        if self.USE_CUDA:
            state = state.cuda()
        # state = state.unsqueeze(1)
        return state

    def value(self, state):
        q_values = self.DQN(state)
        return q_values

    def act(self, state, epsilon=None):
        """
        sample actions with epsilon-greedy policy
        recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
        """
        if epsilon is None: epsilon = self.epsilon

        q_values = self.value(state).cpu().detach().numpy()
        if random.random() < epsilon:
            action = random.randrange(self.action_space.n)
        else:
            action = np.argmax(q_values)
        return action, q_values

    # def act(self, state, epsilon=None):
    #     q_values = self.value(state).cpu().detach().numpy()
    #     action = np.argmax(q_values)
    #     return action, q_values

    # DQN
    # def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
    #     actions = torch.tensor(actions).long()
    #     rewards = torch.tensor(rewards, dtype= torch.float)
    #     is_done = torch.tensor(is_done).bool()
    #
    #     if self.USE_CUDA:
    #         actions = actions.cuda()
    #         rewards = rewards.cuda()
    #         is_done = is_done.cuda()
    #
    #     # get q-values for all actions in current states
    #     predicted_q_values = self.DQN(states)
    #
    #     # select q-values for chosen actions
    #     predicted_q_values_for_actions = predicted_q_values[range(states.shape[0]), actions]
    #
    #     # compute q-values for all actions in next states
    #     predicted_next_q_values = self.DQN_target(next_states)
    #
    #     # compute V*(next_states) using predicted next q-values
    #     next_state_values = predicted_next_q_values.max(-1)[0]
    #
    #     # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    #     target_q_values_for_actions = rewards + gamma * next_state_values
    #
    #     # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    #     target_q_values_for_actions = torch.where(is_done, rewards, target_q_values_for_actions)
    #
    #     # mean squared error loss to minimize
    #     loss = F.smooth_l1_loss(predicted_q_values_for_actions, target_q_values_for_actions.detach())
    #
    #     return loss

    # DDQN
    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards, dtype=torch.float)
        is_done = torch.tensor(is_done).bool()

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        # get q_values for all action in current states
        predicted_q_values = self.DQN(states)

        # select q_values for chosen actions
        predicted_q_values_for_actions = predicted_q_values[range(states.shape[0]), actions]

        # compute q_values for all actions in next states
        predicted_next_q_values_current = self.DQN(next_states)
        predicted_next_q_values_target = self.DQN_target(next_states)

        # compute V*(next_states) using predicted next q_values
        next_state_value = predicted_next_q_values_target.gather(1, torch.max(predicted_next_q_values_current, 1)[
            1].unsqueeze(1)).squeeze(1)

        # compute "target q_values" for loss
        target_q_values_for_actions = rewards + gamma * next_state_value

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_q_values_for_actions = torch.where(is_done, rewards, target_q_values_for_actions)

        # mean squared error loss to minimize
        loss = F.smooth_l1_loss(predicted_q_values_for_actions, target_q_values_for_actions)

        return loss

    # 正常采样
    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            state, action, reward, next_state, done = data
            states.append(self.observe(state))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_state))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def learn_from_experience(self, batch_size, gamma):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones, gamma)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.DQN.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return td_loss.item()
        else:
            return 0
