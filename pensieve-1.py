from dataclasses import dataclass
import numpy as np
import env  # this is actually a file env.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

@dataclass
class constants:
    S_INFO: int = 6  # bit_rate, buffer_size, next_chunk_size, throughput, time, chunks_left
    S_LEN: int = 8  # how many past chunks to consider
    VIDEO_BIT_RATE: tuple = (300, 750, 1200, 1850, 2850, 4300)  # in kbps
    A_DIM: int = len(VIDEO_BIT_RATE) 
    MODEL_SAVE_INTERVAL: int = 100
    HD_REWARD: tuple = (1, 2, 3, 12, 15, 20)
    BUFFER_NORM_FACTOR: float = 10.0  # DBT: what is this?
    CHUNK_TILL_VIDEO_END_CAP: float = 48.0  # DBT: should this not be an integer?
    CONV_INPUT_CHANNELS: int = 1
    NN_INPUT_FEATURES: int = 1


@dataclass
class hyperparameters:
    ACTOR_LR_RATE: float = 0.0001
    CRITIC_LR_RATE: float = 0.001
    NUM_AGENTS: int = 6
    TRAIN_SEQ_LEN: int = 100  # train batch size
    ADAM_EPSILON: float = 1e-8
    ADAM_WEIGHT_DECAY: float = 0
    ADAM_BETAS: tuple = (0.9, 0.99)
    NUM_HIDDEN_NEURONS: int = 128
    NUM_CONV_CHANNELS: int = 128
    CONV_KERNEL_SIZE: int = 4
    CONV_STRIDE: int = 1
    CONV_PADDING: int = 0
    NN_ACTIVATION_FUNCTION: int = F.ReLU
    DISCOUNT_FACTOR: float = 0.99
    INITIAL_ENTROPY_WEIGHT: float = 1.0
    FINAL_ENTROPY_WEIGHT: float = 0.1


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=hyperparameters.ACTOR_LR_RATE, betas=hyperparameters.ADAM_BETAS, eps=hyperparameters.ADAM_EPSILON, weight_decay=hyperparameters.ADAM_WEIGHT_DECAY):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # param_groups is a member of the optimizer base class from which Adam inherits
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self)__init__()
        self.bitrate = nn.Linear(constants.NN_INPUT_FEATURES, hyperparameters.NUM_HIDDEN_NEURONS)
        self.buffer_size = nn.Linear(constants.NN_INPUT_FEATURES, hyperparameters.NUM_HIDDEN_NEURONS)
        self.throughput = nn.Conv1d(constants.CONV_INPUT_CHANNELS, hyperparameters.NUM_CONV_CHANNELS, hyperparameters.CONV_KERNEL_SIZE, stride=hyperparameters.CONV_STRIDE, padding=hyperparameters.CONV_PADDING)
        self.download_time = nn.Conv1d(constants.CONV_INPUT_CHANNELS, hyperparameters.NUM_CONV_CHANNELS, hyperparameters.CONV_KERNEL_SIZE, stride=hyperparameters.CONV_STRIDE, padding=hyperparameters.CONV_PADDING)
        self.chunk_size = nn.Conv1d(constants.CONV_INPUT_CHANNELS, hyperparameters.NUM_CONV_CHANNELS, hyperparameters.CONV_KERNEL_SIZE, stride=hyperparameters.CONV_STRIDE, padding=hyperparameters.CONV_PADDING)
        self.chunks_remaining = nn.Linear(constants.NN_INPUT_FEATURES, hyperparameters.NUM_HIDDEN_NEURONS)

        flattened_input_size = 3 * hyperparameters.NUM_HIDDEN_NEURONS + 2 * hyperparameters.NUM_CONV_CHANNELS * math.floor((constants.S_LEN - hyperparameters.CONV_KERNEL_SIZE + 2 * hyperparameters.CONV_PADDING) / hyperparameters.CONV_STRIDE + 1) + hyperparameters.NUM_CONV_CHANNELS * math.floor((constants.A_DIM - hyperparameters.CONV_KERNEL_SIZE + 2 * hyperparameters.CONV_PADDING) / hyperparameters.CONV_STRIDE + 1)

        self.common_fc_layer = nn.Linear(flattened_input_size, hyperparameters.NUM_HIDDEN_NEURONS)
        self.actor_output_layer = nn.Linear(hyperparameters.NUM_HIDDEN_NEURONS, constants.A_DIM)
        self.critic_output_layer = nn.Linear(hyperparameters.NUM_HIDDEN_NEURONS, 1)

        self.states, self.actions, self.rewards = [], [], []

    def forward(self, state):
        bitrate = hyperparameters.NN_ACTIVATION_FUNCTION(self.bitrate(state[: , 0: 1, -1]))
        buffer_size = hyperparameters.NN_ACTIVATION_FUNCTION(self.buffer_size(state[: , 1: 2, -1]))
        throughput = hyperparameters.NN_ACTIVATION_FUNCTION(self.throughput(state[: , 2: 3, : ]))
        download_time = hyperparameters.NN_ACTIVATION_FUNCTION(self.download_time(state[: , 3: 4, : ]))
        chunk_size = hyperparameters.NN_ACTIVATION_FUNCTION(self.chunk_size(state[: , 4: 5, : constants.A_DIM]))
        chunks_remaining = hyperparameters.NN_ACTIVATION_FUNCTION(self.chunks_remaining(state[: , 5: 6, -1]))

        throughput_flat = torch.flatten(throughput, 1)  # flatten everything except batch
        download_time_flat = torch.flatten(download_time, 1)
        chunk_size_flat = torch.flatten(chunk_size, 1)
        flattened_input = torch.cat([bitrate, buffer_size, throughput_flat, download_time_flat, chunk_size_flat, chunks_remaining], dim=1)

        interim_outputs = hyperparameters.NN_ACTIVATION_FUNCTION(self.common_fc_layer(flattened_input))
        actor_output = F.softmax(self.actor_output_layer(interim_outputs))
        critic_output = self.critic_output_layer(interim_outputs)

        return actor_output, critic_output
        
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.reward.append(reward)

    def clear_memory(self):
        self.states, self.actions, self.rewards = [], [], []

    def calculate_return(self, done):
        states = torch.tensor(np.array(self.states), dtype=torch.float)
        _, value = self.forward(states)

        discounted_return = value[-1] * (1 - int(done)) # if it is the terminal step then return for that step is 0, else it is whatever it is

        batch_return = []
        for reward in self.rewards[::-1]:
            discounted_return = reward + hyperparameters.DISCOUNT_FACTOR * discounted_return
            batch_return.append(discounted_return)  # the return for the last step (calculated before for loop) is the value that is added first so we reverse in next step
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)
       
    def calculate_loss(self, done):
        states = torch.tensor(np.array(self.states), dtype=torch.float)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float)

        returns = self.calculate_return(done)

        policy, values = self.forward(states)
        values = values.squeeze()  # DBT: is this needed?
        critic_loss = (returns - values) ** 2  # this seems to be correct

        # actor loss is log(policy) * advantage + entropy
        distribution = Categorical(policy)
        log_probabilities = distribution.log_prob(actions)
        actor_loss = - log_probabilities * (returns - values)  # there is a negative sign because we want to do gradient ascent

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss


def set_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

if __name__ == '__main__':
    set_seeds()


