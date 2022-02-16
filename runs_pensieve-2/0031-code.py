from dataclasses import dataclass
import numpy as np
import env  # this is actually a file env.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
from torch.optim import RMSprop
import math
import load_trace  # this is a file load_trace.py
import random
import matplotlib.pyplot as plt
import os
import time 
import shutil


@dataclass
class cons:
    S_INFO: int = 6  # bit_rate, buffer_size, next_chunk_size, throughput, time, chunks_left
    S_LEN: int = 8  # how many past chunks to consider
    VID_BITRATES: tuple = (300, 750, 1200, 1850, 2850, 4300)  # in kbps
    A_DIM: int = len(VID_BITRATES) 
    HD_REWARD: tuple = (1, 2, 3, 12, 15, 20)
    BUFFER_NORM_FACTOR: float = 10.0  # DBT: what is this?
    CHUNK_TILL_VIDEO_END_CAP: float = 48.0  # DBT: should this not be an integer?
    DEFAULT_QUALITY: int = 1
    CONV_IN_CHANNELS: int = 1
    NN_IN_FEATURES: int = 1


@dataclass
class hyp:  # changing these will theoretically impact the performace of the model
    NUM_EPOCHS: int = 20
    ACTOR_LR: float = 0.0001
    CRITIC_LR: float = 0.001
    NUM_AGENTS: int = 4
    TRAIN_SEQ_LEN: int = 100  # train batch size
    RMSPROP_ALPHA: float = 0.9
    RMSPROP_EPS: float = 1e-10
    NUM_HID_NEURONS: int = 128
    NUM_CONV_CHANNELS: int = 128
    CONV_KERNEL_SIZE: int = 4
    CONV_STRIDE: int = 1
    CONV_PADDING: int = 0
    NN_ACT_FN: int = F.relu
    DISCOUNT_FACTOR: float = 0.99
    INIT_ENTROPY_WT: float = 1.0
    FINAL_ENTROPY_WT: float = 0.1
    REBUF_PENALTY: float = 4.3
    SMOOTH_PENALTY: float = 1.0

@dataclass
class par:  # variables that are not directly related to the model and its performance
    PRINT_EVERY: int = 100
    DATA_PATH: str = r'/mnt/windows_d/Svalbard/Data/Thesis/pensieve_data/cooked_traces/'
    TIMING_DATA: tuple = tuple(load_trace.load_trace(DATA_PATH)[0])
    THROUGHPUT_DATA: tuple = tuple(load_trace.load_trace(DATA_PATH)[1])
    SMOOTHING_AMT: int = hyp.NUM_EPOCHS // 10
    SAVE_DIR: str = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-2/'
    save_num: int = len(set([name[:4] for name in os.listdir(SAVE_DIR)])) + 1
    SAVE_NUM: str = '0' * (4 - len(str(save_num))) + str(save_num)
    CURR_FILE_PATH: str = '/mnt/windows_d/LibraryOfBabel/Projects/A3C/pensieve-2.py'

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.bitrate = nn.Linear(cons.NN_IN_FEATURES, hyp.NUM_HID_NEURONS)
        self.buffer_size = nn.Linear(cons.NN_IN_FEATURES, hyp.NUM_HID_NEURONS)
        self.throughput = nn.Conv1d(cons.CONV_IN_CHANNELS, hyp.NUM_CONV_CHANNELS, hyp.CONV_KERNEL_SIZE, stride=hyp.CONV_STRIDE, padding=hyp.CONV_PADDING)
        self.download_time = nn.Conv1d(cons.CONV_IN_CHANNELS, hyp.NUM_CONV_CHANNELS, hyp.CONV_KERNEL_SIZE, stride=hyp.CONV_STRIDE, padding=hyp.CONV_PADDING)
        self.chunk_size = nn.Conv1d(cons.CONV_IN_CHANNELS, hyp.NUM_CONV_CHANNELS, hyp.CONV_KERNEL_SIZE, stride=hyp.CONV_STRIDE, padding=hyp.CONV_PADDING)
        self.chunks_remaining = nn.Linear(cons.NN_IN_FEATURES, hyp.NUM_HID_NEURONS)

        lin_out_flat = 3 * hyp.NUM_HID_NEURONS
        down_thru_out_flat = 2 * hyp.NUM_CONV_CHANNELS * math.floor((cons.S_LEN - hyp.CONV_KERNEL_SIZE + 2 * hyp.CONV_PADDING) / hyp.CONV_STRIDE + 1)
        chunk_size_out_flat = hyp.NUM_CONV_CHANNELS * math.floor((cons.A_DIM - hyp.CONV_KERNEL_SIZE + 2 * hyp.CONV_PADDING) / hyp.CONV_STRIDE + 1)
        flattened_input_size = lin_out_flat + down_thru_out_flat + chunk_size_out_flat 

        self.actor_layer_1 = nn.Linear(flattened_input_size, hyp.NUM_HID_NEURONS)
        self.actor_layer_2 = nn.Linear(hyp.NUM_HID_NEURONS, cons.A_DIM)

    def forward(self, state):
        bitrate = hyp.NN_ACT_FN(self.bitrate(state[: , 0: 1, -1]))
        buffer_size = hyp.NN_ACT_FN(self.buffer_size(state[: , 1: 2, -1]))
        throughput = hyp.NN_ACT_FN(self.throughput(state[: , 2: 3, : ]))
        download_time = hyp.NN_ACT_FN(self.download_time(state[: , 3: 4, : ]))
        chunk_size = hyp.NN_ACT_FN(self.chunk_size(state[: , 4: 5, : cons.A_DIM]))
        chunks_remaining = hyp.NN_ACT_FN(self.chunks_remaining(state[: , 5: 6, -1]))

        throughput_flat = torch.flatten(throughput, 1)  # flatten everything except batch
        download_time_flat = torch.flatten(download_time, 1)
        chunk_size_flat = torch.flatten(chunk_size, 1)
        flattened_input = torch.cat([bitrate, buffer_size, throughput_flat, download_time_flat, chunk_size_flat, chunks_remaining], dim=1)

        interim_outputs = hyp.NN_ACT_FN(self.actor_layer_1(flattened_input))
        action_probs = F.softmax(self.actor_layer_2(interim_outputs), dim=1)

        return action_probs
        

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.bitrate = nn.Linear(cons.NN_IN_FEATURES, hyp.NUM_HID_NEURONS)
        self.buffer_size = nn.Linear(cons.NN_IN_FEATURES, hyp.NUM_HID_NEURONS)
        self.throughput = nn.Conv1d(cons.CONV_IN_CHANNELS, hyp.NUM_CONV_CHANNELS, hyp.CONV_KERNEL_SIZE, stride=hyp.CONV_STRIDE, padding=hyp.CONV_PADDING)
        self.download_time = nn.Conv1d(cons.CONV_IN_CHANNELS, hyp.NUM_CONV_CHANNELS, hyp.CONV_KERNEL_SIZE, stride=hyp.CONV_STRIDE, padding=hyp.CONV_PADDING)
        self.chunk_size = nn.Conv1d(cons.CONV_IN_CHANNELS, hyp.NUM_CONV_CHANNELS, hyp.CONV_KERNEL_SIZE, stride=hyp.CONV_STRIDE, padding=hyp.CONV_PADDING)
        self.chunks_remaining = nn.Linear(cons.NN_IN_FEATURES, hyp.NUM_HID_NEURONS)

        lin_out_flat = 3 * hyp.NUM_HID_NEURONS
        down_thru_out_flat = 2 * hyp.NUM_CONV_CHANNELS * math.floor((cons.S_LEN - hyp.CONV_KERNEL_SIZE + 2 * hyp.CONV_PADDING) / hyp.CONV_STRIDE + 1)
        chunk_size_out_flat = hyp.NUM_CONV_CHANNELS * math.floor((cons.A_DIM - hyp.CONV_KERNEL_SIZE + 2 * hyp.CONV_PADDING) / hyp.CONV_STRIDE + 1)
        flattened_input_size = lin_out_flat + down_thru_out_flat + chunk_size_out_flat 

        self.critic_layer_1 = nn.Linear(flattened_input_size, hyp.NUM_HID_NEURONS)
        self.critic_layer_2 = nn.Linear(hyp.NUM_HID_NEURONS, 1)

    def forward(self, state):
        bitrate = hyp.NN_ACT_FN(self.bitrate(state[: , 0: 1, -1]))
        buffer_size = hyp.NN_ACT_FN(self.buffer_size(state[: , 1: 2, -1]))
        throughput = hyp.NN_ACT_FN(self.throughput(state[: , 2: 3, : ]))
        download_time = hyp.NN_ACT_FN(self.download_time(state[: , 3: 4, : ]))
        chunk_size = hyp.NN_ACT_FN(self.chunk_size(state[: , 4: 5, : cons.A_DIM]))
        chunks_remaining = hyp.NN_ACT_FN(self.chunks_remaining(state[: , 5: 6, -1]))

        throughput_flat = torch.flatten(throughput, 1)  # flatten everything except batch
        download_time_flat = torch.flatten(download_time, 1)
        chunk_size_flat = torch.flatten(chunk_size, 1)
        flattened_input = torch.cat([bitrate, buffer_size, throughput_flat, download_time_flat, chunk_size_flat, chunks_remaining], dim=1)

        interim_outputs = hyp.NN_ACT_FN(self.critic_layer_1(flattened_input))
        values = self.critic_layer_2(interim_outputs)

        return values

def set_seeds():
    torch.backends.cudnn.deterministic = True
    # random.seed(hash("setting random seeds") % 2**32 - 1)
    # np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    # torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    # torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
    random.seed(hash(time.time()) % 2 ** 32 - 1)

def central_agent(parameter_queues, experience_queues):
    torch.set_num_threads(1)
    actor = Actor()
    actor_criterion = RMSprop(actor.parameters(), lr=hyp.ACTOR_LR, alpha=hyp.RMSPROP_ALPHA, eps=hyp.RMSPROP_EPS)
    actor_criterion.zero_grad()
    critic = Critic()
    critic_criterion = RMSprop(critic.parameters(), lr=hyp.CRITIC_LR, alpha=hyp.RMSPROP_ALPHA, eps=hyp.RMSPROP_EPS)
    critic_criterion.zero_grad()
    critic_loss_fn = nn.MSELoss()

    for epoch in range(hyp.NUM_EPOCHS):

        for i in range(hyp.NUM_AGENTS):
            actor_parameters = actor.parameters()
            parameter_queues[i].put(list(actor_parameters))

            # DBT: to use CUDA with multiprocessing you need to use the spawn method which I have no clue about
            states, actions, rewards, terminal = experience_queues[i].get()
            # DBT: for some reason the pensieve_torch uses torch.cat for states
            states = torch.cat(states)
            # DBT: why use a LongTensor
            actions = torch.LongTensor(actions)  # LongTensor is used for 64 bit signed integers
            rewards = torch.tensor(np.array(rewards))
            returns = torch.zeros(rewards.shape)

            returns[-1] = rewards[-1]
            for t in reversed(range(rewards.shape[0] - 1)):
                returns[t] = rewards[t] + hyp.DISCOUNT_FACTOR * returns[t + 1]
            with torch.no_grad():
                values = critic.forward(states).squeeze()
                advantage = returns - values
            
            curr_entropy_weight = hyp.INIT_ENTROPY_WT - (epoch + 1) / float(hyp.NUM_EPOCHS) * (hyp.FINAL_ENTROPY_WT - hyp.INIT_ENTROPY_WT)
            probs = actor.forward(states)
            dist = Categorical(probs)
            actor_loss = torch.sum(dist.log_prob(actions) * (- advantage)) + curr_entropy_weight * torch.sum(dist.entropy()) 
            actor_loss.backward()

            critic_loss = critic_loss_fn(returns, critic.forward(states).squeeze())
            critic_loss.backward()

        actor_criterion.step()
        actor_criterion.zero_grad()
        critic_criterion.step()
        critic_criterion.zero_grad()

def agent(i, agent_parameter_queue, agent_experience_queue):
    torch.set_num_threads(1)
    actor = Actor()

    environ = env.Environment(par.TIMING_DATA, par.THROUGHPUT_DATA, random_seed=hash(str(time.time()) + str(i)) & 2 ** 32 - 1)
    scores = []
    for epoch in range(hyp.NUM_EPOCHS):
        global_actor_params = agent_parameter_queue.get()
        for target, source in zip(actor.parameters(), global_actor_params):
            target.data.copy_(source.data)

        prev_bitrate_idx = cons.DEFAULT_QUALITY
        curr_bitrate_idx = cons.DEFAULT_QUALITY

        states, actions, rewards = [], [], []
        state = torch.zeros((1, cons.S_INFO, cons.S_LEN))

        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = environ.get_video_chunk(curr_bitrate_idx)
        if epoch == 10 and check_print == 0:
            print(delay, sleep_time, buffer_size, rebuf, video_chunk_size, video_chunk_remain)
            check_print = 1

        score = 0
        check_print = 0
        while not end_of_video and len(states) < hyp.TRAIN_SEQ_LEN:
            prev_bitrate_idx = curr_bitrate_idx

            state = state.clone().detach()
            state = torch.roll(state, shifts=-1,dims=-1)
            state[0,0, -1] = cons.VID_BITRATES[curr_bitrate_idx] / float(np.max(cons.VID_BITRATES))
            state[0,1, -1] = buffer_size / cons.BUFFER_NORM_FACTOR
            state[0,2, -1] = float(video_chunk_size) / float(delay) / 1000
            state[0,3, -1] = float(delay) / 1000 / cons.BUFFER_NORM_FACTOR
            state[0,4, :cons.A_DIM] = torch.tensor(next_video_chunk_sizes) / pow(10, 6)
            state[0,5, -1] = min(video_chunk_remain, cons.CHUNK_TILL_VIDEO_END_CAP) / float(cons.CHUNK_TILL_VIDEO_END_CAP)

            if epoch == 10 and check_print == 1:
                print(state)
                check_print = 2
            with torch.no_grad():
                probs = actor.forward(state)
                dist = Categorical(probs)
                curr_bitrate_idx = dist.sample().item()

            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = environ.get_video_chunk(curr_bitrate_idx)

            bitrate_reward = cons.VID_BITRATES[curr_bitrate_idx] / 1000.0
            rebuf_penalty = hyp.REBUF_PENALTY * rebuf
            smooth_penalty = hyp.SMOOTH_PENALTY * np.abs(cons.VID_BITRATES[curr_bitrate_idx] - cons.VID_BITRATES[prev_bitrate_idx]) / 1000.0
            reward = bitrate_reward - rebuf_penalty - smooth_penalty
            score += reward

            states.append(state)
            actions.append(curr_bitrate_idx)
            rewards.append(reward)

        agent_experience_queue.put([states, actions, rewards, end_of_video])
        scores.append(score)
        if (epoch + 1) % par.PRINT_EVERY == 0:
            print('epoch', epoch+1)
    scores = moving_average(scores, n=par.SMOOTHING_AMT)
    plot(str(i), scores)

def moving_average(a, n=par.SMOOTHING_AMT) :
    ret = np.cumsum(np.array(a), dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot(name, scores):
    ylim_path = par.SAVE_DIR + par.SAVE_NUM + '-ylim_worker' + name + '.png'
    full_path = par.SAVE_DIR + par.SAVE_NUM + '-full-worker' + name + '.png'
    plt.plot([i for i in range(1, len(scores) + 1)], scores)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.title('worker ' + name)
    plt.savefig(full_path, bbox_inches='tight')
    if max(scores) >= 0:
        plt.ylim([-50, 50])
        plt.savefig(ylim_path, bbox_inches='tight')

def save_state(time_taken):
    file_copy_path = par.SAVE_DIR + par.SAVE_NUM + '-code.py'
    shutil.copyfile(par.CURR_FILE_PATH, file_copy_path)
    with open(par.SAVE_DIR + par.SAVE_NUM + '-hyp_cons_par.csv', 'w') as f:
        f.write('name,value\n')
        [f.write(name + ',' + str(value) + '\n') for name, value in vars(hyp).items() if not name.startswith('__')]
        [f.write(name + ',' + str(value) + '\n') for name, value in vars(cons).items() if not name.startswith('__')]
        [f.write(name + ',' + str(value) + '\n') for name, value in vars(par).items() if not name.startswith('__') and not name.endswith('_DATA')]
        f.write('time_taken,' + str(time_taken))

if __name__ == '__main__':
    # set_seeds()
    start_time = time.perf_counter()
    parameter_queues = [mp.Queue(1) for i in range(hyp.NUM_AGENTS)]
    experience_queues = [mp.Queue(1) for i in range(hyp.NUM_AGENTS)]
    
    coordinator = mp.Process(target=central_agent, args=(parameter_queues, experience_queues))
    coordinator.start()

    agents = [mp.Process(target=agent, args=(i, parameter_queues[i], experience_queues[i])) for i in range(hyp.NUM_AGENTS)]
    for process in agents:
        process.start()

    coordinator.join()
    for process in agents:
        process.join()

    time_taken = time.perf_counter() - start_time
    save_state(time_taken)
