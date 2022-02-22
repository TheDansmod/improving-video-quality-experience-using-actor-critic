# This file does not use separate sets of processes for the protagonist and the adversary but does both in the same process. Have also added profiling and torch.set_num_threads(1)
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
import psutil
import datetime
from pathlib import Path
import cProfile
import pstats
from pstats import SortKey


NUM_EPOCHS = 100000 
NUM_AGENTS = 4
LOAD_MODEL = False
ADV_ACTOR_PATH = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0005-adv-chkpt-actor-10000'
QOE_METRIC = 'lin' # other options are 'log' and 'hd'
USE_GPU = False
DO_PROFILING = False
SET_SEEDS = False
MONITOR_PC_STATS = True
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, throughput, time, chunks_left
S_LEN = 8  # how many past chunks to consider
VID_BITRATES = (300, 750, 1200, 1850, 2850, 4300)  # in kbps
A_DIM = len(VID_BITRATES) 
HD_REWARD = (1, 2, 3, 12, 15, 20)
BUFFER_NORM_FACTOR = 10.0  # DBT: what is this?
CHUNK_TILL_VIDEO_END_CAP = 48.0  # DBT: should this not be an integer?
DEFAULT_QUALITY = 1
CONV_IN_CHANNELS = 1
NN_IN_FEATURES = 1
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
TRAIN_SEQ_LEN = 100  # train batch size
RMSPROP_ALPHA = 0.9
RMSPROP_EPS = 1e-10
NUM_HID_NEURONS = 128
NUM_CONV_CHANNELS = 128
CONV_KERNEL_SIZE = 4
CONV_STRIDE = 1
CONV_PADDING = 0
NN_ACT_FN = F.relu
DISCOUNT_FACTOR = 0.99
INIT_ENTROPY_WT = 1.0
FINAL_ENTROPY_WT = 0.1
LIN_REBUF_PENALTY = 4.3
LOG_REBUF_PENALTY = 2.66
HD_REBUF_PENALTY = 8.0
SMOOTH_PENALTY = 1.0
DATA_PATH = r'/mnt/windows_d/Svalbard/Data/Thesis/pensieve_data/cooked_traces/'
TIMING_DATA = tuple(load_trace.load_trace(DATA_PATH)[0])
THROUGHPUT_DATA = tuple(load_trace.load_trace(DATA_PATH)[1])
INIT_DIFF_LVL = 0.001
FINAL_DIFF_LVL = 0.1
OFFSET = pow(10, 5)  # this is added to the pro reward to ensure it is always positive so that the protagonist and adversary are optimizing the right objective. I have put this large value since I have seen min rewards of -27k
SAVE_DIR = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_' + Path(os.path.basename(__file__)).stem + '/'
save_num = len(set([name[:4] for name in os.listdir(SAVE_DIR)])) + 1 if Path(SAVE_DIR).exists() else 1
SAVE_NUM = '0' * (4 - len(str(save_num))) + str(save_num)
RAW_SCORE_FILE = SAVE_DIR + SAVE_NUM + '-raw-score'
RAW_ACTIONS_FILE = SAVE_DIR + SAVE_NUM + '-raw-actions'
CURR_FILE_PATH = Path(__file__).absolute()
PRO_ACTOR_PATH = ADV_ACTOR_PATH.replace('-adv-', '-pro-')
ADV_CRITIC_PATH = ADV_ACTOR_PATH.replace('-actor-', '-critic-')
PRO_CRITIC_PATH = PRO_ACTOR_PATH.replace('-actor-', '-critic-')
EPOCH_START = int(PRO_ACTOR_PATH.split('-')[-1]) if LOAD_MODEL else 0
PRINT_EVERY = NUM_EPOCHS // 10
SMOOTHING_AMT = NUM_EPOCHS // 10
MODEL_SAVE_EVERY = NUM_EPOCHS // 5
DEVICE = 'cuda:0' if (USE_GPU and torch.cuda.is_available()) else 'cpu'
SECS_PER_EPOCH = 0.24


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.bitrate = nn.Linear(NN_IN_FEATURES, NUM_HID_NEURONS)
        self.buffer_size = nn.Linear(NN_IN_FEATURES, NUM_HID_NEURONS)
        self.throughput = nn.Conv1d(CONV_IN_CHANNELS, NUM_CONV_CHANNELS, CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.download_time = nn.Conv1d(CONV_IN_CHANNELS, NUM_CONV_CHANNELS, CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.chunk_size = nn.Conv1d(CONV_IN_CHANNELS, NUM_CONV_CHANNELS, CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.chunks_remaining = nn.Linear(NN_IN_FEATURES, NUM_HID_NEURONS)

        lin_out_flat = 3 * NUM_HID_NEURONS
        down_thru_out_flat = 2 * NUM_CONV_CHANNELS * math.floor((S_LEN - CONV_KERNEL_SIZE + 2 * CONV_PADDING) / CONV_STRIDE + 1)
        chunk_size_out_flat = NUM_CONV_CHANNELS * math.floor((A_DIM - CONV_KERNEL_SIZE + 2 * CONV_PADDING) / CONV_STRIDE + 1)
        flattened_input_size = lin_out_flat + down_thru_out_flat + chunk_size_out_flat 

        self.actor_layer_1 = nn.Linear(flattened_input_size, NUM_HID_NEURONS)
        self.actor_layer_2 = nn.Linear(NUM_HID_NEURONS, A_DIM)

        nn.init.xavier_uniform_(self.bitrate.weight.data)
        nn.init.constant_(self.bitrate.bias.data, 0.0)
        nn.init.xavier_uniform_(self.buffer_size.weight.data)
        nn.init.constant_(self.buffer_size.bias.data, 0.0)
        nn.init.xavier_uniform_(self.throughput.weight.data)
        nn.init.constant_(self.throughput.bias.data, 0.0)
        nn.init.xavier_uniform_(self.download_time.weight.data)
        nn.init.constant_(self.download_time.bias.data, 0.0)
        nn.init.xavier_uniform_(self.chunk_size.weight.data)
        nn.init.constant_(self.chunk_size.bias.data, 0.0)
        nn.init.xavier_uniform_(self.chunks_remaining.weight.data)
        nn.init.constant_(self.chunks_remaining.bias.data, 0.0)
        nn.init.xavier_normal_(self.actor_layer_1.weight.data)
        nn.init.constant_(self.actor_layer_1.bias.data, 0.0)

    def forward(self, state):
        bitrate = NN_ACT_FN(self.bitrate(state[: , 0: 1, -1]))
        buffer_size = NN_ACT_FN(self.buffer_size(state[: , 1: 2, -1]))
        throughput = NN_ACT_FN(self.throughput(state[: , 2: 3, : ]))
        download_time = NN_ACT_FN(self.download_time(state[: , 3: 4, : ]))
        chunk_size = NN_ACT_FN(self.chunk_size(state[: , 4: 5, : A_DIM]))
        chunks_remaining = NN_ACT_FN(self.chunks_remaining(state[: , 5: 6, -1]))

        throughput_flat = torch.flatten(throughput, 1)  # flatten everything except batch
        download_time_flat = torch.flatten(download_time, 1)
        chunk_size_flat = torch.flatten(chunk_size, 1)
        flattened_input = torch.cat([bitrate, buffer_size, throughput_flat, download_time_flat, chunk_size_flat, chunks_remaining], dim=1)

        interim_outputs = NN_ACT_FN(self.actor_layer_1(flattened_input))
        action_probs = F.softmax(self.actor_layer_2(interim_outputs), dim=1)

        return action_probs
        

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.bitrate = nn.Linear(NN_IN_FEATURES, NUM_HID_NEURONS)
        self.buffer_size = nn.Linear(NN_IN_FEATURES, NUM_HID_NEURONS)
        self.throughput = nn.Conv1d(CONV_IN_CHANNELS, NUM_CONV_CHANNELS, CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.download_time = nn.Conv1d(CONV_IN_CHANNELS, NUM_CONV_CHANNELS, CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.chunk_size = nn.Conv1d(CONV_IN_CHANNELS, NUM_CONV_CHANNELS, CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.chunks_remaining = nn.Linear(NN_IN_FEATURES, NUM_HID_NEURONS)

        lin_out_flat = 3 * NUM_HID_NEURONS
        down_thru_out_flat = 2 * NUM_CONV_CHANNELS * math.floor((S_LEN - CONV_KERNEL_SIZE + 2 * CONV_PADDING) / CONV_STRIDE + 1)
        chunk_size_out_flat = NUM_CONV_CHANNELS * math.floor((A_DIM - CONV_KERNEL_SIZE + 2 * CONV_PADDING) / CONV_STRIDE + 1)
        flattened_input_size = lin_out_flat + down_thru_out_flat + chunk_size_out_flat 

        self.critic_layer_1 = nn.Linear(flattened_input_size, NUM_HID_NEURONS)
        self.critic_layer_2 = nn.Linear(NUM_HID_NEURONS, 1)

        nn.init.xavier_uniform_(self.bitrate.weight.data)
        nn.init.constant_(self.bitrate.bias.data, 0.0)
        nn.init.xavier_uniform_(self.buffer_size.weight.data)
        nn.init.constant_(self.buffer_size.bias.data, 0.0)
        nn.init.xavier_uniform_(self.throughput.weight.data)
        nn.init.constant_(self.throughput.bias.data, 0.0)
        nn.init.xavier_uniform_(self.download_time.weight.data)
        nn.init.constant_(self.download_time.bias.data, 0.0)
        nn.init.xavier_uniform_(self.chunk_size.weight.data)
        nn.init.constant_(self.chunk_size.bias.data, 0.0)
        nn.init.xavier_uniform_(self.chunks_remaining.weight.data)
        nn.init.constant_(self.chunks_remaining.bias.data, 0.0)
        nn.init.xavier_normal_(self.critic_layer_1.weight.data)
        nn.init.constant_(self.critic_layer_1.bias.data, 0.0)

    def forward(self, state):
        bitrate = NN_ACT_FN(self.bitrate(state[: , 0: 1, -1]))
        buffer_size = NN_ACT_FN(self.buffer_size(state[: , 1: 2, -1]))
        throughput = NN_ACT_FN(self.throughput(state[: , 2: 3, : ]))
        download_time = NN_ACT_FN(self.download_time(state[: , 3: 4, : ]))
        chunk_size = NN_ACT_FN(self.chunk_size(state[: , 4: 5, : A_DIM]))
        chunks_remaining = NN_ACT_FN(self.chunks_remaining(state[: , 5: 6, -1]))

        throughput_flat = torch.flatten(throughput, 1)  # flatten everything except batch
        download_time_flat = torch.flatten(download_time, 1)
        chunk_size_flat = torch.flatten(chunk_size, 1)
        flattened_input = torch.cat([bitrate, buffer_size, throughput_flat, download_time_flat, chunk_size_flat, chunks_remaining], dim=1)

        interim_outputs = NN_ACT_FN(self.critic_layer_1(flattened_input))
        values = self.critic_layer_2(interim_outputs)

        return values

def set_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

def central_agent(pro_par_q, pro_exp_q, adv_par_q, adv_exp_q):
    torch.set_num_threads(1)
    pro_actor = torch.load(PRO_ACTOR_PATH) if LOAD_MODEL else Actor()
    pro_actor_criterion = RMSprop(pro_actor.parameters(), lr=ACTOR_LR, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)
    pro_actor_criterion.zero_grad()
    pro_critic = torch.load(PRO_CRITIC_PATH) if LOAD_MODEL else Critic()
    pro_critic_criterion = RMSprop(pro_critic.parameters(), lr=CRITIC_LR, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)
    pro_critic_criterion.zero_grad()
    pro_critic_loss_fn = nn.MSELoss()

    adv_actor = torch.load(ADV_ACTOR_PATH) if LOAD_MODEL else Actor()
    adv_actor_criterion = RMSprop(adv_actor.parameters(), lr=ACTOR_LR, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)
    adv_actor_criterion.zero_grad()
    adv_critic = torch.load(ADV_CRITIC_PATH) if LOAD_MODEL else Critic()
    adv_critic_criterion = RMSprop(adv_critic.parameters(), lr=CRITIC_LR, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)
    adv_critic_criterion.zero_grad()
    adv_critic_loss_fn = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        pro_actor_parameters = list(pro_actor.parameters())
        adv_actor_parameters = list(adv_actor.parameters())

        for i in range(NUM_AGENTS):
            pro_par_q.put(pro_actor_parameters)
            adv_par_q.put(adv_actor_parameters)

        for i in range(NUM_AGENTS):
            pro_states, pro_actions, pro_rewards, pro_terminal = pro_exp_q.get()
            pro_states = torch.cat(pro_states)
            pro_actions = torch.LongTensor(pro_actions)
            pro_rewards = torch.tensor(np.array(pro_rewards))
            pro_returns = torch.zeros(pro_rewards.shape)

            adv_states, adv_actions, adv_rewards, adv_terminal = adv_exp_q.get()
            adv_states = torch.cat(adv_states)
            adv_actions = torch.LongTensor(adv_actions)
            adv_rewards = torch.tensor(np.array(adv_rewards))
            adv_returns = torch.zeros(adv_rewards.shape)

            pro_returns[-1] = pro_rewards[-1]
            for t in reversed(range(pro_rewards.shape[0] - 1)):
                pro_returns[t] = pro_rewards[t] + DISCOUNT_FACTOR * pro_returns[t + 1]
            with torch.no_grad():
                pro_values = pro_critic.forward(pro_states).squeeze()
                pro_advantage = pro_returns - pro_values
            
            adv_returns[-1] = adv_rewards[-1]
            for t in reversed(range(adv_rewards.shape[0] - 1)):
                adv_returns[t] = adv_rewards[t] + DISCOUNT_FACTOR * adv_returns[t + 1]
            with torch.no_grad():
                adv_values = adv_critic.forward(adv_states).squeeze()
                adv_advantage = adv_returns - adv_values

            curr_entropy_weight = INIT_ENTROPY_WT - (EPOCH_START + epoch + 1) / (EPOCH_START + float(NUM_EPOCHS)) * (FINAL_ENTROPY_WT - INIT_ENTROPY_WT)

            pro_probs = pro_actor.forward(pro_states)
            pro_dist = Categorical(pro_probs)
            pro_actor_loss = torch.sum(pro_dist.log_prob(pro_actions) * (- pro_advantage)) - curr_entropy_weight * torch.sum(pro_dist.entropy()) 
            pro_actor_loss.backward()

            adv_probs = adv_actor.forward(adv_states)
            adv_dist = Categorical(adv_probs)
            adv_actor_loss = torch.sum(adv_dist.log_prob(adv_actions) * (- adv_advantage)) - curr_entropy_weight * torch.sum(adv_dist.entropy()) 
            adv_actor_loss.backward()

            pro_critic_loss = pro_critic_loss_fn(pro_returns, pro_critic.forward(pro_states).squeeze())
            pro_critic_loss.backward()

            adv_critic_loss = adv_critic_loss_fn(adv_returns, adv_critic.forward(adv_states).squeeze())
            adv_critic_loss.backward()

        pro_actor_criterion.step()
        pro_actor_criterion.zero_grad()
        pro_critic_criterion.step()
        pro_critic_criterion.zero_grad()

        adv_actor_criterion.step()
        adv_actor_criterion.zero_grad()
        adv_critic_criterion.step()
        adv_critic_criterion.zero_grad()

        if (epoch + 1) % MODEL_SAVE_EVERY == 0:
            torch.save(pro_actor, SAVE_DIR + SAVE_NUM + '-pro-chkpt-actor-' + str(EPOCH_START + epoch + 1))
            torch.save(pro_critic, SAVE_DIR + SAVE_NUM + '-pro-chkpt-critic-' + str(EPOCH_START + epoch + 1))
            print('actor and critic saved by protagonist')
            torch.save(adv_actor, SAVE_DIR + SAVE_NUM + '-adv-chkpt-actor-' + str(EPOCH_START + epoch + 1))
            torch.save(adv_critic, SAVE_DIR + SAVE_NUM + '-adv-chkpt-critic-' + str(EPOCH_START + epoch + 1))
            print('actor and critic saved by antagonist')

def agent(i, pro_par_q, pro_exp_q, adv_par_q, adv_exp_q):
    torch.set_num_threads(1)
    pro_actor = Actor()
    adv_actor = Actor()

    environ = env.Environment(TIMING_DATA, THROUGHPUT_DATA, random_seed=hash(str(time.time()) + str(i)) % 2 ** 32 - 1)
    pro_scores, adv_scores = [], []
    pro_chosen_actions, adv_chosen_actions = [], []
    for epoch in range(NUM_EPOCHS):
        pro_global_actor_parameters = pro_par_q.get()
        adv_global_actor_parameters = adv_par_q.get()
        for target, source in zip(pro_actor.parameters(), pro_global_actor_parameters):
            target.data.copy_(source.data)
        for target, source in zip(adv_actor.parameters(), adv_global_actor_parameters):
            target.data.copy_(source.data)

        prev_bitrate_idx = DEFAULT_QUALITY
        curr_bitrate_idx = DEFAULT_QUALITY

        states, actions, pro_rewards, adv_rewards = [], [], [], []
        state = torch.zeros((1, S_INFO, S_LEN))

        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = environ.get_video_chunk(curr_bitrate_idx)

        pro_score, adv_score = 0, 0
        while not end_of_video and len(states) < TRAIN_SEQ_LEN:
            prev_bitrate_idx = curr_bitrate_idx

            state = state.clone().detach()
            state = torch.roll(state, shifts=-1,dims=-1)
            state[0,0, -1] = VID_BITRATES[curr_bitrate_idx] / float(np.max(VID_BITRATES))
            state[0,1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[0,2, -1] = float(video_chunk_size) / float(delay) / 1000
            state[0,3, -1] = float(delay) / 1000 / BUFFER_NORM_FACTOR
            state[0,4, :A_DIM] = torch.tensor(next_video_chunk_sizes) / pow(10, 6)
            state[0,5, -1] = min(video_chunk_remain, CHUNK_TILL_VIDEO_END_CAP) / float(CHUNK_TILL_VIDEO_END_CAP)

            curr_diff_lvl = INIT_DIFF_LVL + (EPOCH_START + epoch + 1) / (EPOCH_START + NUM_EPOCHS) * (FINAL_DIFF_LVL - INIT_DIFF_LVL)
            with torch.no_grad():
                pro_probs = pro_actor.forward(state)
                pro_dist = Categorical(pro_probs)
                pro_curr_bitrate_idx = pro_dist.sample().item()
                pro_chosen_actions.append(pro_curr_bitrate_idx)

                adv_probs = adv_actor.forward(state)
                adv_dist = Categorical(adv_probs)
                adv_curr_bitrate_idx = adv_dist.sample().item()
                adv_chosen_actions.append(adv_curr_bitrate_idx)

                curr_bitrate_idx = round(curr_diff_lvl * adv_curr_bitrate_idx + (1 - curr_diff_lvl) * pro_curr_bitrate_idx)

            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = environ.get_video_chunk(curr_bitrate_idx)

            if QOE_METRIC == 'lin':
                bitrate_reward = VID_BITRATES[curr_bitrate_idx] / 1000.0
                rebuf_penalty = LIN_REBUF_PENALTY * rebuf
            elif QOE_METRIC == 'log':
                bitrate_reward = np.log(VID_BITRATES[curr_bitrate_idx] / float(min(VID_BITRATES)))  # DBT: here I am not dividing by 1000 since it is a ratio
                rebuf_penalty = LOG_REBUF_PENALTY * rebuf
            else:
                bitrate_reward = HD_REWARD[curr_bitrate_idx]
                rebuf_penalty = HD_REBUF_PENALTY * rebuf
            smooth_penalty = SMOOTH_PENALTY * np.abs(VID_BITRATES[curr_bitrate_idx] - VID_BITRATES[prev_bitrate_idx]) / 1000.0
            pro_reward = bitrate_reward - rebuf_penalty - smooth_penalty + OFFSET
            pro_score += pro_reward - OFFSET

            adv_reward = - pro_reward
            adv_score += adv_reward + OFFSET

            states.append(state)
            actions.append(curr_bitrate_idx)
            pro_rewards.append(pro_reward)
            adv_rewards.append(adv_reward)

        pro_exp_q.put([states, actions, pro_rewards, end_of_video])
        pro_scores.append(pro_score)

        adv_exp_q.put([states, actions, adv_rewards, end_of_video])
        adv_scores.append(adv_score)
        if ((epoch + 1) == 10) and (10 % PRINT_EVERY != 0):  # just to ensure it is working   
            print('epoch', EPOCH_START + epoch + 1, '/', EPOCH_START + NUM_EPOCHS, 'agent', i, datetime.datetime.today().strftime('%d %b %Y %H:%M:%S %p'))
        if (epoch + 1) % PRINT_EVERY == 0:
            print('epoch', EPOCH_START + epoch + 1, '/', EPOCH_START + NUM_EPOCHS, 'agent', i, datetime.datetime.today().strftime('%d %b %Y %H:%M:%S %p'))
        if (epoch + 1) % NUM_EPOCHS == 0:
            print('last epoch, sleeping for central agent to finish')
            time.sleep(2)
    with open(RAW_SCORE_FILE + '-pro' + '-w' + str(i) + '.csv', 'w') as f:
        f.write('epoch,score\n')
        for ind, val in enumerate(pro_scores):
            f.write(str(ind + 1) + ',' + str(val) + '\n')
    with open(RAW_SCORE_FILE + '-adv' + '-w' + str(i) + '.csv', 'w') as f:
        f.write('epoch,score\n')
        for ind, val in enumerate(adv_scores):
            f.write(str(ind + 1) + ',' + str(val) + '\n')
    with open(RAW_ACTIONS_FILE+ '-pro' + '-w' + str(i) + '.csv', 'w') as f:
        f.write('index\n')
        for act in pro_chosen_actions:
            f.write(str(act) + '\n')
    with open(RAW_ACTIONS_FILE+ '-adv' + '-w' + str(i) + '.csv', 'w') as f:
        f.write('index\n')
        for act in adv_chosen_actions:
            f.write(str(act) + '\n')
    pro_scores = moving_average(pro_scores, n=SMOOTHING_AMT)
    plot(True, str(i), pro_scores)

def moving_average(a, n=SMOOTHING_AMT):
    ret = np.cumsum(np.array(a), dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot(is_pro, name, scores):
    agent_type = 'pro' if is_pro else 'adv' 
    ylim_path = SAVE_DIR + SAVE_NUM + '-score-zoom-' + agent_type + '-w' + name + '.png'
    full_path = SAVE_DIR + SAVE_NUM + '-score-full-' + agent_type + '-w' + name + '.png'
    plt.plot([i for i in range(EPOCH_START + 1, EPOCH_START + len(scores) + 1)], scores)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.title('worker ' + name)
    plt.savefig(full_path, bbox_inches='tight')
    if max(scores) >= 0 and is_pro:
        plt.ylim([-50, 50])
        plt.savefig(ylim_path, bbox_inches='tight')
    if max(scores) <= 0 and not is_pro:
        plt.ylim([-50, 50])
        plt.savefig(ylim_path, bbox_inches='tight')
    plt.clf()

def save_state(time_taken, monitor_queue):
    file_copy_path_pars = SAVE_DIR + SAVE_NUM + '-parameters.py'
    with open(file_copy_path_pars, 'a') as f:
        f.write('\ntime_taken = ' + str(time_taken))
    if MONITOR_PC_STATS:
        idx = 0
        cpu, ram, temp = [], [], []
        while not monitor_queue.empty():
            value = monitor_queue.get()
            if idx % 3 == 0:
                cpu.append(value)
            elif idx % 3 == 1:
                ram.append(value)
            else:
                temp.append(value)
            idx += 1
        smoothing_amount = max(len(cpu) // 10, 1)
        smooth_cpu = moving_average(cpu, n=smoothing_amount)
        plt.plot(list(range(1, len(smooth_cpu) + 1)), smooth_cpu)
        plt.xlabel('time in secs')
        plt.ylabel('cpu use in %')
        plt.title('cpu use')
        cpu_plot_file_path = SAVE_DIR + SAVE_NUM + '-stats-cpu.png'
        plt.savefig(cpu_plot_file_path)
        plt.clf()

        smooth_ram = moving_average(ram, n=smoothing_amount)
        plt.plot(list(range(1, len(smooth_ram) + 1)), smooth_ram)
        plt.xlabel('time in secs')
        plt.ylabel('ram use in %')
        plt.title('ram use')
        ram_plot_file_path = SAVE_DIR + SAVE_NUM + '-stats-ram.png'
        plt.savefig(ram_plot_file_path)
        plt.clf()

        smooth_temp = moving_average(temp, n=smoothing_amount)
        plt.plot(list(range(1, len(smooth_temp) + 1)), smooth_temp)
        plt.xlabel('time in secs')
        plt.ylabel('temp in degrees celcius')
        plt.title('avg cpu core temps')
        temp_plot_file_path = SAVE_DIR + SAVE_NUM + '-stats-temp.png'
        plt.savefig(temp_plot_file_path)

def monitor_fn(training_ongoing, monitor_queue):
    while training_ongoing.value == 1:
        virt_mem_used = (7.61 - psutil.virtual_memory().available / pow(1024, 3)) / 7.61 * 100
        cpu_use = psutil.cpu_percent(interval=1)
        temp = psutil.sensors_temperatures()['coretemp']
        temps = [t.current for t in temp if 'Core' in t.label]
        monitor_queue.put(cpu_use)
        monitor_queue.put(round(virt_mem_used, 2))
        monitor_queue.put(np.mean(temps))

def central_agent_profiler(pro_par_q, pro_exp_q, adv_par_q, adv_exp_q):
    central_profile = SAVE_DIR + SAVE_NUM + '-central_profile.dat'
    cProfile.runctx('central_agent(pro_par_q, pro_exp_q, adv_par_q, adv_exp_q)', {'pro_par_q': pro_par_q, 'pro_exp_q': pro_exp_q, 'adv_par_q': adv_par_q, 'adv_exp_q': adv_exp_q, 'central_agent': central_agent}, {},  central_profile)
    central_profile_time = SAVE_DIR + SAVE_NUM + '-central_profile_time.txt'
    central_profile_calls = SAVE_DIR + SAVE_NUM + '-central_profile_calls.txt'
    with open(central_profile_time, 'w') as f:
        p = pstats.Stats(central_profile, stream=f)
        p.sort_stats('time').print_stats()
    with open(central_profile_calls, 'w') as f:
        p = pstats.Stats(central_profile, stream=f)
        p.sort_stats('calls').print_stats()

def agent_profiler(i, pro_par_q, pro_exp_q, adv_par_q, adv_exp_q):
    agent_profile = SAVE_DIR + SAVE_NUM + '-agent' + str(index) + '-profile.dat'
    cProfile.runctx('agent(i, pro_par_q, pro_exp_q, adv_par_q, adv_exp_q)', {'i': i, 'pro_par_q': pro_par_q, 'pro_exp_q': pro_exp_q, 'adv_par_q': adv_par_q, 'adv_exp_q': adv_exp_q, 'agent': agent}, {}, agent_profile)
    agent_profile_time = SAVE_DIR + SAVE_NUM + '-agent' + str(index) + '-profile_time.txt'
    agent_profile_calls  = SAVE_DIR + SAVE_NUM + '-agent' + str(index) + '-profile_calls.txt'
    with open(agent_profile_time, 'w') as f:
        p = pstats.Stats(agent_profile, stream=f)
        p.sort_stats('time').print_stats()
    with open(agent_profile_calls, 'w') as f:
        p = pstats.Stats(agent_profile, stream=f)
        p.sort_stats('calls').print_stats()

def preliminaries():
    variable_names = ['S_INFO', 'S_LEN', 'VID_BITRATES', 'A_DIM', 'HD_REWARD', 'BUFFER_NORM_FACTOR', 'CHUNK_TILL_VIDEO_END_CAP', 'DEFAULT_QUALITY', 'CONV_IN_CHANNELS', 'NN_IN_FEATURES', 'ACTOR_LR', 'CRITIC_LR', 'TRAIN_SEQ_LEN', 'RMSPROP_ALPHA', 'RMSPROP_EPS', 'NUM_HID_NEURONS', 'NUM_CONV_CHANNELS', 'CONV_KERNEL_SIZE', 'CONV_STRIDE', 'CONV_PADDING', 'NN_ACT_FN', 'DISCOUNT_FACTOR', 'INIT_ENTROPY_WT', 'FINAL_ENTROPY_WT', 'LIN_REBUF_PENALTY', 'LOG_REBUF_PENALTY', 'HD_REBUF_PENALTY', 'SMOOTH_PENALTY', 'DATA_PATH', 'INIT_DIFF_LVL', 'FINAL_DIFF_LVL', 'OFFSET', 'SAVE_DIR', 'SAVE_NUM', 'RAW_SCORE_FILE', 'RAW_ACTIONS_FILE', 'CURR_FILE_PATH', 'ADV_ACTOR_PATH', 'PRO_ACTOR_PATH', 'ADV_CRITIC_PATH', 'PRO_CRITIC_PATH', 'LOAD_MODEL', 'EPOCH_START', 'QOE_METRIC', 'NUM_EPOCHS', 'PRINT_EVERY', 'SMOOTHING_AMT', 'MODEL_SAVE_EVERY', 'USE_GPU', 'DEVICE', 'NUM_AGENTS', 'DO_PROFILING', 'SET_SEEDS', 'MONITOR_PC_STATS']
    variable_values = [S_INFO, S_LEN, VID_BITRATES, A_DIM, HD_REWARD, BUFFER_NORM_FACTOR, CHUNK_TILL_VIDEO_END_CAP, DEFAULT_QUALITY, CONV_IN_CHANNELS, NN_IN_FEATURES, ACTOR_LR, CRITIC_LR, TRAIN_SEQ_LEN, RMSPROP_ALPHA, RMSPROP_EPS, NUM_HID_NEURONS, NUM_CONV_CHANNELS, CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING, NN_ACT_FN, DISCOUNT_FACTOR, INIT_ENTROPY_WT, FINAL_ENTROPY_WT, LIN_REBUF_PENALTY, LOG_REBUF_PENALTY, HD_REBUF_PENALTY, SMOOTH_PENALTY, DATA_PATH, INIT_DIFF_LVL, FINAL_DIFF_LVL, OFFSET, SAVE_DIR, SAVE_NUM, RAW_SCORE_FILE, RAW_ACTIONS_FILE, CURR_FILE_PATH, ADV_ACTOR_PATH, PRO_ACTOR_PATH, ADV_CRITIC_PATH, PRO_CRITIC_PATH, LOAD_MODEL, EPOCH_START, QOE_METRIC, NUM_EPOCHS, PRINT_EVERY, SMOOTHING_AMT, MODEL_SAVE_EVERY, USE_GPU, DEVICE, NUM_AGENTS, DO_PROFILING, SET_SEEDS, MONITOR_PC_STATS]
    # Setting seeds
    if SET_SEEDS:
        torch.backends.cudnn.deterministic, 
        random.seed(hash("setting random seeds") % 2**32 - 1)
        np.random.seed(hash("improves reproducibility") % 2**32 - 1)
        torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
        torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    # printing information
    now = datetime.datetime.today()
    print('started on', now.strftime('%d %b %Y %H:%M:%S %p'))
    print('num epochs', NUM_EPOCHS, 'num agents', NUM_AGENTS, 'device', DEVICE, 'load model', LOAD_MODEL, 'batch size', TRAIN_SEQ_LEN, 'metric', QOE_METRIC)
    print(f'based on {SECS_PER_EPOCH} secs / epoch, {NUM_EPOCHS} epochs will take {SECS_PER_EPOCH * NUM_EPOCHS} secs, so will finish at {(datetime.timedelta(seconds=SECS_PER_EPOCH * NUM_EPOCHS) + now).strftime("%d %b %Y %H:%M:%S %p")}')

    # create save directory if it does not exist
    path = Path(SAVE_DIR)
    if not path.is_dir():
        os.mkdir(path)
        print('created directory to save runs in:', path)

    # save code and parameters
    file_copy_path_code = SAVE_DIR + SAVE_NUM + '-code.py'
    shutil.copyfile(CURR_FILE_PATH, file_copy_path_code)
    file_copy_path_pars = SAVE_DIR + SAVE_NUM + '-parameters.py'
    with open(file_copy_path_pars, 'w') as f:
        f.write('name,value\n')
        for name, value in zip(variable_names, variable_values):
            f.write(name + ',' + str(value) + '\n')
    print('code and parameters file saved')

    # allow cuda be used with multiprocessing
    if DEVICE == 'cuda:0':
        mp.set_start_method('spawn')


if __name__ == '__main__':
    preliminaries()

    start_time = time.perf_counter()
    pro_par_q = mp.Queue()
    pro_exp_q = mp.Queue()

    adv_par_q = mp.Queue()
    adv_exp_q = mp.Queue()
    
    if MONITOR_PC_STATS:
        training_ongoing = mp.Value('i')
        training_ongoing.value = 1
        monitor_queue = mp.Queue()
        monitor_process = mp.Process(target=monitor_fn, args=(training_ongoing, monitor_queue))
        monitor_process.start()

    coordinator_target = central_agent_profiler if DO_PROFILING else central_agent
    coordinator = mp.Process(target=coordinator_target, args=(pro_par_q, pro_exp_q, adv_par_q, adv_exp_q))
    coordinator.start()

    agents_target = agent_profiler if DO_PROFILING else agent
    agents = [mp.Process(target=agents_target, args=(i, pro_par_q, pro_exp_q, adv_par_q, adv_exp_q)) for i in range(NUM_AGENTS)]
    for process in agents:
        process.start()

    coordinator.join()
    for process in agents:
        process.join()

    time_taken = time.perf_counter() - start_time

    if MONITOR_PC_STATS:
        training_ongoing.value = 0
        time.sleep(0.5)
        monitor_process.terminate()
        monitor_process.join()
        print('monitor_done')
    save_state(time_taken, monitor_queue if MONITOR_PC_STATS else None)
