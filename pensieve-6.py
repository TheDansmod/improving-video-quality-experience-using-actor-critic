# This one I am trying to do with combined actor and critic
import numpy as np
import env  # this is actually a file env.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
from torch.optim import RMSprop
import math
import random
import load_trace  # this is a file load_trace.py
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


NUM_EPOCHS = 10000
NUM_AGENTS = 4
TRAIN_SEQ_LEN = 100  # train batch size
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
RMSPROP_LR = 0.0001
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
QOE_METRIC = 'lin' # other options are 'log' and 'hd'
LOAD_MODEL = False
SAVE_DIR = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_' + Path(os.path.basename(__file__)).stem + '/'
CURR_FILE_PATH = Path(__file__).absolute()
USE_GPU = False
DO_PROFILING = False
SET_SEEDS = False
MONITOR_PC_STATS = True
DEVICE = 'cuda:0' if (USE_GPU and torch.cuda.is_available()) else 'cpu'
PRINT_EVERY = NUM_EPOCHS // 10
MODEL_SAVE_EVERY = NUM_EPOCHS // 5
DATA_PATH = r'/mnt/windows_d/Svalbard/Data/Thesis/pensieve_data/cooked_traces/'
TIMING_DATA = tuple(load_trace.load_trace(DATA_PATH)[0])
THROUGHPUT_DATA = tuple(load_trace.load_trace(DATA_PATH)[1])
SMOOTHING_AMT = NUM_EPOCHS // 10
save_num = max(set([name[:4] for name in os.listdir(SAVE_DIR)])) + 1 if Path(SAVE_DIR).exists() else 1
SAVE_NUM = '0' * (4 - len(str(save_num))) + str(save_num)
EPOCH_START = int(MODEL_PATH.split('-')[-1]) if LOAD_MODEL else 0
RAW_SCORE_FILE = SAVE_DIR + SAVE_NUM + '-raw-score'
RAW_ACTIONS_FILE = SAVE_DIR + SAVE_NUM + '-raw-actions'
ACTION_GRAPH_SMOOTHING_AMT = 1000
SECS_PER_EPOCH = 0.125 if NUM_AGENTS == 4 else (0.28 if NUM_AGENTS == 6 else 0.102)
BIAS_INIT = 0.01
MODEL_PATH = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-3-async/0011-chkpt-actor-50'


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
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

        self.common_layer = nn.Linear(flattened_input_size, NUM_HID_NEURONS)
        self.actor_out = nn.Linear(NUM_HID_NEURONS, A_DIM)
        self.critic_out = nn.Linear(NUM_HID_NEURONS, 1)

        nn.init.xavier_normal_(self.bitrate.weight.data)
        nn.init.constant_(self.bitrate.bias.data, BIAS_INIT)
        nn.init.xavier_normal_(self.buffer_size.weight.data)
        nn.init.constant_(self.buffer_size.bias.data, BIAS_INIT)
        nn.init.xavier_normal_(self.throughput.weight.data)
        nn.init.constant_(self.throughput.bias.data, BIAS_INIT)
        nn.init.xavier_normal_(self.download_time.weight.data)
        nn.init.constant_(self.download_time.bias.data, BIAS_INIT)
        nn.init.xavier_normal_(self.chunk_size.weight.data)
        nn.init.constant_(self.chunk_size.bias.data, BIAS_INIT)
        nn.init.xavier_normal_(self.chunks_remaining.weight.data)
        nn.init.constant_(self.chunks_remaining.bias.data, BIAS_INIT)
        nn.init.xavier_normal_(self.common_layer.weight.data)
        nn.init.constant_(self.common_layer.bias.data, BIAS_INIT)

    def forward(self, state):
        bitrate = NN_ACT_FN(self.bitrate(state[: , 0: 1, -1]), inplace=True)
        buffer_size = NN_ACT_FN(self.buffer_size(state[: , 1: 2, -1]), inplace=True)
        throughput = NN_ACT_FN(self.throughput(state[: , 2: 3, : ]), inplace=True)
        download_time = NN_ACT_FN(self.download_time(state[: , 3: 4, : ]), inplace=True)
        chunk_size = NN_ACT_FN(self.chunk_size(state[: , 4: 5, : A_DIM]), inplace=True)
        chunks_remaining = NN_ACT_FN(self.chunks_remaining(state[: , 5: 6, -1]), inplace=True)

        throughput_flat = torch.flatten(throughput, 1)  # flatten everything except batch
        download_time_flat = torch.flatten(download_time, 1)
        chunk_size_flat = torch.flatten(chunk_size, 1)
        flattened_input = torch.cat([bitrate, buffer_size, throughput_flat, download_time_flat, chunk_size_flat, chunks_remaining], dim=1)

        common_outputs = NN_ACT_FN(self.common_layer(flattened_input))
        action_probs = F.softmax(self.actor_out(common_outputs), dim=1)
        values = self.critic_out(common_outputs)

        return action_probs, values


def central_agent(parameter_queue, experience_queue):
    torch.set_num_threads(1)
    model = torch.load(MODEL_PATH).to(DEVICE) if LOAD_MODEL else ActorCritic().to(DEVICE)
    criterion = RMSprop(model.parameters(), lr=RMSPROP_LR, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)
    criterion.zero_grad()
    critic_loss_fn = nn.MSELoss()

    central_agent_scores = []
    for epoch in range(NUM_EPOCHS):
        model_parameters = list(model.parameters())

        for i in range(NUM_AGENTS):
            parameter_queue.put(model_parameters)

        total_reward = 0
        for i in range(NUM_AGENTS):
            states, actions, rewards, terminal = experience_queue.get()
            states = torch.cat(states).to(DEVICE)
            actions = torch.LongTensor(actions).to(DEVICE)
            rewards = torch.tensor(np.array(rewards)).to(DEVICE)
            returns = torch.zeros(rewards.shape).to(DEVICE)

            with torch.no_grad():
                total_reward += torch.sum(rewards)

            returns[-1] = rewards[-1]  # you need returns to have grad since you use it for critic
            for t in reversed(range(rewards.shape[0] - 1)):
                returns[t] = rewards[t] + DISCOUNT_FACTOR * returns[t + 1]
            # this extra forward pass does not seem needed, so I will put advantage later as returns - values.squeeze()
            # with torch.no_grad():
            #     _, values = model.forward(states)
            #     advantage = returns - values.squeeze()
            
            curr_entropy_weight = INIT_ENTROPY_WT - (epoch + 1) / float(NUM_EPOCHS) * (INIT_ENTROPY_WT - FINAL_ENTROPY_WT)
            probs, values = model.forward(states)
            dist = Categorical(probs)
            actor_loss = torch.sum(dist.log_prob(actions) * (values.squeeze().to(DEVICE) - returns)) - curr_entropy_weight * torch.sum(dist.entropy())
            critic_loss = critic_loss_fn(returns, values.squeeze())
            loss = (critic_loss + actor_loss).mean()
            loss.backward()
        central_agent_scores.append(float(total_reward / NUM_AGENTS))

        criterion.step()
        criterion.zero_grad()
        if (epoch +1) % MODEL_SAVE_EVERY == 0:
            torch.save(model, SAVE_DIR + SAVE_NUM + '-chkpt-' + str(EPOCH_START + epoch + 1))
            print('saved model')
    with open(RAW_SCORE_FILE + '-central-agent.csv', 'w') as f:
        f.write('epoch,score\n')
        for epoch, qoe in enumerate(central_agent_scores):
            f.write(str(epoch) + ',' + str(qoe) + '\n')

def agent(i, parameter_queue, experience_queue):
    torch.set_num_threads(1)
    model = ActorCritic()

    environ = env.Environment(TIMING_DATA, THROUGHPUT_DATA, random_seed=hash(str(time.time()) + str(i)) % 2 ** 32 - 1)
    scores = []
    chosen_actions = []
    for epoch in range(NUM_EPOCHS):
        global_model_params = parameter_queue.get()
        for target, source in zip(model.parameters(), global_model_params):
            target.data.copy_(source.data)

        prev_bitrate_idx = DEFAULT_QUALITY
        curr_bitrate_idx = DEFAULT_QUALITY

        states, actions, rewards = [], [], []
        state = torch.zeros((1, S_INFO, S_LEN))

        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = environ.get_video_chunk(curr_bitrate_idx)

        score = 0
        while not end_of_video and len(states) < TRAIN_SEQ_LEN:
            prev_bitrate_idx = curr_bitrate_idx

            state = state.clone().detach()
            state = torch.roll(state, shifts=-1,dims=-1)
            state[0,0, -1] = VID_BITRATES[curr_bitrate_idx] / float(np.max(VID_BITRATES))
            state[0,1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[0,2, -1] = float(video_chunk_size) / float(delay) / 1000
            state[0,3, -1] = float(delay) / 1000 / BUFFER_NORM_FACTOR
            state[0,4, :A_DIM] = torch.tensor(next_video_chunk_sizes) / float(pow(10, 6))
            state[0,5, -1] = min(video_chunk_remain, CHUNK_TILL_VIDEO_END_CAP) / float(CHUNK_TILL_VIDEO_END_CAP)

            with torch.no_grad():
                probs, _ = model.forward(state)
                dist = Categorical(probs)
                curr_bitrate_idx = dist.sample().item()

            chosen_actions.append(curr_bitrate_idx)
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
            reward = bitrate_reward - rebuf_penalty - smooth_penalty
            score += reward

            states.append(state)
            actions.append(curr_bitrate_idx)
            rewards.append(reward)

        experience_queue.put([states, actions, rewards, end_of_video])
        scores.append(score)
        if ((epoch + 1) == 10) and (10 % PRINT_EVERY != 0):  # just to ensure it is working   
            print('epoch', EPOCH_START + epoch + 1, '/', EPOCH_START + NUM_EPOCHS, 'agent', i, datetime.datetime.today().strftime('%d %b %Y %H:%M:%S %p'))
        if (epoch + 1) % PRINT_EVERY == 0:
            print('epoch', EPOCH_START + epoch + 1, '/', EPOCH_START + NUM_EPOCHS, 'agent', i, datetime.datetime.today().strftime('%d %b %Y %H:%M:%S %p'))
    with open(RAW_SCORE_FILE + '-w' + str(i) + '.csv', 'w') as f:
        f.write('epoch,score\n')
        for ind, val in enumerate(scores):
            f.write(str(ind + 1) + ',' + str(val) + '\n')
    with open(RAW_ACTIONS_FILE + '-w' + str(i) + '.csv', 'w') as f:
        f.write('index\n')
        for act in chosen_actions:
            f.write(str(act) + '\n')
    scores = moving_average(scores, n=SMOOTHING_AMT)
    chosen_actions = moving_average(chosen_actions, n=ACTION_GRAPH_SMOOTHING_AMT)
    plot_scores(str(i), scores)
    plot_actions(str(i), chosen_actions)

def moving_average(a, n=SMOOTHING_AMT):
    ret = np.cumsum(np.array(a), dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_scores(name, scores):
    ylim_path = SAVE_DIR + SAVE_NUM + '-score-zoom_w' + name + '.png'
    full_path = SAVE_DIR + SAVE_NUM + '-score-full-w' + name + '.png'
    plt.plot([i for i in range(EPOCH_START + 1, EPOCH_START + len(scores) + 1)], scores)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.title('worker ' + name)
    plt.savefig(full_path, bbox_inches='tight')
    if max(scores) >= 0:
        plt.ylim([-50, 50])
        plt.savefig(ylim_path, bbox_inches='tight')
    plt.clf()

def plot_actions(name, actions):
    path = SAVE_DIR + SAVE_NUM + '-actions_plot-w' + name + '.png'
    plt.plot([i for i in range(EPOCH_START + 1, EPOCH_START + len(actions) + 1)], actions)
    plt.xlabel('steps (not epochs) over ' + str(NUM_EPOCHS) + ' epochs')
    plt.ylabel('average of actions taken (0 to 5 incl. only integers)')
    plt.title('worker ' + name + ' actions with moving average of ' + str(ACTION_GRAPH_SMOOTHING_AMT))
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

def save_state(time_taken, monitor_queue):
    file_copy_path_pars = SAVE_DIR + SAVE_NUM + '-parameters.py'
    with open(file_copy_path_pars, 'a') as f:
        f.write('\ntime_taken = ' + str(time_taken))
    if MONITOR_PC_STATS:
        with open(file_copy_path_pars, 'a') as f:
            f.write('\ntime_taken = ,' + str(time_taken))
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

def central_agent_profiler(parameter_queue, experience_queue):
    central_profile = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/results/central_profile.dat'
    cProfile.runctx('central_agent(parameter_queue, experience_queue)', {'parameter_queue': parameter_queue, 'experience_queue': experience_queue, 'central_agent': central_agent}, {},  central_profile)
    central_profile_time = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/results/central_profile_time.txt'
    central_profile_calls = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/results/central_profile_calls.txt'
    with open(central_profile_time, 'w') as f:
        p = pstats.Stats(central_profile, stream=f)
        p.sort_stats('time').print_stats()
    with open(central_profile_calls, 'w') as f:
        p = pstats.Stats(central_profile, stream=f)
        p.sort_stats('calls').print_stats()

def agent_profiler(index, parameter_queue, experience_queue):
    agent_profile = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/results/agent' + str(index) + '_profile.dat'
    cProfile.runctx('agent(index, parameter_queue, experience_queue)', {'index': index, 'parameter_queue': parameter_queue, 'experience_queue': experience_queue, 'agent': agent}, {}, agent_profile)
    agent_profile_time = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/results/agent' + str(index) + '_profile_time.txt'
    agent_profile_calls = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/results/agent' + str(index) + '_profile_calls.txt'
    with open(agent_profile_time, 'w') as f:
        p = pstats.Stats(agent_profile, stream=f)
        p.sort_stats('time').print_stats()
    with open(agent_profile_calls, 'w') as f:
        p = pstats.Stats(agent_profile, stream=f)
        p.sort_stats('calls').print_stats()

def preliminaries():
    variable_names = ['S_INFO', 'S_LEN', 'VID_BITRATES', 'A_DIM', 'HD_REWARD', 'BUFFER_NORM_FACTOR', 'CHUNK_TILL_VIDEO_END_CAP', 'DEFAULT_QUALITY', 'CONV_IN_CHANNELS', 'NN_IN_FEATURES', 'RMSPROP_LR', 'TRAIN_SEQ_LEN', 'RMSPROP_ALPHA', 'RMSPROP_EPS', 'NUM_HID_NEURONS', 'NUM_CONV_CHANNELS', 'CONV_KERNEL_SIZE', 'CONV_STRIDE', 'CONV_PADDING', 'NN_ACT_FN', 'DISCOUNT_FACTOR', 'INIT_ENTROPY_WT', 'FINAL_ENTROPY_WT', 'LIN_REBUF_PENALTY', 'LOG_REBUF_PENALTY', 'HD_REBUF_PENALTY', 'SMOOTH_PENALTY', 'DATA_PATH', 'SAVE_DIR', 'SAVE_NUM', 'RAW_SCORE_FILE', 'RAW_ACTIONS_FILE', 'CURR_FILE_PATH', 'MODEL_PATH', 'LOAD_MODEL', 'EPOCH_START', 'QOE_METRIC', 'NUM_EPOCHS', 'PRINT_EVERY', 'SMOOTHING_AMT', 'MODEL_SAVE_EVERY', 'USE_GPU', 'DEVICE', 'NUM_AGENTS', 'DO_PROFILING', 'SET_SEEDS', 'MONITOR_PC_STATS']
    variable_values = [S_INFO, S_LEN, VID_BITRATES, A_DIM, HD_REWARD, BUFFER_NORM_FACTOR, CHUNK_TILL_VIDEO_END_CAP, DEFAULT_QUALITY, CONV_IN_CHANNELS, NN_IN_FEATURES, RMSPROP_LR, TRAIN_SEQ_LEN, RMSPROP_ALPHA, RMSPROP_EPS, NUM_HID_NEURONS, NUM_CONV_CHANNELS, CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING, NN_ACT_FN, DISCOUNT_FACTOR, INIT_ENTROPY_WT, FINAL_ENTROPY_WT, LIN_REBUF_PENALTY, LOG_REBUF_PENALTY, HD_REBUF_PENALTY, SMOOTH_PENALTY, DATA_PATH, SAVE_DIR, SAVE_NUM, RAW_SCORE_FILE, RAW_ACTIONS_FILE, CURR_FILE_PATH, MODEL_PATH, LOAD_MODEL, EPOCH_START, QOE_METRIC, NUM_EPOCHS, PRINT_EVERY, SMOOTHING_AMT, MODEL_SAVE_EVERY, USE_GPU, DEVICE, NUM_AGENTS, DO_PROFILING, SET_SEEDS, MONITOR_PC_STATS]
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
    parameter_queue = mp.Queue()
    experience_queue = mp.Queue()

    if MONITOR_PC_STATS:
        training_ongoing = mp.Value('i')
        training_ongoing.value = 1
        monitor_queue = mp.Queue()
        monitor_process = mp.Process(target=monitor_fn, args=(training_ongoing, monitor_queue))
        monitor_process.start()
    
    coordinator_target = central_agent_profiler if DO_PROFILING else central_agent
    coordinator = mp.Process(target=coordinator_target, args=(parameter_queue, experience_queue))
    coordinator.start()

    agents_target = agent_profiler if DO_PROFILING else agent
    agents = [mp.Process(target=agents_target, args=(i, parameter_queue, experience_queue)) for i in range(NUM_AGENTS)]
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
