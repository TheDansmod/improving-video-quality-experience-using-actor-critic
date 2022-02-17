from dataclasses import dataclass
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
from torch.optim import RMSprop
import math
import random
import matplotlib.pyplot as plt
import os
import time
import shutil
import psutil
from datetime import datetime
from pathlib import Path


@dataclass
class cons:
    NN_OUT_FEATURES: int = 1
    NN_IN_FEATURES: int = 1


@dataclass
class hyp:  # changing these will theoretically impact the performace of the model
    NUM_EPOCHS: int = 10000
    ACTOR_LR: float = 0.0001
    CRITIC_LR: float = 0.001
    NUM_AGENTS: int = 4
    TRAIN_SEQ_LEN: int = 100  # train batch size
    RMSPROP_ALPHA: float = 0.9
    RMSPROP_EPS: float = 1e-10
    NUM_HID_NEURONS: int = 128
    NN_ACT_FN: int = F.relu
    DISCOUNT_FACTOR: float = 0.99
    INIT_ENTROPY_WT: float = 1.0
    FINAL_ENTROPY_WT: float = 0.1

@dataclass
class par:  # variables that are not directly related to the model and its performance
    PRINT_EVERY: int = 100, hyp.NUM_EPOCHS // 10
    MODEL_SAVE_EVERY: int = 1000, hyp.NUM_EPOCHS // 5
    DATA_PATH: str = r'/mnt/windows_d/Svalbard/Data/Thesis/pensieve_data/cooked_traces/'
    TIMING_DATA: tuple = tuple(load_trace.load_trace(DATA_PATH)[0])
    THROUGHPUT_DATA: tuple = tuple(load_trace.load_trace(DATA_PATH)[1])
    SMOOTHING_AMT: int = hyp.NUM_EPOCHS // 10
    SAVE_DIR: str = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_' + Path(os.path.basename(__file__)).stem + '/'
    save_num: int = len(set([name[:4] for name in os.listdir(SAVE_DIR)])) + 1 if Path(SAVE_DIR).exists() else 0
    SAVE_NUM: str = '0' * (4 - len(str(save_num))) + str(save_num)
    CURR_FILE_PATH: str = os.path.realpath(__file__)
    LOAD_MODEL: bool = False
    ACTOR_PATH: str = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-3-async/0011-chkpt-actor-50'
    CRITIC_PATH: str = ACTOR_PATH.replace('-actor-', '-critic-')
    EPOCH_START: int = int(ACTOR_PATH.split('-')[-1]) if LOAD_MODEL else 0
    RAW_SCORE_FILE: str = SAVE_DIR + SAVE_NUM + '-raw-score'


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
		self.actor_layer1 = nn.Linear(cons.NN_IN_FEATURES, hyp.NUM_HID_NEURONS)
		self.actor_layer2 = nn.Linear(hyp.NUM_HID_NEURONS, hyp.NUM_HID_NEURONS)
		self.actor_layer3 = nn.Linear(hyp.NUM_HID_NEURONS, cons.NN_OUT_FEATURES)

    def forward(self, x):
		x = hyp.NN_ACT_FN(self.actor_layer1(x))
        x = hyp.NN_ACT_FN(self.actor_layer2(x))
        out = F.sigmoid(self.actor_layer3(x))
        return out
        

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
		self.critic_layer1 = nn.Linear(cons.NN_IN_FEATURES, hyp.NUM_HID_NEURONS)
		self.critic_layer2 = nn.Linear(hyp.NUM_HID_NEURONS, hyp.NUM_HID_NEURONS)
		self.critic_layer3 = nn.Linear(hyp.NUM_HID_NEURONS, 1)

    def forward(self, x):
		x = hyp.NN_ACT_FN(self.critic_layer1(x))
        x = hyp.NN_ACT_FN(self.critic_layer2(x))
        out = self.critic_layer3(x)
        return out

def set_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

def central_agent(parameter_queue, experience_queue):
    actor = torch.load(par.ACTOR_PATH) if par.LOAD_MODEL else Actor()
    actor_criterion = RMSprop(actor.parameters(), lr=hyp.ACTOR_LR, alpha=hyp.RMSPROP_ALPHA, eps=hyp.RMSPROP_EPS)
    actor_criterion.zero_grad()
    critic = torch.load(par.CRITIC_PATH) if par.LOAD_MODEL else Critic()
    critic_criterion = RMSprop(critic.parameters(), lr=hyp.CRITIC_LR, alpha=hyp.RMSPROP_ALPHA, eps=hyp.RMSPROP_EPS)
    critic_criterion.zero_grad()
    critic_loss_fn = nn.MSELoss()

    for epoch in range(hyp.NUM_EPOCHS):
        actor_parameters = list(actor.parameters())

        for i in range(hyp.NUM_AGENTS):
            parameter_queue.put(actor_parameters)

        for i in range(hyp.NUM_AGENTS):
            # DBT: to use CUDA with multiprocessing you need to use the spawn method which I have no clue about
            states, actions, rewards, terminal = experience_queue.get()
            states = torch.tensor(np.array(states))
            actions = torch.tensor(np.array(actions))
            rewards = torch.tensor(np.array(rewards))
            returns = torch.zeros(rewards.shape)

            returns[-1] = rewards[-1]
            for t in reversed(range(rewards.shape[0] - 1)):
                returns[t] = rewards[t] + hyp.DISCOUNT_FACTOR * returns[t + 1]
            with torch.no_grad():
                values = critic.forward(states).squeeze()
                advantage = returns - values
            
            actor_loss = torch.sum(dist.log_prob(actions) * (- advantage)) + curr_entropy_weight * torch.sum(dist.entropy()) 
            actor_loss.backward()

            critic_loss = critic_loss_fn(returns, critic.forward(states).squeeze())
            critic_loss.backward()

        actor_criterion.step()
        actor_criterion.zero_grad()
        critic_criterion.step()
        critic_criterion.zero_grad()
        if (epoch +1) % par.MODEL_SAVE_EVERY == 0:
            torch.save(actor, par.SAVE_DIR + par.SAVE_NUM + '-chkpt-actor-' + str(par.EPOCH_START + epoch + 1))
            torch.save(critic, par.SAVE_DIR + par.SAVE_NUM + '-chkpt-critic-' + str(par.EPOCH_START + epoch + 1))

def agent(i, parameter_queue, experience_queue):
    torch.set_num_threads(1)
    actor = Actor()

    environ = env.Environment(par.TIMING_DATA, par.THROUGHPUT_DATA, random_seed=hash(str(time.time()) + str(i)) % 2 ** 32 - 1)
    scores = []
    for epoch in range(hyp.NUM_EPOCHS):
        global_actor_params = parameter_queue.get()
        for target, source in zip(actor.parameters(), global_actor_params):
            target.data.copy_(source.data)

        prev_bitrate_idx = cons.DEFAULT_QUALITY
        curr_bitrate_idx = cons.DEFAULT_QUALITY

        states, actions, rewards = [], [], []
        state = torch.zeros((1, cons.S_INFO, cons.S_LEN))

        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = environ.get_video_chunk(curr_bitrate_idx)

        score = 0
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

            with torch.no_grad():
                probs = actor.forward(state)
                dist = Categorical(probs)
                curr_bitrate_idx = dist.sample().item()

            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = environ.get_video_chunk(curr_bitrate_idx)

            if hyp.QOE_METRIC == 'lin':
                bitrate_reward = cons.VID_BITRATES[curr_bitrate_idx] / 1000.0
                rebuf_penalty = hyp.LIN_REBUF_PENALTY * rebuf
            elif hyp.QOE_METRIC == 'log':
                bitrate_reward = np.log(cons.VID_BITRATES[curr_bitrate_idx] / float(min(cons.VID_BITRATES)))  # DBT: here I am not dividing by 1000 since it is a ratio
                rebuf_penalty = hyp.LOG_REBUF_PENALTY * rebuf
            else:
                bitrate_reward = cons.HD_REWARD[curr_bitrate_idx]
                rebuf_penalty = hyp.HD_REBUF_PENALTY * rebuf
            smooth_penalty = hyp.SMOOTH_PENALTY * np.abs(cons.VID_BITRATES[curr_bitrate_idx] - cons.VID_BITRATES[prev_bitrate_idx]) / 1000.0
            reward = bitrate_reward - rebuf_penalty - smooth_penalty
            score += reward

            states.append(state)
            actions.append(curr_bitrate_idx)
            rewards.append(reward)

        experience_queue.put([states, actions, rewards, end_of_video])
        scores.append(score)
        if (epoch + 1) % par.PRINT_EVERY == 0:
            print('epoch', par.EPOCH_START + epoch + 1, '/', par.EPOCH_START + hyp.NUM_EPOCHS, 'agent', i, datetime.today().strftime('%d %b %Y %H:%M:%S %p'))
    with open(par.RAW_SCORE_FILE + '-w' + str(i) + '.csv', 'w') as f:
        f.write('epoch,score\n')
        for ind, val in enumerate(scores):
            f.write(str(ind + 1) + ',' + str(val) + '\n')
    scores = moving_average(scores, n=par.SMOOTHING_AMT)
    plot(str(i), scores)

def moving_average(a, n=par.SMOOTHING_AMT):
    ret = np.cumsum(np.array(a), dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot(name, scores):
    ylim_path = par.SAVE_DIR + par.SAVE_NUM + '-score-zoom_w' + name + '.png'
    full_path = par.SAVE_DIR + par.SAVE_NUM + '-score-full-w' + name + '.png'
    plt.plot([i for i in range(par.EPOCH_START + 1, par.EPOCH_START + len(scores) + 1)], scores)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.title('worker ' + name)
    plt.savefig(full_path, bbox_inches='tight')
    if max(scores) >= 0:
        plt.ylim([-50, 50])
        plt.savefig(ylim_path, bbox_inches='tight')

def save_state(time_taken, monitor_queue):
    file_copy_path = par.SAVE_DIR + par.SAVE_NUM + '-code.py'
    shutil.copyfile(par.CURR_FILE_PATH, file_copy_path)
    with open(par.SAVE_DIR + par.SAVE_NUM + '-hyp_cons_par.csv', 'w') as f:
        f.write('name,value\n')
        [f.write(name + ',' + str(value) + '\n') for name, value in vars(hyp).items() if not name.startswith('__')]
        [f.write(name + ',' + str(value) + '\n') for name, value in vars(cons).items() if not name.startswith('__')]
        [f.write(name + ',' + str(value) + '\n') for name, value in vars(par).items() if not name.startswith('__') and not name.endswith('_DATA')]
        f.write('time_taken,' + str(time_taken))
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
    cpu_plot_file_path = par.SAVE_DIR + par.SAVE_NUM + '-stats-cpu.png'
    plt.savefig(cpu_plot_file_path)
    plt.clf()

    smooth_ram = moving_average(ram, n=smoothing_amount)
    plt.plot(list(range(1, len(smooth_ram) + 1)), smooth_ram)
    plt.xlabel('time in secs')
    plt.ylabel('ram use in %')
    plt.title('ram use')
    ram_plot_file_path = par.SAVE_DIR + par.SAVE_NUM + '-stats-ram.png'
    plt.savefig(ram_plot_file_path)
    plt.clf()

    smooth_temp = moving_average(temp, n=smoothing_amount)
    plt.plot(list(range(1, len(smooth_temp) + 1)), smooth_temp)
    plt.xlabel('time in secs')
    plt.ylabel('temp in degrees celcius')
    plt.title('avg cpu core temps')
    temp_plot_file_path = par.SAVE_DIR + par.SAVE_NUM + '-stats-temp.png'
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

def create_save_dir():
    path = Path(par.SAVE_DIR)
    if not path.is_dir():
        os.mkdir(path)
        print('created directory to save runs in:', path)

if __name__ == '__main__':
	# Action Space
    # | Num | Action | Min  | Max |
    # |-----|--------|------|-----|
    # | 0   | Torque | -2.0 | 2.0 |
    # Observation Space
    # The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.
    # | Num | Observation      | Min  | Max |
    # |-----|------------------|------|-----|
    # | 0   | x = cos(theta)   | -1.0 | 1.0 |
    # | 1   | y = sin(angle)   | -1.0 | 1.0 |
    # | 2   | Angular Velocity | -8.0 | 8.0 |
    # Rewards
    # The reward is defined as:
    # r = -(theta^2 + 0.1*theta_dt^2 + 0.001*torque^2)
    # where `theta` is the pendulum's angle normalized between `[-pi, pi]`.
    # set_seeds()
    create_save_dir()
    start_time = time.perf_counter()
    parameter_queue = mp.Queue()
    experience_queue = mp.Queue()

    training_ongoing = mp.Value('i')
    training_ongoing.value = 1
    monitor_queue = mp.Queue()
    monitor_process = mp.Process(target=monitor_fn, args=(training_ongoing, monitor_queue))
    monitor_process.start()
    
    coordinator = mp.Process(target=central_agent, args=(parameter_queue, experience_queue))
    coordinator.start()

    agents = [mp.Process(target=agent, args=(i, parameter_queue, experience_queue)) for i in range(hyp.NUM_AGENTS)]
    for process in agents:
        process.start()

    coordinator.join()
    for process in agents:
        process.join()

    time_taken = time.perf_counter() - start_time

    training_ongoing.value = 0
    # Sometimes the process does not end for some reason, so I am forcibly terminating it
    time.sleep(2)  # allow the process to end on its own
    monitor_process.terminate()
    monitor_process.join()
    save_state(time_taken, monitor_queue)
