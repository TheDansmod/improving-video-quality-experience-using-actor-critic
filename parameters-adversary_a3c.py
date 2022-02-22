import torch.nn.functional as F
import torch
import os
from pathlib import Path
import load_trace  # this is a file load_trace.py

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
PARAM_FILE_PATH = Path(__file__).absolute()
INIT_DIFF_LVL = 0.001
FINAL_DIFF_LVL = 0.1
OFFSET = pow(10, 5)  # this is added to the pro reward to ensure it is always positive so that the protagonist and adversary are optimizing the right objective. I have put this large value since I have seen min rewards of -27k
SAVE_DIR = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/'
save_num = len(set([name[:4] for name in os.listdir(SAVE_DIR)])) + 1 if Path(SAVE_DIR).exists() else 1
SAVE_NUM = '0' * (4 - len(str(save_num))) + str(save_num)
RAW_SCORE_FILE = SAVE_DIR + SAVE_NUM + '-raw-score'
RAW_ACTIONS_FILE = SAVE_DIR + SAVE_NUM + 'raw-actions'
CURR_FILE_PATH = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/adversary_a3c_pensieve-4.py'
ADV_ACTOR_PATH = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0005-adv-chkpt-actor-10000'
PRO_ACTOR_PATH = ADV_ACTOR_PATH.replace('-adv-', '-pro-')
ADV_CRITIC_PATH = ADV_ACTOR_PATH.replace('-actor-', '-critic-')
PRO_CRITIC_PATH = PRO_ACTOR_PATH.replace('-actor-', '-critic-')
LOAD_MODEL = False
EPOCH_START = int(PRO_ACTOR_PATH.split('-')[-1]) if LOAD_MODEL else 0
QOE_METRIC = 'lin' # other options are 'log' and 'hd'
NUM_EPOCHS = 200
PRINT_EVERY = NUM_EPOCHS // 10
SMOOTHING_AMT = NUM_EPOCHS // 10
MODEL_SAVE_EVERY = NUM_EPOCHS // 5
USE_GPU = False
DEVICE = 'cuda:0' if (USE_GPU and torch.cuda.is_available()) else 'cpu'
NUM_AGENTS = 4
DO_PROFILING = False
SET_SEEDS = False
MONITOR_PC_STATS = True

# variable_names =  ['S_INFO', 'S_LEN', 'VID_BITRATES', 'A_DIM', 'HD_REWARD', 'BUFFER_NORM_FACTOR', 'CHUNK_TILL_VIDEO_END_CAP', 'DEFAULT_QUALITY', 'CONV_IN_CHANNELS', 'NN_IN_FEATURES', 'NUM_EPOCHS', 'ACTOR_LR', 'CRITIC_LR', 'NUM_AGENTS', 'TRAIN_SEQ_LEN', 'RMSPROP_ALPHA', 'RMSPROP_EPS', 'NUM_HID_NEURONS', 'NUM_CONV_CHANNELS', 'CONV_KERNEL_SIZE', 'CONV_STRIDE', 'CONV_PADDING', 'NN_ACT_FN', 'DISCOUNT_FACTOR', 'INIT_ENTROPY_WT', 'FINAL_ENTROPY_WT', 'LIN_REBUF_PENALTY', 'LOG_REBUF_PENALTY', 'HD_REBUF_PENALTY', 'SMOOTH_PENALTY', 'QOE_METRIC', 'LOAD_MODEL', 'PRO_ACTOR_PATH', 'PRO_CRITIC_PATH', 'ADV_ACTOR_PATH', 'ADV_CRITIC_PATH', 'USE_GPU', 'DEVICE', 'PRINT_EVERY', 'MODEL_SAVE_EVERY', 'DATA_PATH', 'SMOOTHING_AMT', 'SAVE_DIR', 'SAVE_NUM', 'CURR_FILE_PATH', 'EPOCH_START', 'RAW_SCORE_FILE', 'DO_PROFILING', 'SET_SEEDS', 'PARAM_FILE_PATH', 'MONITOR_PC_STATS', 'INIT_DIFF_LVL', 'FINAL_DIFF_LVL']
# variables = [S_INFO, S_LEN, VID_BITRATES, A_DIM, HD_REWARD, BUFFER_NORM_FACTOR, CHUNK_TILL_VIDEO_END_CAP, DEFAULT_QUALITY, CONV_IN_CHANNELS, NN_IN_FEATURES, NUM_EPOCHS, ACTOR_LR, CRITIC_LR, NUM_AGENTS, TRAIN_SEQ_LEN, RMSPROP_ALPHA, RMSPROP_EPS, NUM_HID_NEURONS, NUM_CONV_CHANNELS, CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING, NN_ACT_FN, DISCOUNT_FACTOR, INIT_ENTROPY_WT, FINAL_ENTROPY_WT, LIN_REBUF_PENALTY, LOG_REBUF_PENALTY, HD_REBUF_PENALTY, SMOOTH_PENALTY, QOE_METRIC, LOAD_MODEL, PRO_ACTOR_PATH, PRO_CRITIC_PATH, ADV_ACTOR_PATH, ADV_CRITIC_PATH, USE_GPU, DEVICE, PRINT_EVERY, MODEL_SAVE_EVERY, DATA_PATH, SMOOTHING_AMT, SAVE_DIR, SAVE_NUM, CURR_FILE_PATH, EPOCH_START, RAW_SCORE_FILE, DO_PROFILING, SET_SEEDS, PARAM_FILE_PATH, MONITOR_PC_STATS, INIT_DIFF_LVL, FINAL_DIFF_LVL]
