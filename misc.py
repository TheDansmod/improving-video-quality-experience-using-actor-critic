from dataclasses import dataclass

@dataclass
class Test:
    a: int = 3
    b: float = 4.32
    c: tuple = (1, 9, 7, 5, 2)
    d: int = len(c)

def dataclassTest():
    print(Test.a)
    print(Test.b)
    print(Test.c)
    print(Test.d)

def queue_1():
    import queue
    q = queue.Queue()
    a = q.get()
    print(a)

def rolling():
    import torch
    a = torch.tensor(list(range(1, 9))).view(4, 2)
    print(a)
    b = torch.roll(a, shifts=-1, dims=-1)
    print(b)

def rolling_2():
    import torch
    a = torch.randint(low=0, high=10, size=(1, 6, 8))
    print(a)
    b = torch.roll(a, shifts=-1, dims=-1)
    print(b)
    print(b.shape)

def get_vars():
    variables = [(name, value) for name, value in vars(Test).items() if not name.startswith('__')]
    print(variables)

def listing():
    import os
    path = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-2/'
    print(len(set([name[0] for name in os.listdir(path)])))

def copy_files():
    import shutil

    src_path = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/misc.py'
    dest_path = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/misc-copy-1.py'
    shutil.copyfile(src_path, dest_path)

def curr_file_path():
    import os
    print(os.path.realpath(__file__))

def rounding():
    print(round(3))
    print(round(3.2))
    print(round(3.51))
    print(round(3.88))

def monitoring():
    import psutil
    import time
    cpu_use = psutil.cpu_percent(interval=0.5, percpu=True)
    cpu_count = psutil.cpu_count()
    virt_mem_used = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    temp = psutil.sensors_temperatures()['coretemp']
    temps = []
    for t in temp:
        if 'Core' in t.label:
            temps.append(t.current)
    print(cpu_use)
    print(cpu_count)
    print(virt_mem_used)
    print(temps)

def printer():
    import numpy
    print(numpy.random.randint(1, 10))

def spawn():
    import multiprocessing as mp
    procs = [mp.Process(target=printer) for i in range(10)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

def big_work():
    import time
    time.sleep(3)

def monitor_fn(control, shared_queue):
    import psutil
    import time
    while control.value != 0:
        virt_mem_used = 7.61 - psutil.virtual_memory().available / pow(1024, 3)
        cpu_use = psutil.cpu_percent(interval=1)
        temp = psutil.sensors_temperatures()['coretemp']
        temps = []
        for t in temp:
            if 'Core' in t.label:
                temps.append(t.current)
        shared_queue.put(cpu_use)
        shared_queue.put(round(virt_mem_used, 2))
        shared_queue.put(temps)

def monitor():
    import multiprocessing as mp
    procs = [mp.Process(target=big_work) for i in range(3)]
    control = mp.Value('i')
    shared_queue = mp.Queue()
    control.value = 1
    monitor_process = mp.Process(target=monitor_fn, args=(control, shared_queue))
    monitor_process.start()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    control.value = 0
    monitor_process.join()
    while not shared_queue.empty():
        print('queue val', shared_queue.get())

def numpy_avg():
    import numpy as np
    a = [1, 2, 3, 4, 5]
    print(np.mean(a))

def filename():
    import os
    from pathlib import Path
    print(Path(os.path.basename(__file__)).stem)
    path1 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/temp_folder'
    path2 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/results'
    print(Path(path1).is_dir())
    print(Path(path2).is_dir())
    if not Path(path1).is_dir():
        os.mkdir(Path(path1))
    print('created ', Path(path1))

def moving_average(a, n):
    import numpy as np
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def raw_vs_smooth_scores():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    path1 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-3-async/0014-raw-score-w0.csv'
    path2 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-3-async/0014-raw-score-w1.csv'
    path3 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-3-async/0014-raw-score-w2.csv'
    path4 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-3-async/0014-raw-score-w3.csv'
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    df4 = pd.read_csv(path4)
    print(np.std(df1['score']))
    print(np.std(df2['score']))
    print(np.std(df3['score']))
    print(np.std(df4['score']))
    score1 = moving_average(df1['score'].values, 200)
    score2 = moving_average(df2['score'].values, 200)
    score3 = moving_average(df3['score'].values, 200)
    score4 = moving_average(df4['score'].values, 200)
    plt.plot(list(range(1, len(score1) + 1)), score1, label='w0')
    plt.plot(list(range(1, len(score1) + 1)), score2, label='w1')
    plt.plot(list(range(1, len(score1) + 1)), score3, label='w2')
    plt.plot(list(range(1, len(score1) + 1)), score4, label='w3')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend()
    plt.ylim([-10,20])
    # plt.show()

def running_average(a):
    import numpy as np
    n = len(a)
    a_csum = np.cumsum(a, dtype=float)
    n_sum = np.cumsum(np.ones(n))
    return a_csum / n_sum

def min_val_a3c():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    path0 = '/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0007-raw-score-pro-w0.csv'
    path1 = '/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0005-raw-score-pro-w0.csv'
    path2 = '/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0006-raw-score-pro-w0.csv'
    df0 = pd.read_csv(path0)
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df = pd.concat([df1, df2])
    scores1 = moving_average(df0['score'].values, 1000)
    scores2 = running_average(df0['score'].values)
    scores3 = moving_average(df['score'].values, 1000)
    scores4 = running_average(df['score'].values)
    plt.plot(list(range(1, len(scores1) + 1)), scores1, label='1000 avg 50k')
    plt.plot(list(range(1, len(scores2) + 1)), scores2, label='runn avg 50k')
    plt.plot(list(range(1, len(scores3) + 1)), scores3, label='1000 avg 20k')
    plt.plot(list(range(1, len(scores4) + 1)), scores4, label='runn avg 20k')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend()
    # plt.ylim([-50, 50])
    plt.show()
    print(len(df0[df0['score'] > 0]))
    print(len(np.where(scores1 > 0)))

    dfA = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8]})
    dfB = pd.DataFrame({
        'A': [10, 11, 12, 13],
        'B': [14, 15, 16, 17]})
    # print(dfA)
    # print(dfB)
    # print(pd.concat([dfA, dfB]))

def plotting_raw(path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    df = pd.read_csv(path)
    score1 = moving_average(df['score'].values, 100)
    score2 = moving_average(df['score'].values, 1000)
    score3 = running_average(df['score'].values)
    plt.plot(list(range(1, len(score1) + 1)), score1, label='mov avg 100')
    plt.plot(list(range(1, len(score2) + 1)), score2, label='mov avg 1000')
    # plt.plot(list(range(1, len(score3) + 1)), score3, label='run avg')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend()
    plt.show()
    plt.plot(list(range(1, len(score1) + 1)), score1, label='mov avg 100')
    plt.plot(list(range(1, len(score2) + 1)), score2, label='mov avg 1000')
    # plt.plot(list(range(1, len(score3) + 1)), score3, label='run avg')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend()
    plt.ylim([-50, 50])
    plt.show()

class Testing_some:
    hello = 'string'

def testing_some():
    print(Testing_some.hello)

def fetch_time(n):
    from datetime import datetime
    import time
    start = time.perf_counter()
    for i in range(n):
        # a = datetime.today().strftime('%d %b %Y %H:%M:%S %p')
        a = time.time()
    end = time.perf_counter()
    print('time taken for', n, end - start)

def adding_time():
    import datetime
    now = datetime.datetime.today()
    SECS_PER_EPOCH = 0.24
    NUM_EPOCHS = 1000
    print(f'based on {SECS_PER_EPOCH} secs / epoch, {NUM_EPOCHS} epochs will take {SECS_PER_EPOCH * NUM_EPOCHS} secs, so will finish at {(datetime.timedelta(seconds=SECS_PER_EPOCH * NUM_EPOCHS) + now).strftime("%d %b %Y %H:%M:%S %p")}')

def bitrates_ratio():
    from statistics import median
    import numpy as np
    bitrates = [300, 750, 1200, 1850, 2850, 4300]
    ratio_sum = [] 
    for i in range(6):
        for j in range(i+1, 6):
            ratio_sum.append(np.log(bitrates[j] / bitrates[i]))
    print(median(ratio_sum))
    print(sum(ratio_sum) / 6)

def central_vs_agent_actions():
    import pandas as pd
    import matplotlib.pyplot as plt
    w0 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-7/0001-raw-score-w0.csv'
    w1 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-7/0001-raw-score-w1.csv'
    w2 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-7/0001-raw-score-w2.csv'
    w3 = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-7/0001-raw-score-w3.csv'
    ca = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-7/0001-raw-score-central-agent.csv'
    vw0 = moving_average(pd.read_csv(w0)['score'].values, 10)
    vw1 = moving_average(pd.read_csv(w1)['score'].values, 10)
    vw2 = moving_average(pd.read_csv(w2)['score'].values, 10)
    vw3 = moving_average(pd.read_csv(w3)['score'].values, 10)
    c = moving_average(pd.read_csv(ca)['score'].values, 10)
    w = (vw0 + vw1 + vw2 + vw3) / 4
    x = list(range(1, len(w) + 1))
    plt.plot(x, w, label='worker')
    plt.plot(x, c, label='centre')
    plt.legend()
    plt.show()

def action_score_combined_plot():
    import pandas as pd
    import matplotlib.pyplot as plt
    act_path = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-5/0038-raw-actions-central-agent.csv'
    scr_path = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-5/0038-raw-score-central-agent.csv'
    act_df = pd.read_csv(act_path)
    scr_df = pd.read_csv(scr_path)
    act_df = (act_df - act_df.mean()) / act_df.std()
    scr_df = (scr_df - scr_df.mean()) / scr_df.std()
    plt.plot(list(range(1, len(act_df) - 98)), moving_average(act_df['action'].values, 100), label='action')
    plt.plot(list(range(1, len(scr_df) - 98)), moving_average(scr_df['score'].values, 100), label='score')
    plt.legend()
    plt.ylim([-2, 2])
    plt.title('action and score normalized for 5k runs')
    plt.show()

def old_new_plot():
    import matplotlib.pyplot as plt
    import pandas as pd
    oldPath = r'/mnt/windows_d/LibraryOfBabel/Projects/Pensieve/pensieve-pytorch/results/log_central'
    newPath = r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_pensieve-5/0046-raw-score-central-agent.csv'
    testPath = r'/mnt/windows_d/LibraryOfBabel/Projects/Pensieve/pensieve-pytorch/results/log_test'
    new_reward = pd.read_csv(newPath)['score'].values
    old_reward, test_reward = [], []
    with open(oldPath, 'r') as f:
        for line in f.readlines():
            if 'INFO:root:Epoch:' in line:
                old_reward.append(float(line.split(' ')[3]))
    with open(testPath, 'r') as f:
        for line in f.readlines():
            test_reward.append(float(line.split('\t')[3]))
    old_reward = moving_average(old_reward, 100)
    new_reward = moving_average(new_reward, 100)
    n_old, n_new, n_test = len(old_reward), len(new_reward), len(test_reward)
    max_old, max_new, max_test = max(old_reward), max(new_reward), max(test_reward)
    plt.plot(list(range(1, n_old + 1)), old_reward, 'b-', label='old_reward')
    # plt.plot(list(range(1, (n_test) * 100, 100)), test_reward, label='test_reward')
    plt.plot(list(range(1, n_new + 1)), new_reward, 'r-', label='new_reward')
    plt.plot([1, 4900], [max_old, max_old], 'b--', label='max_other')
    plt.plot([1, 4900], [max_new, max_new], 'r--', label='max_me')
    # plt.plot([1, 4900], [max_test, max_test], label='max_test')
    plt.title(f'mov avg 100 ; max_other {max_old:.2f} max_me {max_new:.2f}')
    # plt.ylim([-50, 50])
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('avg QoE')
    plt.savefig(r'/mnt/windows_d/LibraryOfBabel/Images/Miscellaneous/thesis/pensieve_pytorch_model0_seed43_vs_me_normal.png')
    # plt.show()


def actions_plot(path_list, label_list, avg_over=1000, save_path=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    df_list = [pd.read_csv(path) for path in path_list]
    for df, label in zip(df_list, label_list):
        plt.plot(list(range(1, len(df) + 1 - avg_over + 1)), moving_average(df['index'].values, avg_over), label=label)
        print(len(df))
    # plt.plot([1, 500000], [1.25, 1.25], 'r--')
    # plt.plot([1, 94000], [2, 2], 'r--')
    epochs = max(map(len, df_list))
    plt.xlabel(f'step (not epoch - this is {epochs} epochs)')
    plt.ylabel('action')
    plt.title(f'moving average of {avg_over} over actions across {epochs} epochs')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def scores_plot(path_list, label_list, avg_over=1000, y_lim_low=-100, y_lim_high=100, save_path=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    df_list = [pd.read_csv(path) for path in path_list]
    max_scores_list = []
    for df, label in zip(df_list, label_list):
        scores = moving_average(df['score'].values, avg_over)
        plt.plot(list(range(1, len(scores) + 1)), scores, label=label)
        max_scores = max(scores)
        plt.plot([1, len(scores) + 1], [max_scores, max_scores], label='max ' + label)
        max_scores_list.append(max_scores)
    epochs = max(map(len, df_list))
    plt.plot([0, epochs], [15, 15], label='y=15')
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.title(f'moving average of {avg_over} over score across {epochs} epochs ; {str(max_scores_list)}')
    plt.legend()
    plt.ylim([y_lim_low, y_lim_high])
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    old_new_plot()
