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

def actions_plot(path_list, label_list):
    import pandas as pd
    import matplotlib.pyplot as plt
    df_list = [pd.read_csv(path) for path in path_list]
    for df, label in zip(df_list, label_list):
        plt.plot(list(range(1, len(df) + 1 - 999)), moving_average(df['index'].values, 1000), label=label)
    # plt.plot([1, 500000], [1.25, 1.25], 'r--')
    # plt.plot([1, 94000], [2, 2], 'r--')
    plt.xlabel('step (not epoch - this is 10^5 epochs)')
    plt.ylabel('action')
    plt.title('moving average of 1000 over actions across 10^5 epochs')
    plt.legend()
    plt.savefig(r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0018-moving-average-1000-actions-w0-plot.png')
    plt.show()

def scores_plot(path_list, label_list):
    import pandas as pd
    import matplotlib.pyplot as plt
    df_list = [pd.read_csv(path) for path in path_list]
    for df, label in zip(df_list, label_list):
        plt.plot(list(range(1, len(df) + 1 - 999)), moving_average(df['score'].values, 1000), label=label)
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.title(f'moving average of 1000 over score across {len(df)} epochs')
    plt.legend()
    plt.ylim([-50, 50])
    plt.savefig(r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0018-moving-average-1000-score-w0-plot-zoom.png')
    plt.show()


if __name__ == '__main__':
    # dataclassTest()
    # queue_1() 
    # rolling()
    # rolling_2()
    # get_vars()
    # listing()
    # copy_files()
    # curr_file_path()
    # rounding()
    # monitoring()
    # spawn()
    # monitor()
    # numpy_avg()
    # filename()
    # raw_vs_smooth_scores()
    # min_val_a3c()
    # testing_some()
    # fetch_time(100000)
    scores_plot(['/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0018-raw-score-pro-w0.csv'], ['pro'])
    # actions_plot([r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0018-raw-actions-adv-w0.csv', r'/mnt/windows_d/LibraryOfBabel/Projects/A3C/runs_adversary_a3c_pensieve-4/0018-raw-actions-pro-w0.csv'], ['adv', 'pro'])
