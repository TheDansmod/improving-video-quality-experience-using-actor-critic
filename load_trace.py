import os

# returns three lists
# all_cooked_time is a list of lists. Each list within it contains the times at which bandwidth was recorded
# all_cooked_bw is a list of lists. Each list within it contains the bandwidth recorded at the times mentioned in the all_cooked_time
# Each of the lists in all_cooked_time and all_cooked_bw correspond to a different file in the data folder
# all_file_names is a list of all files from which data has been extracted

COOKED_TRACE_FOLDER = './cooked_traces/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names
