import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0 # how many bits in one megabit. the throughput data is given in megabits
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48  # EXP: in dash, at least one chunk must be in buffer and so this is 48 not 49
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec  EXP: when the buffer is full, must sleep in intervals of 500ms
PACKET_PAYLOAD_PORTION = 0.95  # EXP: i think this decides what fraction of all the data transmitted over a given span of time was actual data, the rest is just metadeta like headers and handshakes
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = '/mnt/windows_d/Svalbard/Data/Thesis/pensieve_data/video_size_'
# EXP: each video size file corresponds to the folder video_server/videoX/ where X is 1-6 inclusive. in video_size_X as X increases from 0 to 5, the size of the chunk increases. While in videoX folder, as the value of X increases from 1 to 6, the size of the video decreases. I think this might be because the original MIT people must have given the number based on quality where 1 is best quality and 6 is worst while the pytorch guy gave the numbers based on size where 0 is smallest and 5 is highest


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)  # THIS IS IMPORTANT PASS THIS

        self.all_cooked_time = all_cooked_time  # list of list of times at which bandwidth was recorded
        self.all_cooked_bw = all_cooked_bw  # list of list of bandwidth corresponding to the times

        self.video_chunk_counter = 0  # EXP: how many video chunks have been sent
        self.buffer_size = 0  # EXP: should be in milliseconds

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx] # list of times
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]  # list of bandwidths

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]  # DouBT: aren't you actually selecting a random end time rather than a random start time
        # EXP: the last_mahimahi_time is subtracted from the bandwidth times to find out the duration. So, it is actually a random start time. we keep incrementing the mahimahi_ptr until it reaches the end

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):  # EXP: there are 6 bitrate levels -- goes 0, 1, 2, 3, 4, 5
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:  # EXP: each file video_size_X has 49 lines each with an integer in it
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))
        # EXP: at the end you have
        # EXP: {0: [... list with 49 elements ...], 
        # EXP:  1: [... list with 49 elements ...],
        # EXP:  2: [... list with 49 elements ...],
        # EXP:  3: [... list with 49 elements ...],
        # EXP:  4: [... list with 49 elements ...],
        # EXP:  5: [... list with 49 elements ...]}
        # EXP: which gives the sizes in bytes of 49 different video chunks available at 6 different bitrates
        # EXP: i think it is a single video with 49 chunks at 6 different bitrates


    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]  # in bytes
        
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes 
        # EXP: counts how many bytes have been sent from the current chunk
        
        # EXP: loop through the datapoints of throughput until enough data is transmitted to transfer the current chunk
        while True:  # download video chunk over mahimahi
            # the throughput is in MegaBits per second and the video size is in bytes
            throughput = self.cooked_bw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE  # DouBT: as far as I can tell they want to convert from MB (which is the units for the internet speed) to bits, but they do not seem to be doing it correctly
            # EXP: they actually are converting from megabits per second to bytes per second since the video chunk size is in bytes
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time  # last_mahimahi_time is the time at which the previous chunk finished being transmitted and the current chunk began being transmitted
	    
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION  # EXP: PORTION=0.95 so i think they assume that 5% of the data transmitted is headers. 
            # EXP: the packet_payload is in bytes since the throughput is in bytes per second and the duration is in seconds

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                # EXP: above line checks if more than the current chunk was sent in current packet
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / throughput / PACKET_PAYLOAD_PORTION  # finds out for how long the relevant packet was being transmitted
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break
            
            # EXP: if the full video was not able to be transmitted in the given packet then do the below stuff
            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

	    # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)  # EXP: LOW=0.9, HIGH=1.1, just a 10% noise at most
        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)  # EXP: in milliseconds

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH  # EXP: how much you need to drain
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME  # EXP: DRAIN_BUFFER_SLEEP_TIME is 500 ms , this line means that we sleep only in multiples of 500 ms
            self.buffer_size -= sleep_time

            # EXP: we move over throughput datapoints for the duration for which we are sleeping
            while True:
                duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
        # EXP: this means that a given chunk size corresponds to a given bitrate out of the six available bitrates

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain
        # EXP: delay               -> how long it took to transmit the current chunk of video in milliseconds
        # DBT: sleep_time          -> ??????
        # EXP: ret_buf_size / 1000 -> what is the size of the remaining buffer in seconds
        # EXP: rebuf / 1000        -> how long the video was rebuffering for in seconds
        # EXP: video_chunk_size    -> size of the video chunk that was just transmitted in bytes
        # EXP: next_video_chunk_sz -> 6 diff possible sizes at which the next video chunk could be got one for each of the 6 bitrates
        # EXP: end_of_video        -> whether it is the last video chunk (boolean value)
        # EXP: video_chunk_remain  -> how many video chunks still need to be downloaded

    def setEnvironmentPtr(self,ptrTraceIndex,ptrIndex):
        self.trace_idx=ptrTraceIndex
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        self.video_chunk_counter = 0
        self.buffer_size = 0


        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = ptrIndex % len(self.cooked_bw)
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]


