import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class Task:
    def __init__(self, start, elapsed, lbl):
        self.start = start
        self.elapsed = elapsed
        self.lbl = lbl

    def normalize_start(self, min_time):
        self.start = self.start - min_time

#    Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput           Device   Context    Stream  Name
#    254.96ms  1.0502ms                    -               -         -         -         -  6.2915MB  5.9908GB/s  Tesla M2050 (0)         1        13  [CUDA memcpy HtoD]
#    256.01ms  1.0527ms                    -               -         -         -         -  6.2915MB  5.9766GB/s  Tesla M2050 (0)         1        14  [CUDA memcpy HtoD]
#    257.07ms  1.0522ms                    -               -         -         -         -  6.2915MB  5.9792GB/s  Tesla M2050 (0)         1        15  [CUDA memcpy HtoD]
#    258.13ms  1.0526ms                    -               -         -         -         -  6.2915MB  5.9772GB/s  Tesla M2050 (0)         1        16  [CUDA memcpy HtoD]
#    259.18ms  1.0518ms                    -               -         -         -         -  6.2915MB  5.9817GB/s  Tesla M2050 (0)         1        13  [CUDA memcpy HtoD]
#    260.24ms  1.0527ms                    -               -         -         -         -  6.2915MB  5.9765GB/s  Tesla M2050 (0)         1        14  [CUDA memcpy HtoD]
#    261.30ms  1.0521ms                    -               -         -         -         -  6.2915MB  5.9799GB/s  Tesla M2050 (0)         1        15  [CUDA memcpy HtoD]
#    262.35ms  1.0524ms                    -               -         -         -         -  6.2915MB  5.9783GB/s  Tesla M2050 (0)         1        16  [CUDA memcpy HtoD]
#    263.41ms  1.0518ms                    -               -         -         -         -  6.2915MB  5.9816GB/s  Tesla M2050 (0)         1        13  [CUDA memcpy HtoD]
#    264.46ms  1.0556ms                    -               -         -         -         -  6.2915MB  5.9602GB/s  Tesla M2050 (0)         1        14  [CUDA memcpy HtoD]
#    264.47ms  1.3106ms          (49152 1 1)       (128 1 1)        13        0B        0B         -           -  Tesla M2050 (0)         1        13  vector_add(int*, int*, int*, int, int) [118]
#    265.52ms  1.0532ms                    -               -         -         -         -  6.2915MB  5.9737GB/s  Tesla M2050 (0)         1        15  [CUDA memcpy HtoD]
#    265.78ms  1.3132ms          (49152 1 1)       (128 1 1)        13        0B        0B         -           -  Tesla M2050 (0)         1        14  vector_add(int*, int*, int*, int, int) [125]
#    266.58ms  1.0551ms                    -               -         -         -         -  6.2915MB  5.9629GB/s  Tesla M2050 (0)         1        16  [CUDA memcpy HtoD]
#    267.09ms  1.3019ms          (49152 1 1)       (128 1 1)        13        0B        0B         -           -  Tesla M2050 (0)         1        15  vector_add(int*, int*, int*, int, int) [132]
#    268.39ms  1.2832ms          (49152 1 1)       (128 1 1)        13        0B        0B         -           -  Tesla M2050 (0)         1        16  vector_add(int*, int*, int*, int, int) [139]
#    269.68ms  972.68us                    -               -         -         -         -  6.2915MB  6.4682GB/s  Tesla M2050 (0)         1        13  [CUDA memcpy DtoH]
#    270.66ms  973.51us                    -               -         -         -         -  6.2915MB  6.4627GB/s  Tesla M2050 (0)         1        14  [CUDA memcpy DtoH]
#    271.64ms  972.45us                    -               -         -         -         -  6.2915MB  6.4697GB/s  Tesla M2050 (0)         1        15  [CUDA memcpy DtoH]
#    272.61ms  972.90us                    -               -         -         -         -  6.2915MB  6.4667GB/s  Tesla M2050 (0)         1        16  [CUDA memcpy DtoH]

colors = [ 'r', 'y', 'b', 'g', 'c', 'm' ]

min_timestamp = 254.96
max_timestamp = 272.61 + 0.97290

operations = []
operations.append((254.96, 1.0502,  13,  '[CUDA memcpy HtoD]'))
operations.append((256.01, 1.0527,  14,  '[CUDA memcpy HtoD]'))
operations.append((257.07, 1.0522,  15,  '[CUDA memcpy HtoD]'))
operations.append((258.13, 1.0526,  16,  '[CUDA memcpy HtoD]'))
operations.append((259.18, 1.0518,  13,  '[CUDA memcpy HtoD]'))
operations.append((260.24, 1.0527,  14,  '[CUDA memcpy HtoD]'))
operations.append((261.30, 1.0521,  15,  '[CUDA memcpy HtoD]'))
operations.append((262.35, 1.0524,  16,  '[CUDA memcpy HtoD]'))
operations.append((263.41, 1.0518,  13,  '[CUDA memcpy HtoD]'))
operations.append((264.46, 1.0556,  14,  '[CUDA memcpy HtoD]'))
operations.append((264.47, 1.3106,  13,  'vector_add(int*, int*, int*, int, int)'))
operations.append((265.52, 1.0532,  15,  '[CUDA memcpy HtoD]'))
operations.append((265.78, 1.3132,  14,  'vector_add(int*, int*, int*, int, int)'))
operations.append((266.58, 1.0551,  16,  '[CUDA memcpy HtoD]'))
operations.append((267.09, 1.3019,  15,  'vector_add(int*, int*, int*, int, int)'))
operations.append((268.39, 1.2832,  16,  'vector_add(int*, int*, int*, int, int)'))
operations.append((269.68, 0.97268, 13,  '[CUDA memcpy DtoH]'))
operations.append((270.66, 0.97351, 14,  '[CUDA memcpy DtoH]'))
operations.append((271.64, 0.97245, 15,  '[CUDA memcpy DtoH]'))
operations.append((272.61, 0.97290, 16,  '[CUDA memcpy DtoH]'))

nstreams = 4

streams = {}

for op in operations:
    stream = op[2]
    if stream not in streams:
        streams[stream] = []
    streams[stream].append(op)

labels = []
fig = plt.figure(num=0, figsize=(18, 6), dpi=80)
width = 0.35       # the width of the bars: can also be len(x) sequence
ind = 0
for stream in sorted(streams.keys()):
    labels.append(str(stream))

    for op in streams[stream]:
        c = None
        if op[3] == '[CUDA memcpy HtoD]':
            c = 'r'
        elif op[3] == 'vector_add(int*, int*, int*, int, int)':
            c = 'b'
        elif op[3] == '[CUDA memcpy DtoH]':
            c = 'g'
        else:
            raise 'Unknown label ' + op[3]

        plt.barh(ind, op[1], height=width, left=op[0] - min_timestamp, linewidth=1, color=c)

    ind = ind + width

plt.ylabel('Streams')
plt.xlabel('Time (ms)')
plt.yticks(np.arange(0, nstreams, width) + width/2.,
           labels)
# plt.xticks(np.arange(min_timestamp, max_timestamp, 10000))
# plt.axis([ min_timestamp, max_timestamp, 0, 5 ])
plt.axis([ 0, max_timestamp-min_timestamp, 0, ind ])
# plt.legend( (p1[0], p2[0]), ('Men', 'Women') )
plt.show()
