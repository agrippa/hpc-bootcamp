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
#    223.36ms  702.00us                    -               -         -         -         -  4.1943MB  5.9748GB/s  Tesla M2050 (0)         1        13  [CUDA memcpy HtoD]
#    224.07ms  704.47us                    -               -         -         -         -  4.1943MB  5.9539GB/s  Tesla M2050 (0)         1        14  [CUDA memcpy HtoD]
#    224.78ms  702.90us                    -               -         -         -         -  4.1943MB  5.9671GB/s  Tesla M2050 (0)         1        15  [CUDA memcpy HtoD]
#    225.49ms  704.05us                    -               -         -         -         -  4.1943MB  5.9574GB/s  Tesla M2050 (0)         1        16  [CUDA memcpy HtoD]
#    226.19ms  703.54us                    -               -         -         -         -  4.1943MB  5.9617GB/s  Tesla M2050 (0)         1        13  [CUDA memcpy HtoD]
#    226.90ms  704.85us                    -               -         -         -         -  4.1943MB  5.9506GB/s  Tesla M2050 (0)         1        14  [CUDA memcpy HtoD]
#    227.61ms  703.25us                    -               -         -         -         -  4.1943MB  5.9642GB/s  Tesla M2050 (0)         1        15  [CUDA memcpy HtoD]
#    228.32ms  703.73us                    -               -         -         -         -  4.1943MB  5.9601GB/s  Tesla M2050 (0)         1        16  [CUDA memcpy HtoD]
#    229.02ms  703.25us                    -               -         -         -         -  4.1943MB  5.9642GB/s  Tesla M2050 (0)         1        13  [CUDA memcpy HtoD]
#    229.73ms  705.01us                    -               -         -         -         -  4.1943MB  5.9493GB/s  Tesla M2050 (0)         1        14  [CUDA memcpy HtoD]
#    229.74ms  759.78us          (32768 1 1)       (128 1 1)        13        0B        0B         -           -  Tesla M2050 (0)         1        13  vector_add(int*, int*, int*, int, int) [118]
#    230.44ms  707.57us                    -               -         -         -         -  4.1943MB  5.9278GB/s  Tesla M2050 (0)         1        15  [CUDA memcpy HtoD]
#    230.49ms  759.15us          (32768 1 1)       (128 1 1)        13        0B        0B         -           -  Tesla M2050 (0)         1        14  vector_add(int*, int*, int*, int, int) [125]
#    231.15ms  706.32us                    -               -         -         -         -  4.1943MB  5.9382GB/s  Tesla M2050 (0)         1        16  [CUDA memcpy HtoD]
#    231.25ms  759.27us          (32768 1 1)       (128 1 1)        13        0B        0B         -           -  Tesla M2050 (0)         1        15  vector_add(int*, int*, int*, int, int) [132]
#    232.01ms  744.81us          (32768 1 1)       (128 1 1)        13        0B        0B         -           -  Tesla M2050 (0)         1        16  vector_add(int*, int*, int*, int, int) [139]
#    232.76ms  644.69us                    -               -         -         -         -  4.1943MB  6.5059GB/s  Tesla M2050 (0)         1        13  [CUDA memcpy DtoH]
#    233.41ms  645.75us                    -               -         -         -         -  4.1943MB  6.4953GB/s  Tesla M2050 (0)         1        14  [CUDA memcpy DtoH]
#    234.06ms  647.64us                    -               -         -         -         -  4.1943MB  6.4763GB/s  Tesla M2050 (0)         1        15  [CUDA memcpy DtoH]
#    234.71ms  645.75us                    -               -         -         -         -  4.1943MB  6.4953GB/s  Tesla M2050 (0)         1        16  [CUDA memcpy DtoH]

colors = [ 'r', 'y', 'b', 'g', 'c', 'm' ]

min_timestamp = 223.36
max_timestamp = 234.71 + 0.64575

operations = []
operations.append((223.36, 0.70200,  13,  '[CUDA memcpy HtoD]'))
operations.append((224.07, 0.70447,  14,  '[CUDA memcpy HtoD]'))
operations.append((224.78, 0.70290,  15,  '[CUDA memcpy HtoD]'))
operations.append((225.49, 0.70405,  16,  '[CUDA memcpy HtoD]'))
operations.append((226.19, 0.70354,  13,  '[CUDA memcpy HtoD]'))
operations.append((226.90, 0.70485,  14,  '[CUDA memcpy HtoD]'))
operations.append((227.61, 0.70325,  15,  '[CUDA memcpy HtoD]'))
operations.append((228.32, 0.70373,  16,  '[CUDA memcpy HtoD]'))
operations.append((229.02, 0.70325,  13,  '[CUDA memcpy HtoD]'))
operations.append((229.73, 0.70501,  14,  '[CUDA memcpy HtoD]'))
operations.append((229.74, 0.75978,  13,  'vector_add(int*, int*, int*, int, int)'))
operations.append((230.44, 0.70757,  15,  '[CUDA memcpy HtoD]'))
operations.append((230.49, 0.75915,  14,  'vector_add(int*, int*, int*, int, int)'))
operations.append((231.15, 0.70632,  16,  '[CUDA memcpy HtoD]'))
operations.append((231.25, 0.75927,  15,  'vector_add(int*, int*, int*, int, int)'))
operations.append((232.01, 0.74481,  16,  'vector_add(int*, int*, int*, int, int)'))
operations.append((232.76, 0.64469,  13,  '[CUDA memcpy DtoH]'))
operations.append((233.41, 0.64575,  14,  '[CUDA memcpy DtoH]'))
operations.append((234.06, 0.64764,  15,  '[CUDA memcpy DtoH]'))
operations.append((234.71, 0.64575,  16,  '[CUDA memcpy DtoH]'))

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
