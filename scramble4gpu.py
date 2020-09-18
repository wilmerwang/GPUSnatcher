# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import torch


def parse(qargs, results):
    result_np = []
    for line in results[1:]:
        result_np.append([''.join(filter(str.isdigit, word)) for word in line.split(',')])
    result_np = np.array(result_np)

    return result_np


def query_gpu(*args):
    qargs = ['index', 'memory.free', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv, noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()

    return parse(qargs, results), results[0].strip()


class GPUManager(object):
    def __init__(self, *args):
        self._args = args

    def choose_free_gpu(self, num=1):
        qresult, qindex = query_gpu(*self._args)
        qresult = qresult.astype('int')

        if qresult.shape[0] < num:
            print('The number GPU {} < num {}'.format(len(qresult), num))
        else:
            qresult_sort_index = np.argsort(-qresult[:, 1])
            idex = [i for i in qresult_sort_index[:num] if qresult[i][1]/qresult[i][2] > 0.8]
            gpus_index = qresult[:, 0][idex]

            return gpus_index


# if __name__ == '__main__':
def main():
    gpu_manager = GPUManager()
    gpus_free = gpu_manager.choose_free_gpu(num=1)

    if len(gpus_free) > 0:
        for gpus_id in gpus_free:
            print("Scramble GPU {}".format(gpus_id))
            torch.zeros([1000, 1000, 1000], dtype=torch.double, device=gpus_id)
        time.sleep(60000)

    else:
        print("No")


if __name__ == '__main__':
    while True:
        main()
