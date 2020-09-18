# -*- coding: utf-8 -*-
import os
import time
import argparse

import numpy as np
try:
    import torch
except ImportError:
    try:
        import tensorflow as tf
    except ImportError:
        print("No pytorch and tensorflow module")


def set_parser():
    parser = argparse.ArgumentParser(description='..')
    parser.add_argument('-p', '--proportion', type=float, default=0.8, 
        help='The ratio of gpu free memory to total memory')
    parser.add_argument('-n', '--gpu_nums', type=int, default=1,
        help='The numbers of GPU to scramble')
    parser.add_argument('-t', '--times', type=int, default=60000,
        help='Sleep time if scramble gpu')
    args = parser.parse_args()

    return args


def parse(qargs, results):
    result_np = []
    for line in results[1:]:
        result_np.append([''.join(filter(str.isdigit, word)) for word in line.split(',')])
    result_np = np.array(result_np)

    return result_np


def query_gpu():
    qargs = ['index', 'memory.free', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv, noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()

    return parse(qargs, results), results[0].strip()


class GPUManager(object):
    def __init__(self, args):
        self._args = args

    def choose_free_gpu(self, num=1):
        qresult, qindex = query_gpu()
        qresult = qresult.astype('int')

        if qresult.shape[0] < num:
            print('The number GPU {} < num {}'.format(len(qresult), num))
        else:
            qresult_sort_index = np.argsort(-qresult[:, 1])
            idex = [i for i in qresult_sort_index[:num] if qresult[i][1]/qresult[i][2] > self._args.proportion]
            gpus_index = qresult[:, 0][idex]
            gpus_memory = qresult[:, 1][idex]

            return gpus_index, gpus_memory


def compute_storage_size(memory):
    return pow(memory * 1024 * 1024 / 8, 1/3) * 0.9


# if __name__ == '__main__':
def main():
    args = set_parser()

    gpu_manager = GPUManager(args)
    gpus_free, gpus_memory = gpu_manager.choose_free_gpu(num=args.gpu_nums)

    sizes = [int(compute_storage_size(i)) for i in gpus_memory]

    if len(gpus_free) > 0:
        for gpus_id, size in zip(gpus_free, sizes):
            print("Scramble GPU {}".format(gpus_id))
            try:
                torch.zeros([size, size, size], dtype=torch.double, device=gpus_id)
            except:
                # with tf.device('/gpu:{}'.format(gpus_id)):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_id)
                tf.zeros([size, size, size], dtype=tf.dtypes.float64)
        time.sleep(args.times)

    else:
        print()


if __name__ == '__main__':
    while True:
        main()
