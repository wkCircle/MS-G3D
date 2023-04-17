import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

ntu_skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
) # len 25: note the additional (21, 21) compared to its graph

bone_pairs = {
    'ntu/xview': ntu_skeleton_bone_pairs,
    'ntu/xsub': ntu_skeleton_bone_pairs,

    # NTU 120 uses the same skeleton structure as NTU 60
    'ntu120/xsub': ntu_skeleton_bone_pairs,
    'ntu120/xset': ntu_skeleton_bone_pairs,

    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
        (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    ) # len 18: note the dummy node (0, 0)
}

benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    'ntu120': ('ntu120/xset', 'ntu120/xsub'),
    'kinetics': ('kinetics',)
}

parts = { 'train', 'val' }

# The reason of having dummy nodes such as (0,0) in kenetics and (21, 21) in ntu is that the author 
# directly modify the joint array of shape V=num_joint and get bone array of the same shape. To achieve 
# this, num_joints should be equal length of num_bones. 
# In fact, users can also define the bone connections from scratch and doesn't not have to follow 
# num_bone_connections = num_joints - 1. But then the bone array should also be generated from scratch.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NTU60/NTU120/Kinetics')
    parser.add_argument('--dataset', choices=['ntu', 'ntu120', 'kinetics'], required=True)
    args = parser.parse_args()

    for benchmark in benchmarks[args.dataset]:
        for part in parts:
            print(benchmark, part)
            try:
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                N, C, T, V, M = data.shape
                fp_sp = open_memmap( # link disk to RAM to save I/O operations in large file.
                    '../data/{}/{}_data_bone.npy'.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 3, T, V, M))

                fp_sp[:, :C, :, :, :] = data
                for v1, v2 in tqdm(bone_pairs[benchmark]): # store bone as a 2-dim vector (x1-x2, y1-y2). store the differenced score.
                    if benchmark != 'kinetics': # the same as in the ntu graph
                        v1 -= 1
                        v2 -= 1
                    fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
            except Exception as e:
                print(f'Run into error: {e}')
                print(f'Skipping ({benchmark} {part})')