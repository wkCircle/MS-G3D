import numpy as np 
from numpy.lib.format import open_memmap
import pickle 
from pathlib import Path 
import os, re
import argparse
import itertools as it 


def kinetics_minigen(ratio: float, data_path: str, data_out_path: str): 
    data = np.load(data_path, mmap_mode='r')
    r = int(len(data) * ratio)
    fp_data = open_memmap( # link disk to RAM to save I/O operations in large file.
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(r, *data.shape[1:])
    )
    fp_data[:] = data[:r, ...]
    print(fp_data.shape)

def kinetics_minilabelgen(ratio: float, label_path: str, label_out_path: str): 
    with open(label_path, 'rb') as fr: 
        label = pickle.load(fr)
    assert len(label) == 2 
    assert len(label[0]) == len(label[1])
    r = int(len(label[0]) * ratio)
    mini_label = (label[0][:r], label[1][:r])
    with open(label_out_path, 'wb') as fw: 
        pickle.dump(mini_label, fw)


if __name__ == "__main__": 
    project_dir = re.search(".*MS-G3D", os.getcwd())[0]
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton MINI Data Converter.')
    parser.add_argument(
        '--ratio', default=0.001)
    parser.add_argument(
        '--data_path', default= project_dir + '/data/kinetics')
    parser.add_argument(
        '--out_folder', default= project_dir + '/data/kinetics_mini')
    arg = parser.parse_args()

    partI = ['val', 'train']
    partII = ['joint', 'bone']
    for p, pp in it.product(partI, partII): 
        print('kinetics ', p, pp)
        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        data_path = '{}/{}_data_{}.npy'.format(arg.data_path, p, pp)
        data_out_path = '{}/{}_data_mini_{}.npy'.format(arg.out_folder, p, pp)
        kinetics_minigen(arg.ratio, data_path, data_out_path)
    
    for p in partI: 
        print('kinetics', p, 'label')
        label_path = '{}/{}_label.pkl'.format(arg.data_path, p)
        label_out_path = '{}/{}_mini_label.pkl'.format(arg.out_folder, p)
        kinetics_minilabelgen(arg.ratio, label_path, label_out_path)
        