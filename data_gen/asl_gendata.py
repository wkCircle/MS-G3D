import os, re
import json
import pickle
import argparse
from typing import List 
import pandas as pd 
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
# from scipy.spatial.transform import Rotation as R

def load_relevant_data_subset(pq_path, ROWS_PER_FRAME: int=543):
    """
    Official function to load each video as the model input.
    Order: face[0:468](len=468), left_hand[468:489](len=21), pose[489:522](len=32), right_hand[522:543](len=21)
    """
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    `reference`_: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    # cross of all zeros only occurs on identical directions
    if np.abs(v).sum() < 1e-4: 
        return np.eye(3) 
    else: #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))


class Feeder_asl(Dataset):
    
    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_hands_frames=True,
                 ignore_window=3, 
                 max_frame=126,
                 num_joint=50, 
                 pad_repeat=True,  
                 num_person_in=1,
                 num_person_out=1):
        """
        The class read and preprocess every video, including: 
            
            - read each video file
            - drop the current hands-empty frame if its next n(=window_shape) frames are all hands-empty, too. (local hands-empty frames deletion)
            - ffill lips (usually lips don't move much unlike hands) as long as the pose is detected in the frame. (luckily no missing pose throughout the dataset.)
            - rotate, shift, and scale lips to make it align with the pose mouth (idx 9, 10)
            - shift and scale both hands to make it align with the pose thumbs as well as wrists (idx 22, 16, 21, 15)
            - (based on the previous two bullet points, all body parts are now united in each frame.)
            - select only partial landmarks: lips (innerlips, 10), left hand (21), right hand (21), partial pose (13, please refer to the source code).
            - repeatedly pad the video till max_frame(=126) with existing frames (from frame 0), so it is not a zero padding but a cyclic padding.
            - for each video, calculate the mean as well as the std of x,y,z and normalize the video.
            - if any missing values still present, fillna with 0. 

        Args:
            data_path (str): data dir.
            label_path (str): label file.
            ignore_empty_hands_frames (bool, optional): _description_. Defaults to True.
            ignore_window (int, optional): decide the local range of a frame and drop it when all local frames are emptry-hands. (Always look ahead, i.e., t=1-7 when current frame is t=0.) Defaults to 7. The argument is valid only when ``ignore_emptry_hands_frames==True``.
            pad_repeat (bool, optional): whether to repeat video to fill the missing frames till ``max_frame``.
            max_frame (int, optional): the output length.
            num_joint (int, optional): the total number of joints for one of the output dimension.
            num_person_in (int, optional): not used. Defaults to 1.
            num_person_out (int, optional): not used. Defaults to 1.
        """
        self.data_path = Path(data_path)
        self.label_path = Path(label_path)
        self.ignore_empty_hands_frames = ignore_empty_hands_frames
        self.ignore_window=ignore_window
        self.max_frame = max_frame
        self.pad_repeat = pad_repeat
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        
        self.load_data(max_frame, num_joint)
        self.load_lips_hands_pose()

    def load_data(self, max_frame: int, num_joint: int):

        # load label: assume the sign2idx is of the same dir as the label_path
        sign2idx = pd.read_json(
            self.label_path.parent / "sign_to_prediction_index_map.json", typ="series"
        ).to_dict()
        self.label_info = pd.read_csv(self.label_path).sort_index().reset_index(drop=True)
        self.label_info["sign_idx"] = self.label_info["sign"].map(sign2idx)

        # output data shape (N, C, T, V, M)
        self.N = len(self.label_info)  # sample
        self.C = 3          # channel x,y,z coordinates
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person
        if self.M != 1: 
            raise NotImplementedError("the data should have M=1.")

    def load_lips_hands_pose(self): 
        # lips 
        # lipsUpperOuter =  [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        # lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        self.lips = lipsUpperInner[::5] + lipsLowerInner[::-5][1:-1]
        assert (78 in self.lips) & (308 in self.lips), "78 & 308 are requried (left/right-most anchor points)."
        # hands 
        lhand_anchor, rhand_anchor = 468, 522
        self.lhand = list(range(0+lhand_anchor, 21+lhand_anchor))
        self.rhand = list(range(0+rhand_anchor, 21+rhand_anchor))
        # pose 
        pose_anchor = 489
        # ear_eye_nose = [8, 6, 4, 0, 1, 3, 7]
        upper_body = [22, 16, 14, 12, 11, 13, 15, 21] # 22, 21 are thumbs and 16, 15 are wrists
        self.pose_all = list(range(0+pose_anchor, 33+pose_anchor))
        self.pose = list(np.array(upper_body[2:-2], dtype=int) + pose_anchor)

        # assertion 
        total = len(self.lips + self.lhand + self.pose + self.rhand)
        assert self.V == total, (
            "the required num_joint {} is not aligned with the in-built ".format(self.V) + 
            "number of landmarks {}. Check the bug!".format(total)
        )

    def handle_lips(self, frame: np.ndarray):
        # rotate, shift, and scale lips (lips idx 308, 78 vs pose idx 9,10.)
        s1, s2 = frame[self.lips[0]], frame[self.lips[2]]
        d1, d2 = frame[self.pose_all[10]], frame[self.pose_all[9]]
        r = rotation_matrix_from_vectors(s2 - s1, d2 - d1)
        c = np.linalg.norm(d2 - d1) / np.linalg.norm(s2 - s1)
        return c * ((frame[self.lips] - s1) @ r.T) + d1
                
    def handle_hand(self, frame: np.ndarray, which: str='right'): 
        # shift and scale hand (hand wrist-thumb idx 0, 4 vs pose wrist-thumb idx 16/21, 16/22)
        thumb_id = 3 # choose 2/3/4 to affect the scale c (longer/shorter hand thumb to align)
        # NOTE: (rotate to align palm surface also?)
        if which == 'right': 
            pose_wrist_id = 16
            pose_thumb_id = 22
            hand = self.rhand 
        elif which == 'left':
            pose_wrist_id = 15
            pose_thumb_id = 21
            hand = self.lhand 

        s1, s2 = frame[hand[0]], frame[hand[thumb_id]] 
        d1, d2 = frame[self.pose_all[pose_wrist_id]], frame[self.pose_all[pose_thumb_id]]
        c = np.linalg.norm(d2 - d1) / np.linalg.norm(s2 - s1)
        return c * (frame[hand] - s1) + d1

    @classmethod 
    def mask_frame(cls, multiframes: np.ndarray, window_shape: int, 
                   joint_idx: list, return_nonempty=True):
        """
        return idx of frames where its next n frames (n=window_shape) all have all NaN values on joint_idx when ``return_noneympty=False``.
        Otherwise, return the idx the other way around. 
        """
        mask = ~np.isnan(multiframes[:, joint_idx, 0]) # take the 'x'-axis as the deputy
        mask = (mask.sum(axis=1) == 0) # True: both hands empty 
        count = sliding_window_view(mask, window_shape=window_shape).sum(axis=1)

        cond = (count == window_shape) # True: all empty within range
        if return_nonempty: 
            cond = ~cond
        idx = np.argwhere(cond).squeeze(axis=1)
        return idx 

    @classmethod 
    def ffill_frame(cls, multiframes: np.ndarray): 
        """
        ffill frames and return the filled array. 
        For bfill, user can get it via ``ffill_frame(arr[::-1])[::-1]``.
        """
        mask = np.isnan(multiframes[..., 0]) # take x-axis as the deputy
        mask = (mask.sum(axis=1) == 0)       # True: not empty frames
        idx = np.where(mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, out=idx)
        output = multiframes[idx, ...]
        return output 

    @classmethod
    def bfill_frame(cls, multiframes: np.ndarray): 
        return cls.ffill_frame(multiframes[::-1, ...])[::-1, ...]

    @classmethod 
    def repeat_trim_frame(cls, multiframes: np.ndarray, max_frame: int): 
        repeats = np.ceil(max_frame / multiframes.shape[0]).astype(int)
        output = np.tile(multiframes, (repeats,1,1))[:max_frame, ...]
        return output 

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # output shape (C, T, V, M=1)
        # get data
        sample_path = self.label_info["path"].iloc[index]
        data_numpy = load_relevant_data_subset(self.data_path / sample_path) # (T, V, C) 

        # delete frames with too long no double hands captured within the lookahead window
        # will delete the frames later to avoid affecting 
        if (self.ignore_empty_hands_frames 
            and self.ignore_window > 0 
            and data_numpy.shape[0] > self.ignore_window
        ):  
            idx = self.mask_frame(data_numpy, self.ignore_window, self.lhand+self.rhand) 
            data_numpy = data_numpy[idx, ...]

        # forward fill lips along frame axis if NaN and then fill 0 if still NaN. 
        data_numpy[:, self.lips, :] = self.ffill_frame(data_numpy[:, self.lips, :])
        # data_numpy[:, self.lips, :] = np.nan_to_num(data_numpy[:, self.lips, :]) # will still be imputed later

        # align lips, hands into pose
        # pose has no missing values at all across all data
        for i_frame, frame in enumerate(data_numpy):
            if (~np.isnan(frame[self.lips])).all(): 
                data_numpy[i_frame, self.lips, :] = self.handle_lips(frame)
            if (~np.isnan(frame[self.rhand])).all(): 
                data_numpy[i_frame, self.rhand, :] = self.handle_hand(frame, which='right')
            if (~np.isnan(frame[self.lhand])).all(): 
                data_numpy[i_frame, self.lhand, :] = self.handle_hand(frame, which='left')
        
        # keep only relevant joints and fillna 
        data_numpy = data_numpy[:, self.lips + self.lhand + self.pose + self.rhand, :]
        data_numpy = np.nan_to_num(data_numpy)

        # repeat frames till max_frame
        if self.pad_repeat: 
            data_numpy = self.repeat_trim_frame(data_numpy, max_frame=self.T)

        # centralization
        mean, std = np.nanmean(data_numpy, axis=(0, 1)), np.nanstd(data_numpy, axis=(0, 1))
        data_numpy = (data_numpy - mean) / std
        data_numpy = np.nan_to_num(data_numpy) # (T, V, C)

        # required output shape (assume self.M=1)
        data_numpy = data_numpy.transpose(2, 0, 1)[..., np.newaxis]
        
        # get corresp. label and label_name (person_id + seq_id)
        label = self.label_info["sign_idx"].iloc[index]
        tmp = sample_path.split("/")[-2:]
        label_str = tmp[0] + "_" + tmp[1].rstrip(".parquet")
        
        return data_numpy, label, label_str
    
def gendata(data_path: Path, label_path: Path, 
            data_out_dir: Path, label_out_dir: Path,
            max_frame=126, 
            num_joint=50, 
            num_person_in=1,  # observe the first 5 persons
            num_person_out=1,  # then choose 2 persons with the highest score
            val_ratio=0.05, 
            verbose=True
    ):
    
    feeder = Feeder_asl(
        data_path=data_path,
        label_path=label_path,
        max_frame=max_frame, 
        num_joint=num_joint, 
        num_person_in=num_person_in,
        num_person_out=num_person_out,
    ) 
    
    fp = np.zeros((len(feeder), 3, max_frame, num_joint, num_person_out), dtype=np.float32)
    labels = [] 
    labels_str = [] 
    for i in tqdm(range(len(feeder))): 
        data, l, lstr = feeder[i]
        fp[i, ...] = data
        labels.append(l)
        labels_str.append(lstr)
    labels = np.array(labels)
    labels_str = pd.Series(labels_str) # save memory compared to np.array

    # get split idx TODO: GroupShuffleSplit? 
    indices = np.random.permutation(fp.shape[0])
    split_idx = int(fp.shape[0] * val_ratio)
    tra_idx, val_idx = indices[:-split_idx].tolist(), indices[-split_idx:].tolist()
    indices = [tra_idx, val_idx]

    # save 
    tra_fname = "train_data_joint.npy"
    val_fname = "val_data_joint.npy"
    tra_flabel = "train_label.pkl"
    val_flabel = "val_label.pkl"
    for i, case in enumerate([tra_fname, val_fname]): 
        print("Writting {} ...".format(case))
        np.save(data_out_dir / case, fp[indices[i]])
    for i, case in enumerate([tra_flabel, val_flabel]): 
        print("Writting {} ...".format(case))
        with (label_out_dir / case).open('wb') as f: 
            pickle.dump(
                (labels_str[indices[i]].tolist(), 
                 labels[indices[i]].tolist())
                , f
            )
    if verbose: 
        print(
            "train/val shape and memory usage:" + 
            "\n\t train shape: {}, memory: {} MB".format(
                fp[indices[0]].shape, fp[indices[0]].nbytes / 1024**2
            ) + 
            "\n\t valid shape: {}, memory: {} MB".format(
                fp[indices[1]].shape, fp[indices[1]].nbytes / 1024**2
            ) + 
            "\n\t each sample shape: {}, memory: {} KB".format(
                fp[0].shape, fp[0].nbytes / 1024
            )
        )



if __name__ == '__main__':
    # some config for gendata()
    max_frame = 82

    # control seed 
    seed = 42 
    np.random.seed(seed)

    # parser
    project_dir = Path(re.search(".*MS-G3D", os.getcwd())[0])
    parser = argparse.ArgumentParser(
        description='asl-sign Data Converter.')
    parser.add_argument(
        '--data_path', default=project_dir / 'data/asl-signs_raw', type=Path)
    parser.add_argument(
        '--out_folder', default=project_dir / 'data/asl-signs', type=Path)
    arg = parser.parse_args()

    # gen data
    print('asl-signs ', "train + val")
    if not os.path.exists(arg.out_folder):
        os.makedirs(arg.out_folder)
    label_path = arg.data_path / 'train.csv'
    gendata(
        arg.data_path, label_path, arg.out_folder, arg.out_folder, 
        max_frame=max_frame
    )
