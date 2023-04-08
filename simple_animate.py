import numpy as np 
import pandas as pd 
import pickle 
from pathlib import Path
import json 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 
from IPython.core.display import Video, HTML # Requires ffmpeg package
from torch.utils.data import Dataset


val_joint_path = Path.home() / r"GitHub\MS-G3D\data\kinetics\val_data_joint.npy"
val_label_path = Path.home() / r"GitHub\MS-G3D\data\kinetics\val_label.pkl"
val_joint = np.load(val_joint_path)
with open(val_label_path, 'rb') as fr: 
    val_label = pickle.load(fr)
print("val array memory size: {} GB".format(val_joint.itemsize * val_joint.size / (1024 ** 3)))

# example video .json file 
sample_path = Path.home() / r"GitHub\MS-G3D\data\kinetics_raw\kinetics_train\___dTOdxzXY.json"
with open(sample_path, 'r') as fr: 
    sample = json.load(fr)
label = [sample["label"], sample["label_index"]]
tmp1 = pd.json_normalize(sample["data"]).explode("skeleton") # explode: some frames may contain more than 1 person. 
tmp2 = tmp1["skeleton"].apply(lambda li: pd.Series(li)) 
tmp2["pose_x"] = tmp2.pose.apply(lambda li: li[0::2])
tmp2["pose_y"] = tmp2.pose.apply(lambda li: li[1::2])
tmp2["score_x"] = tmp2.score.apply(lambda li: li[0::2])
tmp2["score_y"] = tmp2.score.apply(lambda li: li[1::2])
sample_video = pd.concat([tmp1.drop(columns="skeleton"), tmp2.drop(columns=["pose", "score"])], axis=1).reset_index(drop=True)

class SimpleAnimation: 
    @classmethod
    def _animation_frame(cls, frame: int, video: pd.DataFrame, ax: plt.Axes, xyborder: tuple):
        """
        helper function to draw the signer given a frame idx.
        Args: 
            xyborder (4-tuple): (xmin, xmax, ymin, ymax) of float. 
        """
        f = video.query("frame_index==@frame")
        
        # loop over num_person appeared in the frame 
        ax.clear()
        for _, person in f.iterrows(): 
            ax.scatter(person.pose_x, person.pose_y)
        ax.set_xlim(*xyborder[:2])
        ax.set_ylim(*xyborder[1:])
        return ax 

    @classmethod
    def gen_animation(cls, video: pd.DataFrame): 
        """main function to call for video visualization."""
        fig, ax = plt.subplots(1, 1)
        x = video.pose_x.explode()
        y = video.pose_y.explode()
        xyborder = (x.min() - 0.2, x.max() + 0.2, 
                    y.min() - 0.2, y.max() + 0.2)
        animation = FuncAnimation(
            fig, func= cls._animation_frame, 
            frames=video.frame_index.sort_values().unique(), 
            fargs=(video, ax, xyborder)
        )
        
        return HTML(animation.to_html5_video())

html = SimpleAnimation.gen_animation(sample_video)
display(html)
print("Its corresponding lable is: {}".format(label))

# Comment: 
# train_data_jolint.npy (240436, 3, 300, 18, 2) (of type float32, 29 GB): the training data of 5 dim meaning (samples, channels, max_frames, num_joint, num_person_out)
# where the 3 channels are x-coord, y-coord, and the confidence score of the point deteced by the OpenPose. 
# train_label: Tuple(List[str], List[int]) has two elements of the same len 240436. The 1st element is a list of str storing the suffix of original .json file name, and the 2nd element stores a list of ground truth indices indicating the ground truth class label.