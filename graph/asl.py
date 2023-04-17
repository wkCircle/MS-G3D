import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np
from graph import tools

def get_idxmap(): 
    lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    lips = lipsUpperInner[::2] + lipsLowerInner[::-2][1:-1] # len 10 
    assert (78 in lips) & (308 in lips), "78 & 308 are requried (left/right-most anchor points)."

    lhand_anchor, rhand_anchor = 468, 522
    lhand = list(range(0+lhand_anchor, 21+lhand_anchor)) # len 21 
    rhand = list(range(0+rhand_anchor, 21+rhand_anchor)) # len 21

    pose_anchor = 489
    ear_eye_nose = [8, 6, 4, 0, 1, 3, 7]
    upper_body = [22, 16, 14, 12, 24, 23, 11, 13, 15, 21]
    pose = list(np.array(ear_eye_nose + upper_body[2:-2], dtype=int) + pose_anchor) # len 13

    old_idx = lips + lhand + pose + rhand 
    new_idx = range(len(old_idx))

    old2new = dict(zip(old_idx, new_idx))
    new2old = dict(zip(new_idx, old_idx))
    return old2new, new2old

def get_original_connection_idx(verbose=False): 
    # lips 
    lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    lips = lipsUpperInner[::2] + lipsLowerInner[::-2][1:-1] # len 10 
    assert (78 in lips) & (308 in lips), "78 & 308 are requried (left/right-most anchor points)."
    lips_conn = list(map(lambda x, y: (x,y), lips, lips[1:] + [lips[0]])) # len 10

    # hands 
    lhand_anchor, rhand_anchor = 468, 522
    hand_conn = [
        (0, 1), (1, 2), (2, 3), (3, 4), 
        (0, 5), (5, 6), (6, 7), (7, 8), 
        (5, 9), (9, 10), (10, 11), (11, 12), 
        (9, 13), (13, 14), (14, 15), (15, 16), 
        (13, 17), (17, 18), (18, 19), (19, 20), 
        (0, 17)
    ] # len 21
    lhand_conn = [(x+lhand_anchor,y+lhand_anchor) for x,y in hand_conn]
    rhand_conn = [(x+rhand_anchor,y+rhand_anchor) for x,y in hand_conn]

    # pose 
    pose_anchor = 489
    pose_conn = [
        (8, 6), (6, 4), (4, 0), (0, 1), (1, 3), (3, 7), 
        (14, 12), (12, 11), (11, 13), # horizontal shoulder to shoulder shape 
        (12, 24), (24, 23), (23, 11), # U-shape to form the body 
    ] # len 12 
    pose_conn = [(x+pose_anchor, y+pose_anchor) for x, y in pose_conn]

    # cross-body connection 
    cross_conn = [
        (78, 0+pose_anchor), (308, 0+pose_anchor), # dummy node to lips 
        (78, 12+pose_anchor), (308, 11+pose_anchor), # dummy lips to pose shoulders
        (0+lhand_anchor, 13+pose_anchor), # left-hand wrist to left elbow
        (0+rhand_anchor, 14+pose_anchor), # right-hand wrist to right elbow
    ]
    all_conn =lips_conn + lhand_conn + pose_conn + rhand_conn + cross_conn
    if verbose: 
        print(
            "length of all_conn is {} = \n".format(len(all_conn)) + 
            "\t  lips_conn({})".format(len(lips_conn)) + 
            "\n\t+ lhand_conn({})".format(len(lhand_conn)) + 
            "\n\t+ pose_conn({})".format(len(pose_conn)) + 
            "\n\t+ rhand_conn({})".format(len(rhand_conn)) + 
            "\n\t+ cross_conn({})".format(len(cross_conn))
        )
    return all_conn

def get_newidx_connections(): 
    old2new, _ = get_idxmap()
    old_all_conn = get_original_connection_idx()
    new_all_conn = [ (old2new[x], old2new[y]) for x, y in old_all_conn]
    return new_all_conn


class AdjMatrixGraph:
    num_node = 65
    self_link = [(i, i) for i in range(num_node)]
    inward = get_newidx_connections()
    outward = [(j, i) for (i, j) in inward]
    neighbor = inward + outward

    def __init__(self):
        self.num_nodes = self.num_node
        self.edges = self.neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__': # only for debug
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
