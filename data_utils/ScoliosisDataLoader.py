import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import json
from tqdm import tqdm
import cv2
import open3d as o3d
import numpy as np
import math
import random
from pygsp import graphs
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from plyfile import PlyData, PlyElement
import time


class ScoliosisDataset(Dataset):
    """Back Landmarks dataset."""

    def __init__(self, root, transform=None, augmentation=False, npoints=10, normal_channel=False, class_weights=False, split='train', task='reg', mode='train'):
        """
        Arguments:
            root_dir (string): /intensity: the intensity images (.jpg)
                               /xyz: the 3D coordinates (.raw)
                               /landmarks: 2D coordinates (x, y) of each landmark, labeled.
                               All written in a .json file

        """
        self.root = root
        self.transform = transform
        self.npoints = npoints
        self.augmentation = augmentation
        self.normal_channel = normal_channel
        self.class_weights = class_weights
        self.split = split
        self.task = task
        if mode == 'inference':
            self.data = os.listdir(root)

        else:
            with open(self.root + f'/{self.split}.txt', 'r') as data_file:
                self.data = data_file.readlines()
    

        # the image dimensions
        self.w = 1936
        self.h = 1176
        self.header_size = 512
        np.random.seed(0)

    def __len__(self):
        return len(self.data)

    # might need to change this. Update it to be more efficient. For now, I leave it as it is.
    def read_single_xyz_raw_file(self, file_path):
        with open(file_path, 'r') as f:
            f.seek(self.header_size)
            data_array = np.fromfile(f, np.float32).reshape((self.h, self.w, 3))
        return data_array

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # I might want to use cache. Comment it for now.
        # if index in self.cache:
        #     point_set, cls, seg = self.cache[index]
        frame_name = self.data[idx].strip('\n')
        path = self.root + '/' + create_path(frame_name)
        # print(path)
        with open(path, 'rb') as frame_file:
            frame = np.load(frame_file, allow_pickle=True).item()
        frame['radius'] = 0.15
        if 'pt19' in frame_name:
            frame['radius'] = 0.08
        if 'pt20' in frame_name:
            frame['radius'] = 0.08

        points = np.array(frame['xyz'])
        markers = np.array(convert_landmarks_to_tensor(frame['markers']))
        pixel_vals = frame['intensity']
        markers_indices = find_indices(markers, points)
        pixel_vals = np.repeat(pixel_vals, 3).reshape(-1, 3)

        points_and_markers, m = pc_normalize(np.concatenate((points, markers), axis=0))
        
        if self.augmentation:
            points_and_markers = rotate_3d(points_and_markers)
        
        points_normalized = points_and_markers[:-8]
        markers_normalized = points_and_markers[-8:] 
        # print(len(points_and_markers), len(points_normalized), len(markers_normalized))

        downsampled_indices = np.concatenate((np.random.choice(len(points_normalized), self.npoints-len(markers_indices), replace=True), markers_indices))

        # with open("test_repro2.txt", "ab") as f:
        #     np.savetxt(f, downsampled_indices)
        #     f.write(b'\n')
        pointset_downsampled_orig = points[downsampled_indices, :]
        pointset_downsampled = points_normalized[downsampled_indices, :]
        pixel_vals = pixel_vals[downsampled_indices, :]

        # markers_indices_sampled = find_indices(markers_normalized, pointset_downsampled)

        segments_labels = []
        segments_points = []
        if self.task == 'seg':
            segments_labels, segments_points = find_segments(markers_normalized, pointset_downsampled, frame['radius'])      
        
        weights = np.ones([len(segments_points)], dtype=np.float64)
        if self.class_weights:
            num_per_class = np.array([len(segment_points) for segment_points in segments_points], dtype=np.int32)
            weights = get_class_weights(num_per_class, normalize=True)
        points = np.concatenate((pointset_downsampled, pixel_vals), axis=1)
        # print(markers_indices_sampled)
        out = {'points': points, 'class': 0, 'segments': np.array(segments_labels), 'weights': weights, 'markers': markers_normalized, 'frame': frame_name, 'markers_original':markers, 'points_original': pointset_downsampled_orig}
        

        return out
    

class ScoliosisTrackingDataset(Dataset):
    """Scene flow dataset"""

    def __init__(self, root, transform=None, augmentation=False, npoints=10000, normal_channel=False, class_weights=False, split='train', mode='inference'):
        """
        Arguments:
            root_dir (string): /intensity: the intensity images (.jpg)
                               /xyz: the 3D coordinates (.raw)
                               /landmarks: 2D coordinates (x, y) of each landmark, labeled.
                               All written in a .json file

        """
        self.root = root
        self.transform = transform
        self.npoints = npoints
        self.augmentation = augmentation
        self.normal_channel = normal_channel
        self.class_weights = class_weights
        self.split = split
        if mode == 'inference':
            self.data = os.listdir(root)

        else:
            with open(self.root + f'/{self.split}.txt', 'r') as data_file:
                self.data = data_file.readlines()
    

        # the image dimensions
        self.w = 1936
        self.h = 1176
        self.header_size = 512
        np.random.seed(0)

    def __len__(self):
        return len(self.data)

    # might need to change this. Update it to be more efficient. For now, I leave it as it is.
    def read_single_xyz_raw_file(self, file_path):
        with open(file_path, 'r') as f:
            f.seek(self.header_size)
            data_array = np.fromfile(f, np.float32).reshape((self.h, self.w, 3))
        return data_array

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # I might want to use cache. Comment it for now.
        # if index in self.cache:
        #     point_set, cls, seg = self.cache[index]
        frame_name = self.data[idx].strip('\n')
        path = self.root + '/' + create_path(frame_name)
        # print(path)
        with open(path, 'rb') as frame_file:
            frame = np.load(frame_file, allow_pickle=True).item()

        points = np.array(frame['xyz'])
        markers = np.array(convert_landmarks_to_tensor(frame['markers']))
        markers_indices = find_indices(markers, points)
        pixel_vals = frame['intensity']
        pixel_vals = np.repeat(pixel_vals, 3).reshape(-1, 3)

        points_and_markers, m = pc_normalize(np.concatenate((points, markers), axis=0))
        
        if self.augmentation:
            points_and_markers = rotate_3d(points_and_markers)
        
        points_normalized = points_and_markers[:-8]
        markers_normalized = points_and_markers[-8:] 

        downsampled_indices = np.concatenate(np.random.choice(len(points_normalized), self.npoints, replace=True), markers_indices)

        pointset_downsampled_orig = points[downsampled_indices, :]
        pointset_downsampled = points_normalized[downsampled_indices, :]
        pixel_vals = pixel_vals[downsampled_indices, :]
        markers_indices_sampled = find_indices(markers_normalized, pointset_downsampled)
    

        points = np.concatenate((pointset_downsampled, pixel_vals), axis=1)
        out = {'points': points, 'markers_indices': markers_indices_sampled, 'frame': frame_name, 'markers_original':markers, 'points_original': pointset_downsampled_orig}
        

        return out
    


def create_path(frame_name):
    if '.npy' in frame_name:
        # print('found npy in frame name')
        return frame_name
    info = frame_name.split('_')
    if int(info[0][2:]) < 10 and len(info[0][2:]) == 1:
        if info[1] == 'auto':
            sequence_path = ('Participant0' + info[0][2:] + f'/{info[1]}/Prise{info[2]}/')
        else:
            sequence_path = ('Participant0' + info[0][2:] + f'/{info[1]}/{info[2]}/Prise{info[3]}/').replace('libr/', 'Libre/').replace('libre', 'Libre')
    else:
        if info[1] == 'auto':
            sequence_path = ('Participant' + info[0][2:] + f'/{info[1]}/Prise{info[2]}/')
        else:
            sequence_path = ('Participant' + info[0][2:] + f'/{info[1]}/{info[2]}/Prise{info[3]}/').replace('libr/', 'Libre/').replace('libre', 'Libre')

    frame_path = sequence_path + 'data/' + frame_name + '.npy'
    
    return frame_path


def find_indices(markers, points):
    indices = []
    
    for i, point in enumerate(points):
        if any(np.array_equal(point, marker) for marker in markers):
            indices.append(i)
    
    return indices

def get_dimensions(my_dict):

    if isinstance(my_dict, dict):
        return [len(my_dict)] + get_dimensions(next(iter(my_dict.values())))
    elif isinstance(my_dict, torch.Tensor):
        return list(my_dict.size())
    elif isinstance(my_dict, list):
        return [len(my_dict)] + get_dimensions(my_dict[0])
    else:
        return []

def convert_landmarks_to_tensor(landmarks):

    # l, c = get_dimensions(landmarks)
    landmarks_tensor = torch.zeros(8, 3)
    # torch.tensor(landmarks['C'])
    # print(landmarks)
    try:
        landmarks_tensor[0] = torch.tensor(landmarks['C'])
    except:
        landmarks_tensor[0] = torch.tensor(landmarks['C7'])
    try:
        landmarks_tensor[1] = torch.tensor(landmarks['G'])
    except:
        try:
            landmarks_tensor[1] = torch.tensor(landmarks['ScG'])
        except:
            landmarks_tensor[1] = torch.tensor(landmarks['SG'])
    try:
        landmarks_tensor[2] = torch.tensor(landmarks['D'])
    except:
        try:
            landmarks_tensor[2] = torch.tensor(landmarks['ScD'])
        except:
            landmarks_tensor[2] = torch.tensor(landmarks['SD'])
    landmarks_tensor[3] = torch.tensor(landmarks['IG'])
    landmarks_tensor[4] = torch.tensor(landmarks['ID'])
    # summer 2023 participants
    if 'Tsup' in landmarks.keys() and 'Tap' in landmarks.keys():
        landmarks_tensor[5] = torch.tensor(landmarks['Tsup'])
        landmarks_tensor[6] = torch.tensor(landmarks['Tap'])
        try:
            landmarks_tensor[7] = torch.tensor(landmarks['Tinf'])
        except:
            try:
                landmarks_tensor[7] = torch.tensor(landmarks['L1'])
            except:
                try:
                    landmarks_tensor[7] = torch.tensor(landmarks['L'])
                except:
                    landmarks_tensor[7] = landmarks_tensor[6]
    
    # summer 2022 participants
    elif 'T1' in landmarks.keys() and 'T2' in landmarks.keys():
        landmarks_tensor[5] = torch.tensor(landmarks['T1'])
        landmarks_tensor[6] = torch.tensor(landmarks['T2'])
        if 'L' in landmarks.keys():
            landmarks_tensor[7] = torch.tensor(landmarks['L'])
        elif 'L1' in landmarks.keys() and 'L2' in landmarks.keys():
            landmarks_tensor[7] = torch.tensor(landmarks['L1'])
    elif 'T' in landmarks.keys():
        landmarks_tensor[5] = torch.tensor(landmarks['T'])
        landmarks_tensor[6] = torch.tensor(landmarks['L1'])
        landmarks_tensor[7] = torch.tensor(landmarks['L2'])

    
    return landmarks_tensor

# chatGPT generated this for rotation
def rotate_3d(point_cloud, angle_x=30, angle_y=30, angle_z=30, random=False):
    # Generate random angles for rotation around x, y, and z axes
    if random:
        angle_x = np.random.uniform(0, 30)
        angle_y = np.random.uniform(0, 30)
        angle_z = np.random.uniform(0, 30)
    # print(angle_x, angle_y, angle_z)

    # Convert angles to radians
    angle_x_rad = np.deg2rad(angle_x)
    angle_y_rad = np.deg2rad(angle_y)
    angle_z_rad = np.deg2rad(angle_z)

    # Compute rotation matrices for x, y, and z axes
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
                                  [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]])

    rotation_matrix_y = np.array([[np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
                                  [0, 1, 0],
                                  [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]])

    rotation_matrix_z = np.array([[np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
                                  [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
                                  [0, 0, 1]])

    # Apply rotations successively to the point cloud
    rotated_point_cloud = np.dot(np.dot(np.dot(point_cloud, rotation_matrix_x), rotation_matrix_y), rotation_matrix_z)

    return rotated_point_cloud

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, m

def euclidean_dist(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def find_segments(landmarks, body, radius=0.15):
    # takes the body points and the landmarks positions and divides the
    # body points into neighborhoods around each point based on geodesic distance
    segment_labels = []
    segments_points = [[] for i in range(len(landmarks) + 1)]
    # print(segments_points[4])
    for point in body:
        segment_label = 8
        min_distance = math.inf
        for i, landmark in enumerate(landmarks):
            # print(landmark)
            distance = euclidean_dist(landmark, point)
            if distance < min_distance and distance < radius:
                min_distance = distance
                segment_label = i
        # print(segment_label)
        segments_points[segment_label].append(point)    
        segment_labels.append(segment_label)
    

    return segment_labels, segments_points

# copied from pointnext data_utils
def get_class_weights(num_per_class, normalize=False):
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)

    if normalize:
        ce_label_weight = (ce_label_weight *
                           len(ce_label_weight)) / ce_label_weight.sum()
    return torch.from_numpy(ce_label_weight.astype(np.float32))


if __name__ == '__main__':

    manual_seed = 0
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(manual_seed)
    # # testing the dataset
    dataset = ScoliosisDataset(root='/home/travail/ghebr/Data', split='data')


    # # train_dataset, val_dataset, test_dataset = random_split(dataset, (0.8, 0.2))
    DataLoader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=10, worker_init_fn=seed_worker, generator=g)



    for batch_id, data in tqdm(enumerate(DataLoader), total=len(DataLoader), smoothing=0.9):
        continue
        # with open("test_repro2.txt", "ab") as f:
            # batch_id = str()
            # f.write(b'\n')
            # np.savetxt(f, np.array([batch_id]))
    # #     # print(points.size(), segments_labels.size(), pixel_vals.size(), weights.size())
    #     points, label, target, weight, pixel_vals, markers, frame_name = points[0], label[0], target[0], weight[0], pixel_vals[0], markers[0], frame_name[0]
    #     # points = points.transpose(2, 1)
    #     # pixel_vals = pixel_vals.transpose(2, 1)
    #     num_part = 9
    #     pred_seg_centers = torch.zeros((num_part, 3))
    #     pred_seg_num = torch.zeros(num_part)
    #     for j, point in enumerate(points):
    #         pred_seg_centers[target[j]] += point
    #         pred_seg_num[target[j]] += 1
            
    #     pred_seg_centers = pred_seg_centers / pred_seg_num.unsqueeze(-1)
    #     num_labels = 9
    #     segments_points = [[] for i in range(num_labels+1)]
    #     for i, point in enumerate(points):
    #         label = target[i]
    #         segments_points[label].append(point)
    #     segments_points[num_labels] = pred_seg_centers

        # print([np.array(segments_points[i]) for i in range(10)])
        # visualize_point_cloud(segments_points, frame_name=frame_name)
        # break
    #     # continue
    #     # segments_points = np.array([[segments_points[0][j][i] for i in range(len(segments_points[0][j]))] for j in range(len(segments_points[0]))])
    #     # with open('segmentaion_test' + str(batch_id) + '.json', 'w') as f:
    #     #     json.dump(segments_points.size(), f)
    #     #     points, landmarks = points.float().cuda(), landmarks.cuda()
    #     print(pred_seg_centers)
    # print(create_path('pt02_BD_libre_01_001133_82'))






