import open3d as o3d
import numpy as np
import json
import os
import time
import torch
import math
from tqdm import tqdm
import cv2

w = 1936
h = 1176
header_size = 512

def visualize_pc_with_intensities(points, intensities, normalize=False):
    
    if normalize:
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
    intensities = np.repeat(intensities, 3).reshape(-1, 3)


    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    # Assign color to each point in the point cloud
    cloud.colors = o3d.utility.Vector3dVector(np.array(intensities))
    o3d.visualization.draw_geometries([cloud])  

def euclidean_dist(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def find_segments(landmarks, body, radius=0.05):
    # takes the body points and the landmarks positions and divides the
    # body points into neighborhoods around each point based on geodesic distance
    segment_labels = []
    segments_points = [[] for i in range(len(landmarks) + 1)]
    # print(len(landmarks))
    # print(segments_points[4])
    for point in body:
        segment_label = 8
        if len(landmarks) < 8:
            segment_label = len(landmarks)
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
    
    # print(len(segments_points))

    return segment_labels, segments_points

    

    return segment_labels, segments_points

def visualize_point_cloud(point_cloud_list, save_path=None, margin=False, colors=None, landmarks=False, offset=50):

    # colors = np.random.uniform(0, 1, size=(len(point_cloud_list), 3))  # Generate random colors for each point cloud
    # colors = np.array([[0, 0.98, 0.957], [0.835, 0, 0.98], [0.749, 0.98, 0],
    #           [1, 1, 0.322], [0.98, 0, 0.58], [0.333, 0, 0.98],
    #           [0.98, 0.533, 0], [0.631, 1, 0.8], [0.871, 0.8, 0.831], []])
    # colors = [np.random.rand(3, ) for i in range(len(point_cloud_list))]
    # print(colors)
    # color = (color + 1.0) / 2.0  # Adjust the range to [0.5, 1]
    
    if colors == None:
        colors = [
            (0.984, 0.705, 0.682),  # Pastel Red
            (0.992, 0.805, 0.643),  # Pastel Orange
            (0.992, 0.941, 0.643),  # Pastel Yellow
            (0.992, 0.858, 0.745),  # Pastel Peach
            (0.698, 0.850, 0.992),  # Pastel Blue
            (0.898, 0.698, 0.992),  # Pastel Purple
            (0.752, 0.937, 0.752),  # Pastel Green
            (0.745, 0.745, 0.992),  # Pastel Lavender
            (0.901, 0.901, 0.980),  # Pastel Lavender Blue
            (0, 0, 1),   # Blue
            (1, 0, 0)   # Red
        ]

    # Create an empty point cloud
    merged_point_cloud = o3d.geometry.PointCloud()

    print(len(point_cloud_list))
    
    for i, points in enumerate(point_cloud_list):
        
        # print(i)
        if len(points) == 0:
            continue
        # Convert the list of 3D coordinates to a numpy array
        points = np.array(points)
        if margin:
            points = [point + [0, i*0.5, 0] for point in points]
        if landmarks:
            if i > 1:
                points = [point + [0, 0, offset] for point in points]
        # Create an Open3D point cloud from the numpy array
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        
        # Assign color to each point in the point cloud
        cloud.colors = o3d.utility.Vector3dVector(np.tile(colors[i], (len(points), 1)))
        
        # Merge the current point cloud with the previous ones
        merged_point_cloud += cloud
    
    # rotating the point cloud
    merged_point_cloud.rotate(merged_point_cloud.get_rotation_matrix_from_xyz((0, np.pi, np.pi / 2)))
    # Saving the point cloud
    # save_2D_projection(merged_point_cloud, output_file=save_path)
    # Visualize the merged point cloud
    if save_path is not None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(merged_point_cloud)
        vis.update_geometry(merged_point_cloud)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([merged_point_cloud])



def visualize_segments(path=None, points_file=None, labels_file=None, landmarks_file=None, show_landmarks=True, num_labels=9, save_path=None):
    if path is not None:
        with open(path, 'r') as file:
            data = json.load(file)
        points = data['points']
        points = np.array(points)[:, :3]
        labels = data['labels']
        landmarks = data['markers']
        centers = data['centers']
    else:
        # Read the PLY file
        with open(points_file, 'r') as file:
            points = json.load(file)

        # Load the segmentation labels from JSON
        with open(labels_file, 'r') as file:
            labels = json.load(file)

        if landmarks_file is not None:
            with open(landmarks_file, 'r') as file:
                landmarks = json.load(file)
        else:
            landmarks = []
            centers = []

    # points = np.array(point_cloud.points)
    num_labels = num_labels
    segments_points = [[] for i in range(num_labels+2)]
    for i, point in enumerate(points):
        label = labels[i]
        segments_points[label].append(point)

    segments_points[num_labels] = np.array(landmarks) + [0, 0, - 50]
    if show_landmarks:
        segments_points[num_labels+1] = np.array(centers) + [0, 0, - 50]
    # print(len(segments_points[9]))
    visualize_point_cloud(segments_points, save_path=save_path)
        

def visualize_errors(path=None, points_path=None, targets_path=None, pred_path=None):

    if path is not None:
        with open(path, 'r') as file:
            data = json.load(file)
        points = data['points']
        points = np.array(points)[:, :3]
        preds = data['labels']
        landmarks = data['markers']
        targets = data['targets']
    
    else:
        with open(points_path, 'r') as file:
            points = json.load(file)
        with open(pred_path, 'r') as file:
            preds = json.load(file)
        # print(preds)
        with open(targets_path, 'r') as file:
            targets = json.load(file)
        # print(targets)
    correct = []
    wrong = []

    for i, point in enumerate(points):
        if preds[i] >= 5:
            preds[i] -= 2
        if targets[i] == preds[i]:
            correct.append(point)
        else:
            wrong.append(point)
    
    point_cloud_correct = o3d.geometry.PointCloud()
    point_cloud_correct.points = o3d.utility.Vector3dVector(correct)
    point_cloud_correct.paint_uniform_color([0, 1, 0])
    # point_cloud_correct.rotate(point_cloud_correct.get_rotation_matrix_from_xyz((0, np.pi, np.pi / 2)))
    
    point_cloud_wrong = o3d.geometry.PointCloud()
    point_cloud_wrong.points = o3d.utility.Vector3dVector(wrong)
    point_cloud_wrong.paint_uniform_color([1, 0, 0])
    # point_cloud_wrong.rotate(point_cloud_wrong.get_rotation_matrix_from_xyz((0, np.pi, np.pi / 2)))

    total = point_cloud_wrong + point_cloud_correct
    total.rotate(total.get_rotation_matrix_from_xyz((0, np.pi, np.pi / 2)))

    o3d.visualization.draw_geometries([total])
    
def visualize_sequence(seq_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    frames_list= os.listdir(seq_path)

    sequence = []
    for i in range(len(frames_list)):
        pc = o3d.io.read_point_cloud(seq_path + frames_list[i])
        sequence.append(pc)

    # geometry is the point cloud used in your animaiton
    # geometry = o3d.geometry.PointCloud()
    # vis.add_geometry(geometry)

    vis.add_geometry(sequence[0])
    vis.poll_events()
    vis.update_renderer()

    for i in range(len(sequence)):
        time.sleep(10)
        # now modify the points of your geometry
        # you can use whatever method suits you best, this is just an example
        geometry.points = np.array(sequence[i].points)
        # print(np.array(sequence[i].points))
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()

def convert_landmarks_to_tensor(landmarks):

    # l, c = get_dimensions(landmarks)
    n = min(8, len(landmarks.keys()))
    landmarks_tensor = torch.zeros(n, 3)
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
    try:
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
    except:
        if 'Tsup' in landmarks.keys() and 'Tap' in landmarks.keys():
            landmarks_tensor[3] = torch.tensor(landmarks['Tsup'])
            landmarks_tensor[4] = torch.tensor(landmarks['Tap'])
            try:
                landmarks_tensor[5] = torch.tensor(landmarks['Tinf'])
            except:
                try:
                    landmarks_tensor[5] = torch.tensor(landmarks['L1'])
                except:
                    try:
                        landmarks_tensor[5] = torch.tensor(landmarks['L'])
                    except:
                        landmarks_tensor[5] = landmarks_tensor[6]
    
    return landmarks_tensor

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

# root = '/home/travail/ghebr/project/PointNet2/Pointnet_Pointnet2_pytorch/log/back_segmentation/pointnet2_part_seg_msg_weighed_intensities_all_data_reg_tr02/2500/predicted/'
# visualize_segments(root + 'frame_xyz_231.json', root + 'seg_pred_231.json', root + 'coordinates_231.json')
# an example of poor model performance
# visualize_errors(root + 'frame_xyz_100.json', root + 'seg_pred_100.json', root + 'seg_target_100.json')
# visualize_sequence('data_test/sequences/pt2_BD_Libre_01/')

def remove_bg(points):
    non_zero =  np.where(points[:, 2]>0)
    points_removed_0 = points[non_zero[0]]
    
    z_bg = np.max(points_removed_0[:, 2])
    non_max = np.where((points_removed_0[:, 2] < z_bg - 500))
    z_body = np.median(points_removed_0[non_max, 2])

    
    # print(z_body)
    # z_bg = np.max(points_removed_0[:, 2])
    # print(z_bg - z_body)
    # z_front = np.min(points_removed_0[:, 2])
    non_bg = np.where((points_removed_0[:, 2] < ((z_body + z_bg) / 2.)))
    points_non_bg = points_removed_0[non_bg]
    y_middle = np.mean(points_non_bg[:, 0])
    y_bottom = np.where(points_non_bg[:, 0] < y_middle)
    # print(y_middle)
    # print(y_bottom)
    z_kun = np.min(points_non_bg[y_bottom, 2])
    kun = np.where(points_non_bg[:, 2] == z_kun)
    # print(kun)
    y_kun = points_non_bg[kun, 0][0][0]
   
    # y_kun = np.where(points_non_bg[:, 2] == )
    y_down = np.min(points_non_bg[:, 0])
    # print('y_kun is ', y_kun, 'y_down is ', y_down)
    # non_feet = np.where((points_non_bg[:, 0] > (y_down + 700)))
    non_feet = np.where((points_non_bg[:, 0] > (y_kun - 10)))
    # print(len(points))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_non_bg[non_feet])
    cl, ind = pc.remove_statistical_outlier(nb_neighbors=20,
                                            std_ratio=2.0)
    inlier = pc.select_by_index(ind)
    # print(ind)
    # return ind
    return non_bg[0], non_zero[0], non_feet, ind

def create_path(frame_name):
    info = frame_name.split('_')
    

    if int(info[0][2:]) < 10 and len(info[0][2:]) == 1:
        if 'auto' in frame_name:
            sequence_path = ('Participant0' + info[0][2:] + f'/{info[1]}/Prise{info[2]}/').replace('auto', 'autocorrection')
        else: 
            sequence_path = ('Participant0' + info[0][2:] + f'/{info[1]}/{info[2]}/Prise{info[3]}/').replace('libr/', 'Libre/').replace('libre', 'Libre')

    else:
        if 'auto' in frame_name:
            sequence_path = ('Participant' + info[0][2:] + f'/{info[1]}/Prise{info[2]}/').replace('auto', 'autocorrection')
        else:
            sequence_path = ('Participant' + info[0][2:] + f'/{info[1]}/{info[2]}/Prise{info[3]}/').replace('libr/', 'Libre/').replace('libre', 'Libre')

    frame_path = sequence_path + 'data/' + frame_name + '.npy'
    
    return frame_path

# chatGPT
def save_2D_projection(point_cloud, fx=0, fy=0, cx=0, cy=0, image_height=h, image_width=w, output_file=''):
    # Convert the Open3D point cloud to a NumPy array
    point_cloud_np = np.asarray(point_cloud.points)*[image_height, image_width, 1]
    colors = np.asarray(point_cloud.colors)

    # Create intrinsic matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Create an empty 2D image
    image = np.ones((image_height, image_width, 3), dtype=np.uint8)*255
    min_x = np.min(point_cloud_np[:, 1])
    min_y = np.min(point_cloud_np[:, 0])
    point_cloud_np -= [min_y, min_x, 0]
    # Project each point from the Open3D point cloud onto the 2D image
    for point, color in zip(point_cloud_np, colors):
        X, Y, Z = point

        # Perform projection
        point_3d = np.array([[X], [Y], [Z]])
        point_2d = np.array([[X], [Y]])
        u = int(X)
        v = int(Y)
        # print(u, v)
        # print(image_width, image_height)
        # u, v, _ = (point_2d / point_2d[2]).flatten()

        # Ensure the projected point is within the image bounds
        if 0 <= u < image_width and 0 <= v < image_height:
            # Use color information from the Open3D point cloud
            # color = point_cloud.colors[int(v), int(u)]
            bgr_color = (int(color[2]*255), int(color[1] * 255), int(color[0] * 255))  # OpenCV uses BGR color order
            image[int(v), int(u)] = bgr_color

    # Save the resulting image
    # print(output_file)
    # print(image)
    cv2.imwrite(output_file, image)

# path2 = '/home/travail/ghebr/project/PointNet2/Pointnet_Pointnet2_pytorch/log/back_segmentation/pointnet2_part_seg_msg/10000/predictions/pt8_BD_Libre_01_002199_60.json'
# path = '/home/travail/ghebr/project/PointNet2/Pointnet_Pointnet2_pytorch/log/back_segmentation/pointnet2_part_seg_msg/10000/predictions/pt8_BD_Libre_01_002113_0.json'
# visualize_errors(path=path)
# visualize_segments(path=path2)

def find_segments_and_visualize_sequence(root='/home/travail/ghebr/Data/', split='train', npoints=10000):
    root = root
    npoints = npoints
    split= split
    with open(root + f'/{split}.txt', 'r') as data_file:
        data = data_file.readlines()



    # for thesis visulization
    data = data[233:]
    for frame_name in data:
        frame_name = frame_name.strip('\n')
        info = frame_name.split('_')
        os.makedirs(f'/home/travail/ghebr/Data/annotations2/{info[0]}_{info[1]}_{info[2]}_{info[3]}', exist_ok=True)
        save_path = f'/home/travail/ghebr/Data/annotations2/{info[0]}_{info[1]}_{info[2]}_{info[3]}/' + frame_name + '.jpg'
        path = root + '/' + create_path(frame_name)
        if os.path.exists(save_path):
            continue
        with open(path, 'rb') as frame_file:
            frame = np.load(frame_file, allow_pickle=True).item()
        frame['radius'] = 0.15
        if 'pt19' in frame_name:
            frame['radius'] = 0.08
        points = frame['xyz']
        print(frame['markers'])
        markers = np.array(convert_landmarks_to_tensor(frame['markers']))
        print(len(markers))
        pixel_vals = frame['intensity']
        pixel_vals = np.repeat(pixel_vals, 3).reshape(-1, 3)

        points_and_markers = pc_normalize(np.concatenate((points, markers), axis=0))
        
        points_normalized = points_and_markers[:-len(markers)]
        markers_normalized = points_and_markers[-len(markers):] 
        # print(len(points_and_markers), len(points_normalized), len(markers_normalized))

        downsampled_indices = np.random.choice(len(points_normalized), npoints, replace=True)
        pointset_downsampled = points_normalized[downsampled_indices, :]
        pixel_vals = pixel_vals[downsampled_indices, :]
        print(len(markers_normalized))
        segments_labels, segments_points = find_segments(markers_normalized, pointset_downsampled)
        segments_points.append(markers_normalized)
        # print(len(segments_points))
        visualize_point_cloud(segments_points, save_path=save_path)
        # break



def read_segment_and_visualize_sequence(seq_path, save=False):
    frames  = os.listdir(seq_path)
    for frame in frames:
        path = seq_path + frame
        if os.path.isfile(path):
            save_path = None
            if save:
                save_path = seq_path + f'vis/{frame[:-5]}.jpg'
                os.makedirs(seq_path + 'vis/', exist_ok=True)
            visualize_segments(path, save_path=save_path)



find_segments_and_visualize_sequence()
        
# path = '/home/travail/ghebr/project/seg_results/Participant09_autocorrection_Prise01_data_10000/'
# path_frame = '/home/travail/ghebr/project/seg_results/Participant09_autocorrection_Prise01_data_10000/pt09_auto_01_008298_0.json'

# read_segment_and_visualize_sequence(path, True)
# visualize_errors(path_frame)
# visualize_segments(path_frame)